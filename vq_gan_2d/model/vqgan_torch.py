"""Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vq_gan_2d.utils import shift_dim, adopt_weight, comp_getattr
from vq_gan_2d.model.lpips import LPIPS
from vq_gan_2d.model.codebook import Codebook

def main_params(optimizer):
    for group in optimizer.param_groups:
        yield from group["params"]


def silu(x):
    return x*torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQGAN2D_torch(nn.Module):
    def __init__(self, cfg, logger):
        super().__init__()
        self.cfg = cfg
        self.extended = cfg.model.extended
        self.embedding_dim = cfg.model.embedding_dim
        self.n_codes = cfg.model.n_codes

        self.logger = logger

        self.encoder = Encoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.padding_type,
                               cfg.model.num_groups,
                               )
        self.decoder = Decoder(
            cfg.model.n_hiddens, cfg.model.downsample, cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.num_groups,
            self.extended)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv2d(
            self.enc_out_ch, cfg.model.embedding_dim, 1, padding_type=cfg.model.padding_type)
        self.post_vq_conv = SamePadConv2d(
            cfg.model.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
                                 no_random_restart=cfg.model.no_random_restart, restart_thres=cfg.model.restart_thres)

        self.gan_feat_weight = cfg.model.gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            cfg.dataset.image_channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm2d)
        
        if self.extended:
            self.mask_discriminator = NLayerDiscriminator(
                cfg.dataset.image_channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm2d)

        if cfg.model.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif cfg.model.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = cfg.model.image_gan_weight

        self.perceptual_weight = cfg.model.perceptual_weight

        self.l1_weight = cfg.model.l1_weight
        #TODO save hyperparameters
        #self.save_hyperparameters()

        self.optimizers = self.configure_optimizers()

        self.accumulate_grad_batches = cfg.model.accumulate_grad_batches
        self.gradient_clip_val = cfg.model.gradient_clip_val
        self.automatic_optimization=False

        self.sync_dist = False
        if cfg.model.gpus > 1:
            self.sync_dist = True

    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x, mask=None, optimizer_idx=None, log_image=False):
        B, C, H, W = x.shape

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = torch.unsqueeze(self.decoder(self.post_vq_conv(vq_output['embeddings']))[:,0], dim=1)

        recon_loss_image = F.l1_loss(x_recon, x) * self.l1_weight
        recon_loss = recon_loss_image.clone()

        # Selects one random 2D image from each 3D Image
       
        frames = x
        frames_recon = x_recon

        mask_recon = None
        if self.extended:
            mask_recon = torch.unsqueeze(self.decoder(self.post_vq_conv(vq_output['embeddings']))[:,1], dim=1)

            recon_loss_mask = F.l1_loss(mask_recon, mask) * self.l1_weight
            recon_loss += recon_loss_mask.clone()
        
            frames_mask = mask
            frames_recon_mask = mask_recon

        if log_image:
            return x, x_recon, mask, mask_recon

        if optimizer_idx == 0:
            # Autoencoder - train the "generator"

            # Perceptual loss
            perceptual_loss = perceptual_loss_image = perceptual_loss_mask = 0
            if self.perceptual_weight > 0:
                perceptual_loss_image = self.perceptual_model(
                    frames, frames_recon).mean() * self.perceptual_weight
                perceptual_loss = perceptual_loss_image.clone()
                if self.extended:
                    perceptual_loss_mask = self.perceptual_model(
                    frames_mask, frames_recon_mask).mean() * self.perceptual_weight
                    perceptual_loss += perceptual_loss_mask.clone()

            # Discriminator loss (turned on after a certain epoch)
            logits_image_fake, pred_image_fake = self.image_discriminator(
                frames_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_loss = self.image_gan_weight*g_image_loss
            disc_factor = adopt_weight(
                self.optimizer_step, threshold=self.cfg.model.discriminator_iter_start)
            aeloss_image = disc_factor * g_loss
            aeloss = aeloss_image.clone()

            if self.extended:
                logits_mask_fake, pred_mask_fake = self.mask_discriminator(
                frames_recon_mask)
                g_mask_loss = -torch.mean(logits_mask_fake)
                g_loss_mask = self.image_gan_weight*g_mask_loss
                aeloss_mask = disc_factor * g_loss_mask
                aeloss += aeloss_mask.clone()

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            image_gan_feat_loss = 0
            mask_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(
                    frames)
                if self.extended:
                    logits_mask_real, pred_mask_real = self.mask_discriminator(
                    frames_mask)
                for i in range(len(pred_image_fake)-1):
                    image_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_image_fake[i], pred_image_real[i].detach(
                        )) * (self.image_gan_weight > 0)
                    if self.extended:
                        mask_gan_feat_loss += feat_weights * \
                            F.l1_loss(pred_mask_fake[i], pred_mask_real[i].detach(
                            )) * (self.image_gan_weight > 0)
            gan_feat_loss = disc_factor * self.gan_feat_weight * \
                (image_gan_feat_loss + mask_gan_feat_loss)

            self.logger.log("train/g_image_loss", g_image_loss, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            self.logger.log("train/image_gan_feat_loss", image_gan_feat_loss, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            self.logger.log("train/perceptual_loss", perceptual_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            self.logger.log("train/recon_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            self.logger.log("train/aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            self.logger.log("train/commitment_loss", vq_output['commitment_loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            self.logger.log('train/perplexity', vq_output['perplexity'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            if self.extended:
                self.logger.log("train/g_mask_loss", g_mask_loss, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/mask_gan_feat_loss", mask_gan_feat_loss, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/perceptual_loss_image", perceptual_loss_image, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/perceptual_loss_mask", perceptual_loss_mask, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/recon_loss_image", recon_loss_image, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/recon_loss_mask", recon_loss_mask, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/aeloss_image", aeloss_image, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/aeloss_mask", aeloss_mask, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss

        if optimizer_idx == 1:
            # Train discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach())

            logits_image_fake, _ = self.image_discriminator(
                frames_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            disc_factor = adopt_weight(
                self.optimizer_step, threshold=self.cfg.model.discriminator_iter_start)
            discloss_image = disc_factor * \
                (self.image_gan_weight*d_image_loss)
            discloss = discloss_image.clone()

            self.logger.log("train/logits_image_real", logits_image_real.mean().detach(), on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            self.logger.log("train/logits_image_fake", logits_image_fake.mean().detach(), on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            self.logger.log("train/d_image_loss", d_image_loss, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            self.logger.log("train/discloss", discloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
            if self.extended:
                logits_mask_real, _ = self.mask_discriminator(frames_mask.detach())

                logits_mask_fake, _ = self.mask_discriminator(
                frames_recon_mask.detach())

                d_mask_loss = self.disc_loss(logits_mask_real, logits_mask_fake)
                discloss_mask = disc_factor * \
                    (self.image_gan_weight*d_mask_loss)
                discloss += discloss_mask.clone()

                self.logger.log("train/logits_mask_real", logits_mask_real.mean().detach(), on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/logits_mask_fake", logits_mask_fake.mean().detach(), on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/d_mask_loss", d_mask_loss, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/discloss_image", discloss_image, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                self.logger.log("train/discloss_mask", discloss_mask, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
                return discloss
            return discloss

        perceptual_loss_image = self.perceptual_model(
            frames, frames_recon) * self.perceptual_weight
        perceptual_loss = perceptual_loss_image.clone()
        if self.extended:
            perceptual_loss_mask = self.perceptual_model(
                frames_mask, frames_recon_mask) * self.perceptual_weight
            perceptual_loss += perceptual_loss_mask.clone()
            return recon_loss, x_recon, vq_output, perceptual_loss, recon_loss_image, recon_loss_mask, perceptual_loss_image, perceptual_loss_mask
        return recon_loss, x_recon, vq_output, perceptual_loss

    def training_step(self, batch, epoch, step):
        opt_ae, opt_disc = self.optimizers
        self.epoch = epoch
        self.step = step
        self.optimizer_step = step * 2
        x = batch['data']
        y = None
        if self.extended:
            y = batch['target'].to(torch.float32)

        ### Optimize Discriminator
        discloss = self.forward(x, y, 1)

        opt_disc.zero_grad()
        discloss.backward()
        torch.nn.utils.clip_grad_norm_(main_params(opt_disc), self.gradient_clip_val)
        opt_disc.step()
        
        ## scale losses by 1/N (for N batches of gradient accumulation)
        #discloss_scaled = discloss / self.accumulate_grad_batches
        #discloss_scaled.backward()
        ## accumulate gradients of N batches
        #if (batch_idx + 1) % self.accumulate_grad_batches == 0:
        #   self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
        #   opt_disc.step()
        #   opt_disc.zero_grad()

        
        ### Optimize Autoencoder - the "generator"

        recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(
            x, y, 0)
        commitment_loss = vq_output['commitment_loss']
        loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss

        opt_ae.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params(opt_ae), self.gradient_clip_val)
        opt_ae.step()

        ## scale losses by 1/N (for N batches of gradient accumulation)
        #loss_scaled = loss / self.accumulate_grad_batches
        #loss_scaled.backward()
        ## accumulate gradients of N batches
        #if (batch_idx + 1) % self.accumulate_grad_batches == 0:
        #   self.clip_gradients(opt_ae, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
        #   opt_ae.step()
        #   opt_ae.zero_grad()
        #self.logger.log_dict({"loss": loss, "loss_scaled": loss_scaled, "disc_loss": discloss, "disc_loss_scaled": discloss_scaled}, prog_bar=True)
        self.logger.log_dict({"loss": loss, "disc_loss": discloss}, prog_bar=True)

    def validation_step(self, batch):
        x = batch['data']  # TODO: batch['stft']
        y = None
        if self.extended:
            y = batch['target'].to(torch.float32)
            recon_loss, _, vq_output, perceptual_loss, recon_loss_image, recon_loss_mask, perceptual_loss_image, perceptual_loss_mask = self.forward(x, y)
            self.logger.log('val/recon_loss_image', recon_loss_image, prog_bar=True, sync_dist=self.sync_dist)
            self.logger.log('val/recon_loss_mask', recon_loss_mask, prog_bar=True, sync_dist=self.sync_dist)
            self.logger.log('val/perceptual_loss_image', perceptual_loss_image.mean(), prog_bar=True, sync_dist=self.sync_dist)
            self.logger.log('val/perceptual_loss_mask', perceptual_loss_mask.mean(), prog_bar=True, sync_dist=self.sync_dist)
        else:
            recon_loss, _, vq_output, perceptual_loss = self.forward(x)

        self.logger.log('val/recon_loss', recon_loss, prog_bar=True, sync_dist=self.sync_dist)
        self.logger.log('val/perceptual_loss', perceptual_loss.mean(), prog_bar=True, sync_dist=self.sync_dist)
        self.logger.log('val/perplexity', vq_output['perplexity'], prog_bar=True, sync_dist=self.sync_dist)
        self.logger.log('val/commitment_loss',
                 vq_output['commitment_loss'], prog_bar=True, sync_dist=self.sync_dist)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.pre_vq_conv.parameters()) +
                                  list(self.post_vq_conv.parameters()) +
                                  list(self.codebook.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc_param_list = list(self.image_discriminator.parameters())
                                    
        if self.extended:
            opt_disc_param_list = opt_disc_param_list + list(self.mask_discriminator.parameters())
        opt_disc = torch.optim.Adam(opt_disc_param_list , lr=lr, betas=(0.5, 0.9))
        return opt_ae, opt_disc

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch['data']
        x = x
        y = None
        if self.extended:
            y = batch['target'].to(torch.float32)
        frames, frames_rec, mask, mask_rec = self(x, y, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        if mask is not None and mask_rec is not None:
            log["inputs_mask"] = mask
            log["reconstructions_mask"] = mask_rec
        #log['mean_org'] = batch['mean_org']
        #log['std_org'] = batch['std_org']
        return log

def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv2d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv2d(
                in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32, extended=False):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        in_channels = n_hiddens*2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose2d(
                in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv2d(
            out_channels, image_channel + extended, kernel_size=3)

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv2d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv2d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv2d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h


# Does not support dilation
class SamePadConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _