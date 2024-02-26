# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import numpy as np
from PIL import Image

import torch
import torchvision

from vq_gan_3d.utils import save_video_grid


class ImageLogger():
    def __init__(self, batch_frequency, model, save_dir, max_images, clamp=True, increase_log_steps=True):
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.model = model
        self.save_dir = save_dir.replace('lightning_logs', 'images')
        self.log_steps = [
            2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    #@rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, split)
        for k in images:
            if "mask" in k:
                images[k] = (images[k] + 1.0) / 2.  # binary mask
                images[k] = torch.round(images[k]) * 255
            else:
                images[k] = (images[k] + 1.0) * 127.5  # std + mean
                torch.clamp(images[k], 0, 255)
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = grid
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, batch, batch_idx, global_step, current_epoch, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                self.max_images > 0):

            is_train = self.model.training
            if is_train:
                self.model.eval()

            with torch.no_grad():
                images = self.model.log_images(batch, split=split)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()

            self.log_local(self.save_dir, split, images,
                           global_step, current_epoch, batch_idx)

            if is_train:
                self.model.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False


class VideoLogger():
    def __init__(self, batch_frequency, model, save_dir, max_videos, clamp=True, increase_log_steps=True):
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        self.model = model
        self.save_dir = save_dir.replace('lightning_logs', 'videos')
        self.log_steps = [
            2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    #@rank_zero_only
    def log_local(self, save_dir, split, videos,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, split)
        for k in videos:
            if "mask" in k:
                videos[k] = (videos[k] + 1.0) / 2.  # binary mask
                videos[k] = torch.round(videos[k])
            else:
                videos[k] = (videos[k] + 1.0) * 127.5  # std + mean
                torch.clamp(videos[k], 0, 255)
                videos[k] = videos[k] / 255.0
            grid = videos[k]
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.mp4".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            save_video_grid(grid, path)

    def log_vid(self, batch, batch_idx, global_step, current_epoch, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                self.max_videos > 0):
            # print(batch_idx, self.batch_freq,  self.check_frequency(batch_idx))

            is_train = self.model.training
            if is_train:
                self.model.eval()

            with torch.no_grad():
                videos = self.model.log_videos(
                    batch, split=split)

            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().cpu()

            self.log_local(self.save_dir, split, videos,
                           global_step, current_epoch, batch_idx)

            if is_train:
                self.model.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False
