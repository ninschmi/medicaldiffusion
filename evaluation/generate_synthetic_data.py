import torch
from ddpm import Unet3D, GaussianDiffusion, num_to_groups
from ddpm.unet import UNet
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import nibabel as nib
import imageio
from torchvision import transforms as T
from PIL import Image

@hydra.main(config_path='../config', config_name='data_cfg', version_base=None)
def main(cfg: DictConfig) -> None:
    print("Start Generating Images")
    assert cfg.model.vqgan_ckpt is not None, 'Please provide a path to the VQGAN checkpoint'
    assert cfg.data.diff_ckpt is not None, 'Please provide a path to the DDPM checkpoint'
    
    #parameters
    batch_size = cfg.data.batch_size or 16
    num_samples = cfg.data.num_samples or 10
    #parameters models
    vqgan_ckpt = cfg.model.vqgan_ckpt
    diff_ckpt = cfg.data.diff_ckpt
    diffusion_img_size = cfg.model.diffusion_img_size or 32
    diffusion_depth_size = cfg.model.diffusion_depth_size or 32
    diffusion_num_channels = cfg.model.diffusion_num_channels or 8
    map_location = None

    if not cfg.data.postfix:
        postfix = vqgan_ckpt.split('/')[8]
    else:
        postfix = cfg.data.postfix

    #parameters saving
    save_dir = '/home/oliverbr/ninschmi/synthetic_data/' + cfg.data.dataset + '/' + postfix + '/'
    filename = 'synthetic'
    #check path
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    #set CUDA_VISIBLE_DEVICES
    torch.cuda.set_device(0)

    #denoising function
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
        ).cuda()
    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    #load pretrained generative model
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=vqgan_ckpt,
        image_size=diffusion_img_size,
        num_frames=diffusion_depth_size,
        channels=diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        # objective=cfg.objective,
        extended=True).cuda()
    diff_weights = torch.load(diff_ckpt, map_location=map_location)['ema']
    diffusion.load_state_dict(diff_weights)
    diffusion.eval()
    
    batches = num_to_groups(num_samples, batch_size)

    with torch.no_grad():
        samples = list(map(lambda n: diffusion.sample(batch_size=n), batches)) # 'num_samples c f h w'
        
        all_videos_list = [torch.unsqueeze(elem[:,0], dim=1) for elem in samples]
        all_videos_list = torch.cat(all_videos_list, dim=0)

        all_mask_videos_list = [torch.unsqueeze(elem[:,1], dim=1) for elem in samples]
        all_mask_videos_list = torch.cat(all_mask_videos_list, dim=0)

    #save synthetic data
    for idx in range(len(all_videos_list)):

        if cfg.data.dataset == 'fives':
            video_filename = os.path.join(save_dir, filename + '_image_' + str(idx) + '.png')
            mask_filename = os.path.join(save_dir, filename + '_mask_' + str(idx) + '.png')
            
            image = all_videos_list[idx].squeeze()
            image = ((image - image.min()) / (image.max() - image.min())) * 1.0
            image = T.ToPILImage(mode="L")(image)
            image.save(video_filename)

            mask = all_mask_videos_list[idx].squeeze()
            mask = ((mask - mask.min()) / (mask.max() - mask.min())) * 1.0
            threshold = mask.mean() * 1.2
            mask_bin = (mask > threshold) * 255
            mask = T.ToPILImage(mode="L")(mask)
            mask.save(mask_filename)
            mask_bin = Image.fromarray(mask_bin.cpu().numpy().astype('bool'))
            mask_bin.save(mask_filename.replace('_mask_', '_mask_binary_'))


        else:
            video_filename = os.path.join(save_dir, filename + '_video_' + str(idx) + '.nii')
            mask_filename = os.path.join(save_dir, filename + '_mask_' + str(idx) + '.nii')
            ##save synthetic data as torch tensor directly
            #torch.save(all_videos_list[idx].cpu(), video_filename)
            #torch.save(all_mask_videos_list[idx].cpu(), mask_filename)
            ##only save visulaization for a few samples
            #if idx < save_num_samples:
            #    #VIDEO VISUALIZATION
            video = all_videos_list[idx].permute(1, 2, 3, 0).cpu().numpy()
            video = ((video - video.min()) / (video.max() - video.min())) * 255
            video = (video).astype('uint16')
            #nii
            nib.save(nib.Nifti1Image(video, affine=None), video_filename.replace('.pt', '.nii'))
            #mp4
            video_mp4 = []
            for i in range(video.shape[0]):
                video_mp4.append(video[i])
            imageio.mimsave(video_filename.replace('.pt', '.mp4'), video_mp4, fps=6)

            #    #MASK VISUALIZATION
            mask_video = all_mask_videos_list[idx].permute(1, 2, 3, 0).cpu().numpy()
            mask_video_binary = mask_video > 0
            mask_video = ((mask_video - mask_video.min()) / (mask_video.max() - mask_video.min())) * 255
            mask_video = (mask_video).astype('uint16')
            mask_video_binary = (mask_video_binary * 255).astype('uint16')
            #nii
            nib.save(nib.Nifti1Image(mask_video, affine=None), mask_filename.replace('.pt', '.nii'))
            nib.save(nib.Nifti1Image(mask_video_binary, affine=None), mask_filename.replace('_mask_', '_mask_binary_'))
            #mp4
            mask_video_mp4 = []
            mask_video_binary_mp4 = []
            for i in range(mask_video.shape[0]):
                mask_video_mp4.append(mask_video[i])
                mask_video_binary_mp4.append(mask_video_binary[i])
            imageio.mimsave(mask_filename.replace('.pt', '.mp4'), mask_video_mp4, fps=6)
            imageio.mimsave(mask_filename.replace('_mask_', '_mask_binary_').replace('.pt', '.mp4'), mask_video_binary_mp4, fps=6)

    #save config file
    OmegaConf.save(cfg, os.path.join(save_dir, 'config.yaml'))
    print("Finished Generating Images")

if __name__ == '__main__':
    main()