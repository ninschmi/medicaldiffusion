from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import crop, pil_to_tensor


PREPROCESSING_TRANSFORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.CropOrPad(target_shape=(2048, 2048, 1))
])

EVAL_PREPROCESSING_TRANSFORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.CropOrPad(target_shape=(2048, 2048, 1))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
    tio.transforms.Resize([512,512,1])    #previous[1024,1024,1]
])

VALID_TRAIN_TRANSFORMS = tio.Compose([
    tio.transforms.Resize([512,512,1])    #previous[1024,1024,1]
])

CROP_TRANSFORM = RandomCrop((64,64))


class FIVESDataset(Dataset):
    def __init__(self, root_dir: str, extended=False, eval=False, valid=False, synthetic=False):
        super().__init__()
        self.root_dir = root_dir
        self.extended = extended
        self.eval = eval
        self.valid = valid
        self.synthetic = synthetic
        if self.eval:
            self.preprocessing = EVAL_PREPROCESSING_TRANSFORMS
        else:
            self.preprocessing = PREPROCESSING_TRANSFORMS
        if self.valid:
            self.transforms = VALID_TRAIN_TRANSFORMS
        else:
            self.transforms = TRAIN_TRANSFORMS
        if self.synthetic:
            self.preprocessing = tio.RescaleIntensity(out_min_max=(0, 1)) if self.eval \
                else tio.RescaleIntensity(out_min_max=(-1, 1))
            self.transforms = tio.RandomFlip(axes=(1), flip_probability=0) if self.valid \
                else tio.RandomFlip(axes=(1), flip_probability=0.5)
        
        self.cropping = CROP_TRANSFORM
        if self.extended:
            self.file_paths, self.mask_file_paths = self.get_data_files()
        else:
            self.file_paths = self.get_data_files()

    def get_data_files(self):
        folder_name = os.path.join(self.root_dir, "Original/")
        file_names = [os.path.join(
            folder_name, file_name) for file_name in os.listdir(folder_name) if file_name.endswith('.png')]
        if self.extended:
            mask_folder_name = folder_name.replace('Original', 'Ground truth')
            mask_file_names = [os.path.join(
            mask_folder_name, file_name) for file_name in os.listdir(folder_name) if file_name.endswith('.png')]
            return file_names, mask_file_names
        return file_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(self.file_paths[idx])
        img = self.preprocessing(img)
        if self.extended:
            mask = tio.LabelMap(self.mask_file_paths[idx]) # 0 or 255 pixelvalues
            if self.eval:
                mask.data = mask.data * 1. / 255.
            else:
                mask.data = mask.data * 2. / 255. - 1
            mask = self.preprocessing(mask)
            subject = tio.Subject(image=img, seg=mask)
            subject = self.transforms(subject)
            #self.sampler = tio.data.LabelSampler(patch_size=(64,64,1), label_name="seg")
            #patches = self.sampler(subject=subject, num_patches=1)
            #patch = [patch for patch in patches]
            #patch = patch[0]
            #return {'data': patch['image'].data.permute(0, -1, 1, 2).squeeze(dim=1), 'target': patch['seg'].data.permute(0, -1, 1, 2).squeeze(dim=1)}
            
            # with cropping
            #i, j, h, w = RandomCrop.get_params(subject['image'].as_pil(), output_size=(64, 64))
            #img_cropped = crop(subject['image'].data.permute(0, -1, 1, 2).squeeze(dim=1), i, j, h, w)
            #mask_cropped = crop(subject['seg'].data.permute(0, -1, 1, 2).squeeze(dim=1), i, j, h, w)
            ##img_cropped = self.cropping(subject['image'].data.permute(0, -1, 1, 2).squeeze(dim=1))
            ##mask_cropped = self.cropping(subject['seg'].data.permute(0, -1, 1, 2).squeeze(dim=1))
            #return {'data': img_cropped, 'target': mask_cropped}
            # without cropping
            if self.synthetic:
                subject['image'].data = subject['image'].data.permute(0, 2, 1, 3)
                subject['seg'].data = subject['seg'].data.permute(0, 2, 1, 3)
            return {'data': subject['image'].data.permute(0, -1, 1, 2).squeeze(dim=1), 'target': subject['seg'].data.permute(0, -1, 1, 2).squeeze(dim=1)}
        else:
            img = self.transforms(img)
            img_cropped = self.cropping(img.data.permute(0, -1, 1, 2).squeeze(dim=1))

            return {'data': img_cropped}
            #return {'data': img.data.permute(0, -1, 1, 2).squeeze(dim=1)}
