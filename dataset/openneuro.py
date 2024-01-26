""" Taken and adapted from https://github.com/cyclomon/3dbraingen """

import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
import argparse
import glob
import torchio as tio

def random_vertical_flip(image, mask=None):
    if np.random.rand() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask) if mask is not None else None
        return image, mask
    return image, mask

class OpenNeuroDataset(Dataset):
    def __init__(self, root_dir='../openneuro', extended=False):
        self.root_dir = root_dir
        self.extended = extended
        self.file_names, self.mask_file_names = self.get_data_files()

    def get_data_files(self):
        file_names = []
        mask_file_names = []
        for folder_name in os.listdir(self.root_dir):
            path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(path):
                continue
            for file_name in os.listdir(path):
                if file_name.startswith('sub') and file_name.endswith('.nii'):
                    file_names.append(os.path.join(path, file_name))
                elif file_name.endswith('Segmentation.nii') and self.extended:
                    mask_file_names.append(os.path.join(path, file_name))
        #TODO remove later
        if len(file_names) != len(mask_file_names):
            file_names = file_names[:len(mask_file_names)]
        return file_names, mask_file_names

    def __len__(self):
        return len(self.file_names)
    
    def roi_crop(self, image, values=None):
        image = image[:, :, :]
        
        if values is None:    
            # Mask of non-black pixels (assuming image has a single channel).
            mask = image > 0

            # Coordinates of non-black pixels.
            coords = np.argwhere(mask)

            # Bounding box of non-black pixels.
            x0, y0, z0 = coords.min(axis=0)
            x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top

            # Get the contents of the bounding box.
            cropped = image[x0:x1, y0:y1, z0:z1]

            crop_or_pad = np.max(cropped.shape)
        else:
            x0, x1 = values['x']
            y0, y1 = values['y']
            z0, z1 = values['z']
            cropped = image[x0:x1, y0:y1, z0:z1]
            
            crop_or_pad = values['crop_or_pad']

        padded_crop = tio.CropOrPad(
            crop_or_pad)(cropped.copy()[None])

        return np.squeeze(padded_crop, 0), {'crop_or_pad': crop_or_pad, 'x': (x0, x1), 'y': (y0, y1), 'z': (z0, z1)}

    def __getitem__(self, index):
        image = nib.load(self.file_names[index])
        image = np.asanyarray(image.dataobj)
        image = np.transpose(image, (2, 1, 0))
        image_resized, values = self.roi_crop(image)
        sp_size = 128
        #TODO order of resize and intensity normalization
        #image_resized = resize(image_resized, (sp_size, sp_size, sp_size), mode='constant')
        image_resized = tio.transforms.Resize([sp_size, sp_size, sp_size], image_interpolation='linear')(torch.unsqueeze(torch.Tensor(image_resized), 0)) 
        image_resized = image_resized * 2. /image.max() - 1
        #TODO max pixel value is not 1 anymore after resize with linear interpolation
        if self.extended:
            mask = nib.load(self.mask_file_names[index])
            mask = np.asanyarray(mask.dataobj)
            mask = np.transpose(mask, (2, 1, 0))
            if image.shape != mask.shape:
                print("Shape mismatch for: ", self.file_names[index], " ", image.shape, "x", mask.shape)
                return self.__getitem__(index+1)
            mask_resized, _ = self.roi_crop(mask, values)
            #mask_resized = resize(mask_resized, (sp_size, sp_size, sp_size), mode='constant')
            mask_resized = tio.transforms.Resize([sp_size, sp_size, sp_size], label_interpolation='nearest')(tio.LabelMap(tensor=torch.unsqueeze(torch.Tensor(mask_resized).to(torch.int8), 0))).tensor
            mask_resized = mask_resized * 2. - 1
            image_resized, mask_resized = random_vertical_flip(image_resized, mask_resized)
            return {'data': image_resized, 'target': mask_resized}
        image_resized,_ = random_vertical_flip(image_resized)
        return {'data': image_resized}
