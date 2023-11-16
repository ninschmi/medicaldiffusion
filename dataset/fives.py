from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse
from torchvision.transforms import RandomCrop


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.CropOrPad(target_shape=(2048, 2048, 1))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
    tio.transforms.Resize([1024,1024,1])
])

CROP_TRANSFORM = RandomCrop((128,128))


class FIVESDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        self.cropping = CROP_TRANSFORM
        self.file_paths = self.get_data_files()

    def get_data_files(self):
        folder_name = os.path.join(self.root_dir, "Original/")
        file_names = [os.path.join(
            folder_name, file_name) for file_name in os.listdir(folder_name) if file_name.endswith('.png')]
        return file_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(self.file_paths[idx])
        img = self.preprocessing(img)
        img = self.transforms(img)
        img_cropped = self.cropping(img.data.permute(0, -1, 1, 2).squeeze(dim=1))
        return {'data': img_cropped}
        #return {'data': img.data.permute(0, -1, 1, 2).squeeze(dim=1)}
