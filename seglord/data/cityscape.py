from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
from glob import glob
from PIL import Image
from torch import Tensor

import torch.nn.functional as F
import numpy as np
import torch
import cv2
import warnings
import albumentations as A


def get_trans_lst():
    return [
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 10.0)),
            A.MultiplicativeNoise(),
            A.RandomRain(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.12, rotate_limit=15, p=0.5,
                          border_mode = cv2.BORDER_CONSTANT),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.ElasticTransform(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),   
            A.Downscale(interpolation = {
                "downscale": cv2.INTER_NEAREST,
                "upscale": cv2.INTER_NEAREST
            }),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(p=0.3),
            A.ColorJitter(p=0.3),
        ], p= 0.3),
        A.RGBShift(p=0.3),
        A.RandomShadow(p=0.2)
    ]

def get_trans_nonnorm_lst():
    return [
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.12, rotate_limit=15, p=0.5,
                          border_mode = cv2.BORDER_CONSTANT),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.ElasticTransform(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),   
            A.Downscale(interpolation = {
                "downscale": cv2.INTER_NEAREST,
                "upscale": cv2.INTER_NEAREST
            }),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(p=0.3),
            A.ColorJitter(p=0.3),
        ], p= 0.3),
        A.RGBShift(p=0.3)
    ]


class CityNormal(Dataset):
    def __init__(self, 
                 root: str, 
                 train: bool, 
                 size: Tuple[int] = (256, 512)
                ) -> None:
        super().__init__()

        self.aug = A.Compose(get_trans_lst(), p = 0.9)
        self.res = A.Compose([A.Resize(size[0], size[1]), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.tr = train
        self.subset = 'train' if train else 'val'

        self.semantic_map = {
            0 : ['unlabeled', 19, 'void'], 
            1 : ['ego vehicle', 19, 'void'],
            2 : ['rectification border', 19, 'void'],
            3 : ['out of roi', 19, 'void'],
            4 : ['static', 19, 'void'],
            5 : ['dynamic', 19, 'void'],
            6 : ['ground', 19, 'void'],
            7 : ['road', 0, 'flat'],
            8 : ['sidewalk', 1, 'flat'],
            9 : ['parking', 19, 'flat'],
            10 : ['rail track', 19, 'flat'],
            11 : ['building', 2, 'construction'],
            12 : ['wall', 3, 'construction'],
            13 : ['fence', 4, 'construction'],
            14 : ['guard rail', 19, 'construction'],
            15 : ['bridge', 19, 'construction'],
            16 : ['tunnel', 19, 'construction'],
            17 : ['pole', 5, 'object'],
            18 : ['polegroup', 19, 'object'],
            19 : ['traffic light', 6, 'object'],
            20 : ['traffic sign', 7, 'object'],
            21 : ['vegetation', 8, 'nature'],
            22 : ['terrain', 9, 'nature'],
            23 : ['sky', 10, 'sky'],
            24 : ['person', 11, 'human'],
            25 : ['rider', 12, 'human'],
            26 : ['car', 13, 'vehicle'],
            27 : ['truck', 14, 'vehicle'],
            28 : ['bus', 15, 'vehicle'],
            29 : ['caravan', 19, 'vehicle'],
            30 : ['trailer', 19, 'vehicle'],
            31 : ['train', 16, 'vehicle'],
            32 : ['motorcycle', 17, 'vehicle'],
            33 : ['bicycle', 18, 'vehicle'],
            34 : ['license plate', -1, 'vehicle']
        }

        self.img_dir = root + f'/leftImg8bit_trainvaltest/leftImg8bit/{self.subset}'
        self.img_paths = glob(self.img_dir + "/*/*")
        self.seg_dir = root + f"/gtFine_trainvaltest/gtFine/{self.subset}"
        self.seg_gts = glob(self.seg_dir + "/*/*_labelIds.png")

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        
        img_path = self.img_paths[index]
        img = np.array(Image.open(img_path).convert("RGB"))

        filename = img_path.split("/")[-1]
        filename_split = filename.split("_")
        city_name = filename_split[0]
        img_name = "_".join(filename_split[:3])

        seg_path = self.seg_dir + f"/{city_name}/{img_name}_gtFine_labelIds.png"
        mask = np.array(Image.open(seg_path))

        if self.tr:
            aug_transformed = self.aug(image=img, masks=[mask])
            transformed = self.res(image = aug_transformed['image'], masks = aug_transformed['masks'])
            transformed_image = transformed['image']
            transformed_masks = transformed['masks']
        else:
            transformed = self.res(image=img, masks=[mask])
            transformed_image = transformed['image']
            transformed_masks = transformed['masks']
        
        mask = self.process_seg(transformed_masks[0])
        img = torch.from_numpy(transformed_image).permute(-1, 0, 1)

        return img, mask

        
    def process_seg(self, x: np.array) -> Tensor:
        x = torch.from_numpy(x).unsqueeze(0)
        encx = torch.zeros(x.shape, dtype=torch.long)
        for label in self.semantic_map:
            encx[x == label] = self.semantic_map[label][1]
        onehot = F.one_hot(encx, 20).permute(0, 3, 1, 2)[0].float()
        return onehot

if __name__ == "__main__":
    ds = CityNormal(
        root='/media/mountHDD3/data_storage/cityscapes/unzip',
        train=True,
        size=(256, 512)
    )

    sample = ds[0]

    img, mask = sample

    print(img.shape, mask.shape, mask.unique())