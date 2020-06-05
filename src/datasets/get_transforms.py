import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import label2rgb
from torchvision.transforms import (
    Compose, Normalize, ToTensor)   
from .. configs import IMG_SIZE


def get_transforms(augs: List):
    return A.Compose(
        augs, 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    ) 


hard_augs = [             
                    A.RandomSizedBBoxSafeCrop(height = IMG_SIZE, width = IMG_SIZE, erosion_rate=0.0, interpolation=1, always_apply=False, p=1.0),
                    # Add occasion blur
                    A.OneOf([A.GaussianBlur(), A.GaussNoise(), A.NoOp()]),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=0, p = 0.5),    
                    # D4 Augmentations
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.2),
                    # Cutout,p=0.5
                    A.Cutout(num_holes=8, max_h_size=IMG_SIZE // 8, max_w_size=IMG_SIZE // 8, fill_value=0, p=0.5),
                    # Spatial-preserving augmentations:
                    A.OneOf(
                        [   A.RandomBrightnessContrast(brightness_by_max=True),
                            A.HueSaturationValue(),
                            A.RGBShift(),
                            A.RandomGamma(),                            
                        ], p=0.9
                    ),
                    # Weather effects                    
                    A.OneOf(
                        [ A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.2), 
                          A.RandomRain( p=0.2),
                        ]
                    ),                 
            ]   


medium_augs = [
            A.RandomSizedCrop(min_max_height=(800, 800), height=IMG_SIZE, width=IMG_SIZE, p=0.5),
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
                  
            # noise                
            A.OneOf([
                    A.GaussNoise(p=0.5),                 
                    A.RandomGamma(p=0.4),                    
                ],p=0.7),

            # D4 transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),

            # Cutout, p=0.5
            A.Cutout(num_holes=8, max_h_size=IMG_SIZE // 8, max_w_size=IMG_SIZE // 8, fill_value=0, p=0.5),
        ]   


light_augs = [
        A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                 val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.9),
        ], p=0.9),
        A.ToGray(p=0.01),

        # D4 transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Cutout,p=0.5
        A.Cutout(num_holes=8, max_h_size=IMG_SIZE // 8, max_w_size=IMG_SIZE // 8, fill_value=0, p=0.5)
        ]


d4_augs = [
        # D4 augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),                        
		]

resize = A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0)

# dictionary of transforms
TRANSFORMS = {
    "d4": d4_augs,
    "hard": hard_augs,
    "medium": medium_augs,
    "light": light_augs,
    "resize": resize,     
}

