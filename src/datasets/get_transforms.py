import os
import random
import sys
from typing import List, Optional
import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import label2rgb
from torchvision.transforms import (
    Compose, Normalize, ToTensor)   


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


class TransformsCfgs():

    def __init__(self, img_size: int = 512):
        self.img_size = img_size

    def set_augs(self):
        return set_augmentations(self.img_size)

    def get_augs(self, augs_name: str = "resize"):
        augs_dict = set_augmentations(self.img_size)
        return get_transforms(augs_dict[augs_name])


def set_augmentations(img_size: int = 512):   

    hard_augs = [ 
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 1.5), ratio=(0.75, 1.25), p=0.5),            
                #A.RandomSizedCrop(min_max_height=(512, 1024), height=img_size, width=img_size, p=0.5),
                A.Resize(height=img_size, width=img_size, p=1.0),
                # Add occasion blur
                A.OneOf([A.GaussianBlur(), A.MotionBlur()]),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=0, p = 0.5), 
                # noise                
                A.OneOf([
                        A.GaussNoise(p=0.5),                 
                        A.RandomGamma(p=0.4),                    
                        ],p=0.5),   
                # D4 Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.2),
                # Spatial-preserving augmentations
                A.RandomBrightnessContrast(brightness_by_max=True, p=0.8),
                A.HueSaturationValue(p=0.7),
                # cutout
                A.Cutout(20, 20, 20, p=0.5), 
            ]   

    medium_augs = [
            A.RandomSizedCrop(min_max_height=(512, 1024), height=img_size, width=img_size, p=0.5),
            A.Resize(height=img_size, width=img_size, p=1.0),
            A.OneOf([
                    A.HueSaturationValue(p=0.9),
                    A.RandomBrightnessContrast(p=0.9),
                ],p=0.5),                  
            # noise                
            A.OneOf([
                    A.GaussNoise(p=0.5),                 
                    A.RandomGamma(p=0.4),                    
                    ],p=0.5),
            # D4 transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),                
            ]   

    light_augs = [
            A.RandomSizedCrop(min_max_height=(800, 1024), height=img_size, width=img_size, p=0.5),
            A.Resize(height=img_size, width=img_size, p=1),
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
            ]

    d4_augs = [
            # D4 augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),                        
            ]

    resize = [A.Resize(height=img_size, width=img_size, p=1.0)]

    # Cutout,p=0.5
    cutout = [A.Cutout(num_holes=16, max_h_size=img_size // 16, max_w_size=img_size // 16, fill_value=0, p=0.5)]

    # Weather effects                    
    weather = [
            A.OneOf(
                    [ A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.2), 
                      A.RandomRain( p=0.2),
                    ], p=0.5),
            ]

    hard_cutout = hard_augs.append(cutout) 

    # dictionary of transforms
    transforms_dict = {
        "d4": d4_augs,
        "hard": hard_augs,
        "medium": medium_augs,
        "light": light_augs,
        "resize": resize,   
        "cutout": cutout,  
        "weather": weather,
        "hard_cutout": hard_augs,
        }

    return transforms_dict  


def get_train_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                 val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.9),
        ], p=0.9),
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=img_size, width=img_size, p=1),
        A.Cutout(num_holes=8, max_h_size=img_size // 8, max_w_size=img_size // 8, fill_value=0, p=0.5)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size, p=1)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


