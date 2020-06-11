"""
Dictionary of the commonly-used image transforms
from albumentations libruary

"""
import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import (
    Compose, Normalize, ToTensor)    
from .. configs import IMG_SIZE


d4_geometric = [                       
			    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=0, 
                    interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                # D4 Group augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.2),
                # crop and resize  
                A.RandomSizedCrop((IMG_SIZE-100, IMG_SIZE), IMG_SIZE, IMG_SIZE, w2h_ratio=1.0, 
                                interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.2),                   	      	
                ]


d4_tansforms = [# D4 Group augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),                        
			    ]

tensor_norm = [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]


normalise = A.Normalize()


resize_norm = [     A.SmallestMaxSize(IMG_SIZE, interpolation=0, p=1.),
                    A.RandomCrop(IMG_SIZE, IMG_SIZE, p=1.), 
                    A.Normalize(),
                    ]
        

geometric_transforms = [                    
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, 
                       interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                    A.RandomSizedCrop((int(0.5*IMG_SIZE), IMG_SIZE), IMG_SIZE, IMG_SIZE),
                    # D4 Group augmentations
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.2),                    
                    ]


hflip_brightness = [                    
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),                    
                    ]
                    

train_medium = [ 
            A.ShiftScaleRotate(shift_limit=0., scale_limit=0.2, rotate_limit=5, p = 0.5),                           
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.7),
                    A.GaussNoise(p=0.5),                 
                    A.RandomGamma(p=0.4),                    
                ],
                p=0.7),
            # D4 Group augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),                  
        ]


valid_ade = [
            A.SmallestMaxSize(IMG_SIZE, p=1.),
            A.Lambda(name="Pad32", image=pad_x32, mask=pad_x32),   
            A.Normalize(),         
        ]


# from bloodaxe 
# https://github.com/BloodAxe/Catalyst-Inria-Segmentation-Example/blob/master/inria/augmentations.py
crop_transform = [A.RandomSizedCrop((int(0.5*IMG_SIZE), IMG_SIZE), IMG_SIZE, IMG_SIZE)]


safe_augmentations = [A.HorizontalFlip(), A.RandomBrightnessContrast()]

light_augmentations = [
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.OneOf([
                    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=7, border_mode=cv2.BORDER_CONSTANT),
                    A.IAAAffine(),
                    A.IAAPerspective(),
                    A.NoOp()
                ]),
                A.HueSaturationValue(),                
            ]


medium_augmentations = [
                    A.HorizontalFlip(),
                    A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=7, border_mode=cv2.BORDER_CONSTANT),
                    # Add occasion blur/sharpening
                    A.OneOf([A.GaussianBlur(), A.IAASharpen(), A.NoOp()]),
                    # Spatial-preserving augmentations:
                    A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=5), A.NoOp()]),
                    A.GaussNoise(),
                    A.OneOf([A.RandomBrightnessContrast(), A.CLAHE(), A.HueSaturationValue(), A.RGBShift(), A.RandomGamma()]),
                    # Weather effects
                    A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),                   
            ]


hard_augmentations = [                    
                    A.RandomGridShuffle(),
                    A.ShiftScaleRotate(
                        scale_limit=0.3, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0
                    ),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, alpha_affine=5, mask_value=0, value=0),
                    # Add occasion blur
                    A.OneOf([A.GaussianBlur(), A.GaussNoise(), A.IAAAdditiveGaussianNoise(), A.NoOp()]),
                    # D4 Augmentations
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.2),

                    A.OneOf([A.CoarseDropout()),
                    # Spatial-preserving augmentations:
                    A.OneOf(
                        [   A.RandomBrightnessContrast(brightness_by_max=True),                            
                            A.HueSaturationValue(),
                            A.RGBShift(),
                            A.RandomGamma(),                            
                        ]
                    ),
                    # Weather effects                    
                    A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),                    
            ]   


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


resize = A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0)

# dictionary of transforms
TRANSFORMS = {
    "d4": d4_tansforms,
    "normalise": normalise,
    "resize_norm": resize_norm,
    "geometric": geometric_transforms,
    "d4_geometric": d4_geometric,
    "ade_light": train_light,
    "ade_medium": train_medium, 
    "ade_valid": valid_ade,     
    "tensor_norm": tensor_transform,
    "inria_safe": safe_augmentations,
    "inria_light": light_augmentations,
    "inria_medium": medium_augmentations,
    "inria_hard": hard_augmentations,
    "inria_valid": safe_augmentations,
    "hard": hard_augs,
    "medium": medium_augs,
    "light": light_augs,
    "resize": resize,   
}
