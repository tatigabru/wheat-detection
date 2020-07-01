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
from typing import Tuple, List, Optional


class TransfromsCfgs():
    """
    Transforms configuration from albumentations libruary

    Args:
        img_size [tuple] = (512, 512): resulting image size after transforms,   
        augs_name [str] = "valid": augs set name, if defined,      
        shift_limit [float]=0.0625, 
        scale_limit [float]=0.3, 
        rotate_limit [int]=7,   
        crop_size [tuple] = (512, 512),    
        hflip: bool = True,
        vflip:bool = False,  
        normalise: bool = True,

    """
    def __init__(
        self,
        img_size: tuple = (512, 512),   
        augs_name: Optional[str] = "valid",      
        shift_limit: float=0.0625, 
        scale_limit: float=0.3, 
        rotate_limit: int=7,   
        crop_size: tuple = (512, 512),    
        hflip: bool = True,
        vflip:bool = False,  
        normalise: bool = True,
    ):
        super(TransfromsCfgs, self).__init__()  # inherit it from torch Dataset
        self.img_size = img_size
        self.augs_name = augs_name
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.crop_size = crop_size
        self.hflip = hflip
        self.vflip = vflip
        self.use_d4 = use_d4
        self.probs = probs
        self.normalise = normalise

    def set_transforms(self) -> dict:       
        
        d4_tansforms = [
                        A.SmallestMaxSize(self.img_size[0], interpolation=0, p=1.),
                        # D4 Group augmentations
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Transpose(p=0.5),                       
			            ]

        tensor_norm  =  [
                        ToTensor(),
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]

        geometric = [
                        A.SmallestMaxSize(self.img_size[0], interpolation=0, p=1.),
                        A.ShiftScaleRotate(shift_limit=self.shift_limit, self.scale_limit, rotate_limit=self.rotate_limit, 
                                interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                        # crop and resize  
                        A.RandomSizedCrop((self.crop_size[0], min(self.crop_size[1], self.img_size[0], self.img_size[1])), 
                                            self.img_size[0], self.img_size[1], w2h_ratio=1.0, 
                                            interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.2),                        
                        ]

        resize_crop =[
                    A.SmallestMaxSize(max(self.img_size[0], self.img_size[1]), interpolation=0, p=1.),
                    A.RandomCrop(self.img_size[0], self.img_size[1], p=1.),                     
                    ]        

        train_light = [     
                        A.SmallestMaxSize(self.img_size[0], interpolation=0, p=1.),
                        A.RandomCrop(self.img_size[0], self.img_size[1], p=1.),
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.5),                            
                        ]

        train_medium = train_light.append([
                        A.ShiftScaleRotate(shift_limit=self.shift_limit, self.scale_limit, rotate_limit=self.rotate_limit, p=0.5),
                        A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=0.7),
                            A.Equalize(p=0.3),
                            A.HueSaturationValue(p=0.5),
                            A.RGBShift(p=0.5),
                            A.RandomGamma(p=0.4),
                            A.ChannelShuffle(p=0.05),
                        ],
                        p=0.9),
                    A.OneOf([
                        A.GaussNoise(p=0.5),
                        A.ISONoise(p=0.5),
                        A.MultiplicativeNoise(0.5),
                    ], p=0.2),                           
                    ])

        valid_ade = [
                    A.SmallestMaxSize(self.img_size, p=1.),
                    A.Lambda(name="Pad32", image=pad_x32, mask=pad_x32),                            
                    ]       

        # from bloodaxe 
        # https://github.com/BloodAxe/Catalyst-Inria-Segmentation-Example/blob/master/inria/augmentations.py
        crop = [ #(image_size: Tuple[int, int], min_scale=0.75, max_scale=1.25, input_size=5000):
                        A.OneOrOther(
                        A.RandomSizedCrop((self.crop_size[0], min(self.crop_size[1], self.img_size[0], self.img_size[1])), 
                                        self.img_size[0], self.img_size[1]), 
                        A.CropNonEmptyMaskIfExists(self.img_size[0], self.img_size[1]),
                    ) 
                    ]

        safe_augmentations =[A.HorizontalFlip(), A.RandomBrightnessContrast()]

        light_augmentations = [
                    A.HorizontalFlip(),
                    A.RandomBrightnessContrast(),
                    A.OneOf([
                        A.ShiftScaleRotate(scale_limit=self.scale_limit, rotate_limit=self.rotate_limit, border_mode=cv2.BORDER_CONSTANT),
                        A.IAAAffine(),
                        A.IAAPerspective(),
                        A.NoOp()
                    ]),
                    A.HueSaturationValue(),                
                    ]

        medium_augmentations = [
                        A.HorizontalFlip(),
                        A.ShiftScaleRotate(scale_limit=self.scale_limit, rotate_limit=self.rotate_limit, border_mode=cv2.BORDER_CONSTANT),
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
                    A.RandomRotate90(),
                    A.Transpose(),
                    A.RandomGridShuffle(),
                    A.ShiftScaleRotate(
                        scale_limit=self.scale_limit, rotate_limit=self.rotate_limit, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0
                    ),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, alpha_affine=5, mask_value=0, value=0),
                    # Add occasion blur
                    A.OneOf([A.GaussianBlur(), A.GaussNoise(), A.IAAAdditiveGaussianNoise(), A.NoOp()]),
                    # D4 Augmentations
                    A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=10), A.NoOp()]),
                    # Spatial-preserving augmentations:
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(brightness_by_max=True),                            
                            A.HueSaturationValue(),
                            A.RGBShift(),
                            A.RandomGamma(),
                            A.NoOp(),
                        ]
                    ),
                    # Weather effects
                    A.OneOf([A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1), A.NoOp()]),                    
            ]

        TRANSFORMS = {
                "d4": d4_tansforms,                
                "resize_crop": resize_crop,
                "geometric": geometric,                
                "light": train_light,
                "medium": train_medium,
                "ade_valid": valid_ade,                
                "flip_bright": safe_augmentations,
                "crop": crop,
                "inria_light": light_augmentations,
                "inria_medium": medium_augmentations,
                "inria_hard": hard_augmentations,
                "inria_valid": safe_augmentations,
                }                   

        return TRANSFORMS

        def get_transforms(self):

            augs = []
            if self.augs_name:
                augs = self.set_transforms[self.augs_name]

            augs.append(A.Resize(height=self.img_size[0], width=self.img_size[1], p=1.0))

            if self.use_d4:  
                augs.append(self.set_transforms["d4"])   

            if self.vflip:
                augs.append(A.VerticalFlip(p=0.5))
            if self.hflip: 
                augs.append(A.HorizontalFlip(p=0.5))       
            
            if self.normalise: 
                augs.append(A.Normalize())  

            return A.Compose(augs)     