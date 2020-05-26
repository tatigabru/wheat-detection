import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def pad_x32(image: np.array, **kwargs) -> np.array:
    h, w = image.shape[:2]

    pad_h = np.ceil(h / 32) * 32 - h
    pad_w = np.ceil(w / 32) * 32 - w

    pad_h_top = int(np.floor(pad_h / 2))
    pad_h_bot = int(np.ceil(pad_h / 2))
    pad_w_top = int(np.floor(pad_w / 2))
    pad_w_bot = int(np.ceil(pad_w / 2))

    padding = ((pad_h_top, pad_h_bot), (pad_w_top, pad_w_bot), (0, 0))
    padding = padding[:2] if image.ndim == 2 else padding
    image = np.pad(image, padding, mode='constant', constant_values=0)

    return image

   
def zero_one_normalize(image: np.array) -> np.array:
    """Scale image to range 0..1"""
    x_max = np.percentile(image, 98)
    x_min = np.percentile(image, 2)    
    image = (image - x_min) / (x_max - x_min)
    image = image.clip(0, 1)
    return image

