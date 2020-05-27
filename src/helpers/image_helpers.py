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

   
def normalize_4_channels(img: np.array, mean: list=[0.485, 0.456, 0.406, 0.406], std: list=[0.229, 0.224, 0.225, 0.225], max_value: float=92.88) -> np.array:
    """
    Noramalize image data in 4 channels to 0-1 range,
    then applymenaand std as in ImageNet pretrain, or any other
    """    
    mean = np.array(mean, dtype=np.float32)
    mean *= max_value
    std = np.array(std, dtype=np.float32)
    std *= max_value

    img = img.astype(np.float32)
    img = img - mean    
    img = img / std

    return img


def preprocess_minmax(img: np.array) -> np.array:
    """
    Normalize image data to 0-1 range usinf percentiles
    
    """    
    im_min = np.percentile(img, 2)
    im_max = np.percentile(img, 98)
    im_range = (im_max - im_min)
    #print(f'percentile 2 {im_min}, percentile 98 {im_max}, im_range {im_range}')
    #img = np.clip(img, im_min, im_max)

    # normalise to the percentile
    img = img.astype(np.float32)
    img = (img - im_min) / im_range
    img = img.clip(0, 1)
    
    return img


def plot_mixup_detection(subplots: Tuple[int] = (3, 3), ) -> plt.figure:

    subplot_h, subplot_w = subplots

    fig, ax = plt.subplots(subplot_h, subplot_w, figsize=(16, 6*subplot_h))

    for i in range(subplot_h):
        image, boxes = dataset.load_image_and_boxes(random.randint(0, dataset.image_ids.shape[0] - 1))
        r_image, r_boxes = dataset.load_image_and_boxes(random.randint(0, dataset.image_ids.shape[0] - 1))
        mixup_image = (image+r_image)/2

        for box in boxes.astype(int):
            cv2.rectangle(image,(box[0], box[1]),(box[2],  box[3]),(0, 0, 1), 3)
            cv2.rectangle(mixup_image,(box[0], box[1]),(box[2],  box[3]),(0, 0, 1), 3)
            
        for box in r_boxes.astype(int):
            cv2.rectangle(r_image,(box[0], box[1]),(box[2],  box[3]),(1, 0, 0), 3)
            cv2.rectangle(mixup_image,(box[0], box[1]),(box[2],  box[3]),(1, 0, 0), 3)
            
        ax[i][0].imshow(image)
        ax[i][1].imshow(r_image)
        ax[i][2].imshow(mixup_image)