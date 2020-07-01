import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

from .get_transforms import TransformsCfgs, set_augmentations, get_transforms


class WheatDataset(Dataset):
    """
    Wheat Dataset

    Args:         
        images_dir: directory with images        
        labels_df: Dataframe with 
        img_size: the desired image size to resize to for prograssive learning
        transforms: the name of transforms set from the transfroms dictionary  
        debug: if True, runs debugging on a few images. Default: 'False'   
        normalise: if True, normalise images. Default: 'True'

    """
    def __init__(self, 
                image_ids: list,
                image_dir: str,  
                box_callback: ,                       
                img_size: int = 512,                 
                transforms: A.Compose, 
                is_test: bool = False,
                use_cutmix: bool = True,
                use_mixup: bool = False,                
                normalise: bool = True,                         
                debug: bool = False,     
                ):

        super().__init__()
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.box_callback = box_callback
        self.transforms = transforms
        self.is_test = is_test
        self.use_cutmix = use_cutmix
        self.use_mixup = use_mixup
        self.normalise = normalise
        # select a subset for the debugging
        if debug:
            self.image_ids = self.image_ids[:160]
            print('Debug mode, samples: ', self.image_ids[:10])  

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        if self.is_test or random.random() > 0.5:
            print('use load_image_and_boxes')
            image, boxes = self.load_image_and_boxes(index)
            self.use_cutmix = not self.is_test or not len(boxes) > 0
        if self.use_cutmix:
            print('use load_cutmix_image_and_boxes')
            image, boxes = self.load_cutmix_image_and_boxes(index)
        if len(boxes) == 0:
            pass  

        n_boxes = len(boxes)   
        # there is only one class
        labels = torch.ones((n_boxes,), dtype=torch.int64)       
              
        # transforms
        if self.transforms:
            sample = self.transforms(**{
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                })
            if len(sample['bboxes']) == 0:
                pass     
            image = sample['image']
            boxes = np.array(sample['bboxes'])        
        image = cv2.resize(image, (self.img_size, self.img_size))        
                                
        # imagenet normalisation
        if self.normalise:
            image = normalize(image, max_value=255)      

        # post-processing
        image = image.transpose(2,0,1).astype(np.float32) # channels first
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]] # why?

        image = torch.from_numpy(image) 
        boxes = torch.as_tensor(boxes, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.float)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        return image, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        boxes = self.box_callback(image_id)

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return image, boxes

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, len(self) - 1) for _ in range(3)]

        result_image = None
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                result_image = np.full((imsize, imsize, 3), 1, dtype=image.dtype)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax for result image
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax for original image
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a, x1a:x2a, :] = image[y1b:y2b, x1b:x2b, :]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]

        return result_image, result_boxes


def normalize(img: np.array, mean: list=[0.485, 0.456, 0.406], std: list=[0.229, 0.224, 0.225], max_value: float=255) -> np.array:
    """
    Normalize image data to 0-1 range,
    then apply mean and std as in ImageNet pretrain, or any other
    """    
    mean = np.array(mean, dtype=np.float32)
    mean *= max_value
    std = np.array(std, dtype=np.float32)
    std *= max_value

    img = img.astype(np.float32)
    img = img - mean    
    img = img / std

    return img


def load_image_boxes(image_id, labels: pd.DataFrame, format: str = 'pascal_voc') -> Tuple[np.array, List[int]]:
    """
    Load image and boxes in coco or pascal_voc format
        Args:
        
    """
    image = cv2.imread(f'{TRAIN_DIR}/{image_id}.jpg', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    records = labels[labels['image_id'] == image_id]

    # coco format
    boxes = records[['x', 'y', 'w', 'h']].values

    # pascal voc format    
    if format == 'pascal_voc':
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] 
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    return image, boxes


def test_dataset():

