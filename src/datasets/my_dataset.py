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
                images_dir: str,  
                labels_df: pd.DataFrame,                       
                img_size: int = 512,                 
                transforms: str ='valid',                 
                normalise: bool = True,                         
                debug: bool = False,            
                ):
    
    
     , marking, image_ids, transforms=None, test=False):
        super().__init__()
        self.images_dir = images_dir                 
        self.image_ids = labels_df.image_id.unique()
        self.labels = labels_df
        self.img_size = img_size
        self.transforms = transforms
        self.normalise = normalise
        #self.test = test
        ids = os.listdir(images_dir)
        self.image_ids = [s[:-4] for s in ids]
        # select a subset for the debugging
        if self.debug:
            self.image_ids = self.image_ids[:160]
            print('Debug mode, samples: ', self.image_ids[:10])  

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        # load image and boxes        
        image = cv2.imread(f'{TRAIN_DIR}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image, boxes = load_image_and_boxes(image_id)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
#                     target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]   


def load_image_boxes(image_id, labels: pd.DataFrame, format: str = 'coco') -> Tuple[np.array, List[int]]:
    """Load image and boxes in coco or pascal_voc format"""
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


def collate_fn(batch):
    return tuple(zip(*batch))
