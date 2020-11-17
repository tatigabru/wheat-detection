import os
import random
import sys
import warnings
#sys.path.append("../../timm-efficientdet-pytorch")
#sys.path.append("../../omegaconf")


import albumentations as A
#import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
#import torchvision
#from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from torch.utils.data import DataLoader, Dataset
#from torch.utils.data.sampler import SequentialSampler
#from torchvision import transforms
from tqdm import tqdm

from constants import META_TRAIN, TEST_DIR, TRAIN_DIR
from datasets.dataset_sergey import WheatDataset
from datasets.get_transforms import (get_train_transforms, get_transforms,
                                     get_valid_transforms, set_augmentations)
from effdet import (DetBenchEval, DetBenchTrain, EfficientDet,
                    get_efficientdet_config)
from effdet.efficientdet import HeadNet
from helpers.boxes_helpers import (filter_box_area, filter_box_size,
                                   format_prediction_string, preprocess_boxes)
from helpers.image_helpers import load_image
#from matplotlib import pyplot as plt
from helpers.metric import competition_map, iou, map_iou
from helpers.model_helpers import (collate_fn, get_effdet_pretrain_names,
                                   load_weights)
from model_runner import ModelRunner

warnings.filterwarnings('ignore')

#import re
#from PIL import Image

fold = 3
num_workers = 2
batch_size = 4
inf_batch_size = 16
image_size = 512
gpu_number=1

model_name = 'effdet4'
experiment_name = f'{model_name}_fold{fold}_{image_size}'
experiment_tag = 'run1'


def main() -> None:
    device = f"cuda:{gpu_number}" if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    train_boxes_df = pd.read_csv(META_TRAIN)
    train_boxes_df = preprocess_boxes(train_boxes_df)
    print(train_boxes_df.head())
    #train_images_df = pd.read_csv('orig_alex_folds.csv')    
    image_id_column = 'image_id'
    train_images_df = pd.read_csv('folds/train_alex_folds.csv')    
    print(f'\nTotal images: {len(train_images_df[image_id_column].unique())}')
    
    # Leave only images with bboxes
    #print('Leave only train images with boxes (all)')
    img_list = train_boxes_df[image_id_column].unique()
    print(len(img_list))
    with_boxes_filter = train_images_df[image_id_column].isin(img_list)
        
    fold = 0
    # val images
    images_val = img_list
    #images_val = train_images_df.loc[
    #    (train_images_df['fold'] == fold) & with_boxes_filter, image_id_column].values       
    print(f'\nValidation images {len(images_val)}')
    
    # get dataset
    valid_dataset = WheatDataset(
                                image_ids = images_val[:16], 
                                image_dir = TRAIN_DIR, 
                                labels_df = train_boxes_df,
                                #train_box_callback,
                                transforms=get_valid_transforms(image_size), 
                                is_test=True
                                )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=inf_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
        )
        
    # efficientdet config
    config = get_efficientdet_config(f'tf_efficientdet_d{model_name[-1]}')
    config.num_classes = 1
    config.image_size = image_size

    net = EfficientDet(config, pretrained_backbone=False)     
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    weights_file = f'{experiment_name}.pth'
        
    if os.path.exists(weights_file):    
        print(f'Loading weights from: {weights_file}')
        load_weights(net, weights_file)         
    else:
        print(f'No {weights_file} checkpoint')        
    model = DetBenchTrain(net, config)  

    # get predictions
    manager = ModelRunner(model, device)
    true_boxes, pred_boxes, pred_scores = manager.predict(valid_data_loader)

    nms_thresholds = np.linspace(min_thres, max_thres, num=points, endpoint=False)     
    best_metric = 0

    for thr in nms_thresholds:
        print('thr', thr)
        cur_metric = competition_metric(true_boxes, pred_boxes, pred_scores, thr)
        if cur_metric > best_metric:
            best_thr = thr
            best_metric = cur_metric
    print(f'best_metric: {best_metric}, best thr: {best_thr}')
   

if __name__ == '__main__':
    main()   
