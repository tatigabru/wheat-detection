import os
import random
#import re
import sys
import warnings

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from tqdm import tqdm
sys.path.append("../../timm-efficientdet-pytorch")
sys.path.append("../../omegaconf")

import neptune
from constants import DATA_DIR, META_TRAIN, TRAIN_DIR
from datasets.dataset_sergey import WheatDataset
from datasets.get_transforms import (get_train_transforms, get_transforms,
                                     get_valid_transforms, set_augmentations)
from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet
from helpers.boxes_helpers import (filter_box_area, filter_box_size, get_box,
                                   preprocess_boxes)
from helpers.metric import competition_map, find_best_nms_threshold
from helpers.model_helpers import (collate_fn, get_effdet_pretrain_names,
                                   load_weights)
from model_runner import ModelRunner

warnings.filterwarnings('ignore')

fold = 3
num_workers = 2
batch_size = 4
inf_batch_size = 16
image_size = 512
n_epochs=60 
factor=0.2
start_lr=1e-3
min_lr=1e-6 
lr_patience=2
overall_patience=10 
loss_delta=1e-4
gpu_number=1

model_name = 'effdet4'
experiment_name = f'{model_name}_fold{fold}_{image_size}'
experiment_tag = 'run1'

# Define parameters
PARAMS = {'fold' : fold,
          'num_workers': num_workers,
          'train_batch_size': batch_size,
          'inf_batch_size': inf_batch_size,
          'our_image_size': image_size,
          'n_epochs': n_epochs, 
          'factor': factor, 
          'start_lr': start_lr, 
          'min_lr': min_lr, 
          'lr_patience': lr_patience, 
          'overall_patience': overall_patience, 
          'loss_delta': loss_delta,          
         }
print(f'parameters: {PARAMS}')

# Create experiment with defined parameters
neptune.init('ods/wheat')
neptune.create_experiment (name=experiment_name,
                          params=PARAMS, 
                          tags=['version_v4'],
                          upload_source_files=['train_tanya.py'])   


def main() -> None:
    device = f"cuda:{gpu_number}" if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    train_boxes_df = pd.read_csv(META_TRAIN)
    train_boxes_df = preprocess_boxes(train_boxes_df)
    train_images_df = pd.read_csv('orig_alex_folds.csv')    
    print(f'\nTotal images: {len(train_images_df.image_id.unique())}')
    
    # Leave only images with bboxes
    image_id_column = 'image_id'
    print('Leave only train images with boxes (all)')
    with_boxes_filter = train_images_df[image_id_column].isin(train_boxes_df[image_id_column].unique())
    
    # train/val images
    images_val = train_images_df.loc[
        (train_images_df['fold'] == fold) & with_boxes_filter, image_id_column].values
    images_train = train_images_df.loc[
        (train_images_df['fold'] != fold) & with_boxes_filter, image_id_column].values     
    print(f'\nTrain images:{len(images_train)}, validation images {len(images_val)}')
    
    # get datasets
    train_dataset = WheatDataset(
                                image_ids = images_train[:16], 
                                image_dir = TRAIN_DIR, 
                                #train_box_callback,
                                labels_df = train_boxes_df,
                                transforms = get_train_transforms(image_size), 
                                is_test = False
                                )
    valid_dataset = WheatDataset(
                                image_ids = images_val[:16], 
                                image_dir = TRAIN_DIR, 
                                labels_df = train_boxes_df,
                                #train_box_callback,
                                transforms=get_valid_transforms(image_size), 
                                is_test=True
                                )
    # get dataloaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
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
    # continue training
    if os.path.exists(weights_file):        
        print(f'Continue training, loading weights from: {weights_file}')
        load_weights(net, weights_file)
    else:
        print(f'Use coco pretrain')
        pretrain = get_effdet_pretrain_names(model_name)
        load_weights(net, '../../timm-efficientdet-pytorch/{pretrain}')
    model = DetBenchTrain(net, config)

    runner = ModelRunner(model, device)
    weights_file = f'{experiment_name}.pth'
    # add tags 
    neptune.append_tags(f'{experiment_name}') 
    neptune.log_text('save checkpoints as', weights_file[:-4])

    # run training 
    runner.run_train(train_data_loader, valid_data_loader, n_epoches=n_epochs, weights_file=weights_file,
                      factor=factor, start_lr=start_lr, min_lr=min_lr, lr_patience=lr_patience, overall_patience=overall_patience, loss_delta=loss_delta)
    neptune.stop()


if __name__ == '__main__':
    main()
