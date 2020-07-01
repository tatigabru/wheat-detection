import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from omegaconf import DictConfig
import torch
import os
import cv2
from albumentations.core.composition import Compose

from constants import *

def get_fold_oof(fold: int = 0):
    all_predictions = []
    validation_dataset = DatasetRetriever(
            image_ids=df_folds[df_folds['fold'] == fold].index.values,
            marking=marking,
            transforms=get_valid_transforms(),
            test=True,
        )

    validation_loader = DataLoader(
            validation_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            collate_fn=collate_fn
        )

        progress_bar = tqdm(validation_loader, total=len(validation_loader))
        for images, targets, image_ids in progress_bar:
            with torch.no_grad():
                images = torch.stack(images)
                images = images.cuda().float()
                det = models[fold_number](images, torch.tensor([1]*images.shape[0]).float().cuda())

                for i in range(images.shape[0]):
                    boxes = det[i].detach().cpu().numpy()[:,:4]    
                    scores = det[i].detach().cpu().numpy()[:,4]
                    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                    all_predictions.append({
                        'pred_boxes': (boxes*2).clip(min=0, max=1023).astype(int),
                        'scores': scores,
                        'gt_boxes': (targets[i]['boxes'].cpu().numpy()*2).clip(min=0, max=1023).astype(int),
                        'image_id': image_ids[i],
                    })
        oof = all_predictions



def get_folds_oof(folds_num: int = 5, save_oof: bool = True, save_dir: str = '') -> List[dict]:
    """
    Calculated OOF for all folds

    Args:
        folds_num : 

    Output:
        predictions: oof predictions for all folds  
    """

    all_predictions = []
    for fold_number in range(folds_num):
        validation_dataset = DatasetRetriever(
            image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
            marking=marking,
            transforms=get_valid_transforms(),
            test=True,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            collate_fn=collate_fn
        )

        progress_bar = tqdm(validation_loader, total=len(validation_loader))
        for images, targets, image_ids in progress_bar:
            with torch.no_grad():
                images = torch.stack(images)
                images = images.cuda().float()
                det = models[fold_number](images, torch.tensor([1]*images.shape[0]).float().cuda())

                for i in range(images.shape[0]):
                    boxes = det[i].detach().cpu().numpy()[:,:4]    
                    scores = det[i].detach().cpu().numpy()[:,4]
                    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                    all_predictions.append({
                        'pred_boxes': (boxes*2).clip(min=0, max=1023).astype(int),
                        'scores': scores,
                        'gt_boxes': (targets[i]['boxes'].cpu().numpy()*2).clip(min=0, max=1023).astype(int),
                        'image_id': image_ids[i],
                    })
        oof = all_predictions

    return all_predictions     


    def calculate_final_score(all_predictions, score_threshold):
        final_scores = []
        for i in range(len(all_predictions)):
            gt_boxes = all_predictions[i]['gt_boxes'].copy()
            pred_boxes = all_predictions[i]['pred_boxes'].copy()
            scores = all_predictions[i]['scores'].copy()
            image_id = all_predictions[i]['image_id']

            indexes = np.where(scores>score_threshold)
            pred_boxes = pred_boxes[indexes]
            scores = scores[indexes]

            image_precision = calculate_image_precision(gt_boxes, pred_boxes,thresholds=iou_thresholds,form='pascal_voc')
            final_scores.append(image_precision)

        return np.mean(final_scores)       