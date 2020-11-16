import os
import random
import re
import sys
import warnings

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms
from tqdm import tqdm
from typing import Optional



# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)

    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)

def convert_to_xyhw_box(box: list) -> list:
    #print('convert_to_xyhw_box', repr(box), box.__class__)
    x1, y1, x2, y2 = box
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    return [x1, y1, x2 - x1, y2 - y1]

def convert_to_xyhw_boxes(boxes):
    return [convert_to_xyhw_box(box.astype(np.int32)) for box in boxes]


# boxes in coco format!
def competition_metric(true_boxes, pred_boxes, pred_scores, score_thr):
    assert len(true_boxes) == len(pred_boxes)
    n_images = len(true_boxes)

    ns = 0
    nfps = 0
    ntps = 0
    overall_maps = 0
    for ind in range(n_images):
        cur_image_true_boxes = np.array(true_boxes[ind], copy=False)
        cur_image_pred_boxes = np.array(pred_boxes[ind], copy=False)
        cur_pred_scores = pred_scores[ind]
        score_filter = cur_pred_scores >= score_thr
        cur_image_pred_boxes = cur_image_pred_boxes[score_filter]
        cur_pred_scores = cur_pred_scores[score_filter]
        if (cur_image_true_boxes.shape[0] == 0 and cur_image_pred_boxes.shape[0] > 0):  # false positive
            ns = ns + 1  # increment denominator but add nothing to numerator
            nfps = nfps + 1  # track number of false positive cases, for curiosity
        elif (cur_image_true_boxes.shape[0] > 0):  # actual positive
            ns = ns + 1  # increment denominator & add contribution to numerator
            contrib = map_iou(cur_image_true_boxes, cur_image_pred_boxes, cur_pred_scores)
            # print('contrib', contrib)
            overall_maps = overall_maps + contrib
            if (cur_image_pred_boxes.shape[0] > 0):  # true positive
                ntps = ntps + 1  # track number of true positive cases, for curiosity
    overall_maps = overall_maps / (ns + 1e-7)
    print("ns:  ", ns)
    print("False positive cases:  ", nfps)
    print("True positive cases: ", ntps)
    print("Overall evaluation score: ", overall_maps)

    return overall_maps