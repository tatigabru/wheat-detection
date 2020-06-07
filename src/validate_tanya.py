import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

import torch
import torchvision
import torch.optim as optim
from torchvision import transforms

import sys
sys.path.append("../timm-efficientdet-pytorch")

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchEval
from effdet.efficientdet import HeadNet

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from constants import META_TRAIN, TRAIN_DIR, TEST_DIR
from helpers.boxes_helpers import preprocess_boxes, filter_box_area, filter_box_size
import random
from datasets.dataset_sergey import WheatDataset
from datasets.get_transforms import set_augmentations, get_transforms, get_train_transforms, get_valid_transforms
from matplotlib import pyplot as plt
from helpers.metric import competition_map, iou, map_iou
from helpers.model_helpers import collate_fn, load_weigths, get_effdet_pretrain_names
from helpers.image_helpers import load_image
from helpers.boxes_helpers import format_prediction_string

num_workers = 2
batch_size = 2
inf_batch_size = 16
image_size = 512

train_boxes_df = pd.read_csv(META_TRAIN)
train_boxes_df = preprocess_boxes(train_boxes_df)
train_images_df = pd.read_csv('orig_alex_folds.csv')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class ModelRunner():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_epoch(self, optimizer, generator):
        self.model.train()
        tqdm_generator = tqdm(generator, mininterval=30)
        current_loss_mean = 0

        for batch_idx, (imgs, labels, image_id) in enumerate(tqdm_generator):
            loss = self.train_on_batch(optimizer, imgs, labels, batch_idx)

            # just slide average
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)

            tqdm_generator.set_description('loss: {:.4} lr:{:.6}'.format(
                current_loss_mean, get_lr(optimizer)))
        return current_loss_mean

    def train_on_batch(self, optimizer, batch_imgs, batch_labels, batch_idx):
        batch_imgs = torch.stack(batch_imgs)
        batch_imgs = batch_imgs.to(self.device).float()
        batch_boxes = [target['boxes'].to(self.device) for target in batch_labels]
        batch_labels = [target['labels'].to(self.device) for target in batch_labels]
        loss, _, _ = self.model(batch_imgs, batch_boxes, batch_labels)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def predict(self, generator):
        self.model.to(self.device)
        self.model.eval()
        tqdm_generator = tqdm(generator)
        true_list = []
        pred_boxes = []
        pred_scores = []
        for batch_idx, (imgs, true_targets, _) in enumerate(tqdm_generator):
            if not (true_targets is None):
                true_list.extend([2 * gt['boxes'].cpu().numpy() for gt in true_targets])
            imgs = torch.stack(imgs)
            imgs = imgs.to(self.device).float()
            with torch.no_grad():
                predicted = self.model(imgs, torch.tensor([2] * len(imgs)).float().cuda())
                for i in range(len(imgs)):
                    pred_boxes.append(predicted[i].detach().cpu().numpy()[:, :4])
                    pred_scores.append(predicted[i].detach().cpu().numpy()[:, 4])
            tqdm_generator.set_description('predict')
        #print(pred_scores)
        #print(pred_boxes)
        return true_list, pred_boxes, pred_scores

    def run_train(self, train_generator, val_generator, n_epoches, weights_file, factor, start_lr, min_lr,
                  lr_patience, overall_patience, loss_delta=0.):
        self.best_loss = 100
        self.best_epoch = 0
        self.curr_lr_loss = 100
        self.best_lr_epoch = 0

        self.model.to(self.device)
        #params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params=self.model.parameters(), lr=start_lr)

        for epoch in range(n_epoches):
            print('!!!! Epoch {}'.format(epoch))
            train_loss = self.train_epoch(optimizer, train_generator)
            print('Train loss:', train_loss)
            if not self.on_epoch_end(epoch, optimizer, val_generator, weights_file, factor, min_lr, lr_patience, overall_patience, loss_delta):
                break

    def on_epoch_end(self, epoch, optimizer, val_generator, weights_file, factor, min_lr, lr_patience, overall_patience, loss_delta):
        #true_boxes, pred_boxes, pred_scores = self.predict(val_generator)
        #current_loss = competition_loss(true_boxes,  pred_boxes, pred_scores)
        tqdm_generator = tqdm(val_generator, mininterval=30)
        current_loss = 0
        with torch.no_grad():
            for batch_idx, (batch_imgs, batch_labels, image_id) in enumerate(tqdm_generator):
                batch_imgs = torch.stack(batch_imgs)
                batch_imgs = batch_imgs.to(self.device).float()
                batch_boxes = [target['boxes'].to(self.device) for target in batch_labels]
                batch_labels = [target['labels'].to(self.device) for target in batch_labels]
                loss, _, _ = self.model(batch_imgs, batch_boxes, batch_labels)

                loss_value = loss.item()
                # just slide average
                current_loss = (current_loss * batch_idx + loss_value) / (batch_idx + 1)
        print('validation loss: ', current_loss)

        if current_loss < self.best_loss - loss_delta:
            print('Loss has been improved from', self.best_loss, 'to', current_loss)
            self.best_loss = current_loss
            self.best_epoch = epoch
            torch.save(self.model.model.state_dict(), weights_file)
        else:
            print('Loss has not been improved from', self.best_loss)
            if epoch - self.best_epoch > overall_patience:
                print('Training finished with patience!')
                return False

        print('curr_lr_loss', self.curr_lr_loss)
        if current_loss >= self.curr_lr_loss - loss_delta:
            print('curr_lr_loss not improved')
            old_lr = float(get_lr(optimizer))
            print('old_lr', old_lr)
            if old_lr > min_lr and epoch - self.best_lr_epoch > lr_patience:
                new_lr = old_lr * factor
                new_lr = max(new_lr, min_lr)
                print('new_lr', new_lr)
                set_lr(optimizer, new_lr)
                self.curr_lr_loss = 100
                self.best_lr_epoch = epoch
                self.best_lr_epoch = epoch
                print('\nEpoch %05d: ReduceLROnPlateau reducing learning rate to %s.' % (epoch, new_lr))
        else:
            print('curr_lr_loss improved')
            self.curr_lr_loss = current_loss
            self.best_lr_epoch = epoch
        return True



def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


def find_best_match(gts, pred, pred_idx, threshold=0.5, form='pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):

        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


def calculate_precision(gts, preds, threshold=0.5, form='coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


def calculate_image_precision(gts, preds, thresholds=(0.5,), form='coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0

    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision

iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

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

        image_precision = calculate_image_precision(gt_boxes, pred_boxes, thresholds=iou_thresholds, form='pascal_voc')
        final_scores.append(image_precision)

    return np.mean(final_scores)

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(len(train_boxes_df))
    print(len(train_images_df))

    # Leave only > 0
    print('leavy only train images with boxes')
    with_boxes_filter = train_images_df[image_id_column].isin(train_boxes_df[image_id_column].unique())

    fold_ = 1
    print('fold', fold_)
    images_val = train_images_df.loc[
        (train_images_df[fold_column] == fold_) & with_boxes_filter, image_id_column].values
    images_train = train_images_df.loc[
        (train_images_df[fold_column] != fold_), image_id_column].values

    print(len(images_train), len(images_val))

    train_dataset = WheatDataset(images_train, DIR_TRAIN, train_box_callback,
                                 transforms=get_train_transform(), is_test=False)
    valid_dataset = WheatDataset(images_val, DIR_TRAIN, train_box_callback,
                                 transforms=get_valid_transform(), is_test=True)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
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

    #weights_file = 'effdet_model14_fold' + str(fold_) + '.pth'
    weights_file = '../Weights/effdet_fold_1_model16_alex_fold1.pth'
    #weights_file = 'effdet_alex_fold0.pth'

    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = 1
    config.image_size = our_image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    load_weights(net, weights_file)
    model = DetBenchEval(net, config)

    manager = ModelManager(model, device)
   
    true_list, pred_boxes, pred_scores = manager.predict(valid_data_loader)
    prob_thresholds = np.linspace(0.35, 0.45, num=10, endpoint=False) # Prediction thresholds to consider a pixel positive
    best_metric = 0
    for thr in prob_thresholds:
        print('----------------------')
        print('thr', thr)
        cur_metric = competition_metric(true_list, pred_boxes, pred_scores, thr)
        if cur_metric > best_metric:
            best_thr = thr
            best_metric = cur_metric
    print('best thr:', best_thr)

    exit(0)
    
    best_thr = 0.01

    #test_df = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
    test_df = pd.read_csv(f'../Data_Zindi/full_dataset.csv')
    print(test_df.shape)
    test_images_ids = test_df[image_id_column].values

    DIR_TEST = '../Data_Zindi/Images'
    test_dataset = WheatDataset(test_images_ids, DIR_TEST, get_valid_transform())
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=6,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    true_list, pred_boxes, pred_scores = manager.predict(test_data_loader)
    pred_boxes = [convert_to_xyhw_boxes(x) for x in pred_boxes]

    results = []
    for i, image in enumerate(test_images_ids):
        image_id = test_images_ids[i]
        cur_boxes = np.array(pred_boxes[i])
        cur_scores = np.array(pred_scores[i])

        score_filter = cur_scores >= best_thr
        cur_boxes = cur_boxes[score_filter]
        cur_scores = cur_scores[score_filter]
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(cur_boxes, cur_scores)
        }
        results.append(result)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv('submission_zindi_model3_fold2_raw.csv', index=False)

    image_index = 1
    image_id = test_images_ids[image_index]
    image_file_name = f'{DIR_TEST}/{image_id}.jpg'
    print(image_file_name)

    sample = load_image(image_file_name)
    print(sample.shape)
    boxes = pred_boxes[image_index][pred_scores[image_index] >= best_thr].astype(np.int32)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        print(box)
        #assert box[0] >= 0 and box[0] <= sample.shap

        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0), 2)
    for box in true_list[image_index]:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0, 220, 0), 2)

    ax.set_axis_off()
    ax.imshow(sample)
    plt.show()