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

from constants import *
from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet

from .datasets.dataset import WheatDataset
from .datasets.get_transforms import TRANSFORMS, get_transforms

warnings.filterwarnings('ignore')

sys.path.append("../timm-efficientdet-pytorch")


fold_column = 'fold'
image_id_column = 'image_id'
num_workers = 2
train_batch_size = 4
inf_batch_size = 16
our_image_size = 512

train_boxes_df = pd.read_csv(META_TRAIN)

train_boxes_df['x'] = -1
train_boxes_df['y'] = -1
train_boxes_df['w'] = -1
train_boxes_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_boxes_df[['x', 'y', 'w', 'h']] = np.stack(train_boxes_df['bbox'].apply(lambda x: expand_bbox(x)))
train_boxes_df.drop(columns=['bbox'], inplace=True)
train_boxes_df['x'] = train_boxes_df['x'].astype(np.float)
train_boxes_df['y'] = train_boxes_df['y'].astype(np.float)
train_boxes_df['w'] = train_boxes_df['w'].astype(np.float)
train_boxes_df['h'] = train_boxes_df['h'].astype(np.float)

train_boxes_df['area'] = train_boxes_df['w'] * train_boxes_df['h']
area_filter = (train_boxes_df['area'] < 160000) & (train_boxes_df['area'] > 50)
if False:
    train_boxes_df = train_boxes_df[area_filter]
else:
    print('No filtering for boxes')

train_images_df = pd.read_csv('folds.csv')

def train_box_callback(image_id):
    records = train_boxes_df[train_boxes_df['image_id'] == image_id]
    return records[['x', 'y', 'w', 'h']].values

def split_prediction_string(str):
    parts = str.split()
    assert len(parts) % 5 == 0
    locations = []
    for ind in range(len(parts) // 5):
        score = float(parts[ind * 5])
        location = int(float(parts[ind * 5 + 1])), int(float(parts[ind * 5 + 2])), \
                   int(float(parts[ind * 5 + 3])), int(float(parts[ind * 5 + 4]))
        # print(score)
        # print(location)
        locations.append(np.array(location))
    assert len(locations) > 0
    return np.array(locations)


def collate_fn(batch):
    return tuple(zip(*batch))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def load_weights(model, weights_file):
    model.load_state_dict(torch.load(weights_file))

class ModelManager():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_epoch(self, optimizer, generator):
        self.model.train()
        tqdm_generator = tqdm(generator, mininterval=30)
        current_loss_mean = 0

        for batch_idx, (imgs, labels, image_id) in enumerate(tqdm_generator):
            #if batch_idx == 0:
            #   print('first batch is', image_id)
            #if batch_idx > 5:
            #    break
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
        print(pred_scores)
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
        self.model.eval()
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

        """
        true_list, pred_boxes, pred_scores = self.predict(val_generator)
        cur_metric = competition_metric(true_list, pred_boxes, pred_scores, 0.4)
        print('competition_metric', cur_metric)
        """

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
                print('\nEpoch %05d: ReduceLROnPlateau reducing learning rate to %s.' % (epoch, new_lr))
        else:
            print('curr_lr_loss improved')
            self.curr_lr_loss = current_loss
            self.best_lr_epoch = epoch
        return True

def do_main():
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = "cuda:1"
    print(device)

    print(len(train_boxes_df))
    print(len(train_images_df))

    # Leave only > 0
    print('leavy only train images with boxes (all)')
    with_boxes_filter = train_images_df[image_id_column].isin(train_boxes_df[image_id_column].unique())

    fold_ = 1
    images_val = train_images_df.loc[
        (train_images_df[fold_column] == fold_) & with_boxes_filter, image_id_column].values
    images_train = train_images_df.loc[
        (train_images_df[fold_column] != fold_) & with_boxes_filter, image_id_column].values

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

    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    load_weights(net, '../timm-efficientdet-pytorch/efficientdet_d5-ef44aea8.pth')

    config.num_classes = 1
    config.image_size = our_image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    fold_weights_file = 'effdet.pth'
    if os.path.exists(fold_weights_file):
        # continue training
        print('Continue training, loading weights: ' + fold_weights_file)
        load_weights(net, fold_weights_file)

    model = DetBenchTrain(net, config)

    manager = ModelManager(model, device)
    weights_file = 'effdet.pth'
    manager.run_train(train_data_loader, valid_data_loader, n_epoches=40, weights_file=weights_file,
                      factor=0.5, start_lr=2e-4, min_lr=1e-6, lr_patience=1, overall_patience=10, loss_delta=1e-4)

if __name__ == '__main__':
    do_main()
