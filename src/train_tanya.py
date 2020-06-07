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
sys.path.append("../timm-efficientdet-pytorch")
import neptune
from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet
from datasets.dataset_sergey import WheatDataset
from datasets.get_transforms import set_augmentations, get_transforms, get_train_transforms, get_valid_transforms
from helpers.metric import competition_map
from helpers.boxes_helpers import get_box, preprocess_boxes, filter_box_size, filter_box_area
from helpers.model_helpers import collate_fn, load_weigths, get_effdet_pretrain_names
from constants import DATA_DIR, TRAIN_DIR, META_TRAIN

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


def get_lr(optimizer ):
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

        for batch_idx, (imgs, labels, image_ids) in enumerate(tqdm_generator):            
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
            print(f'Train loss: {train_loss}, lr: {get_lr(optimizer)}')
            neptune.log_metric('Train loss', train_loss)
            neptune.log_metric('Lr', get_lr(optimizer))
            
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
                # metric

        print('validation loss: ', current_loss)
        neptune.log_metric('Validation loss', current_loss)
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


def main() -> None:
    device = f"cuda:{gpu_number}" if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    train_boxes_df = pd.read_csv(META_TRAIN)
    train_boxes_df = preprocess_boxes(train_boxes_df)
    train_images_df = pd.read_csv('orig_alex_folds.csv')    
    print(f'\nTotal images: {len(train_images_df['image_id'].unique())}')
    
    # Leave only images with bboxes
    image_id_column = 'image_id'
    print('Leave only train images with boxes (all)')
    with_boxes_filter = train_images_df[image_id_column].isin(train_boxes_df[image_id_column].unique())
    print(f'\nImages with bboxes: {len(with_boxes_filter['image_id'].unique())}')

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
        load_weights(net, '../timm-efficientdet-pytorch/{pretrain}')
    model = DetBenchTrain(net, config)

    runner = ModelRunner(model, device)
    weights_file = f'{experiment_name}.pth'
    # add tags 
    neptune.append_tags(f'{experiment_name}') 
    neptune.append_text('save checkpoints as', weights_file[:-4])

    # run training 
    runner.run_train(train_data_loader, valid_data_loader, n_epoches=n_epochs, weights_file=weights_file,
                      factor=factor, start_lr=start_lr, min_lr=min_lr, lr_patience=lr_patience, overall_patience=overall_patience, loss_delta=loss_delta)
    neptune.stop()


if __name__ == '__main__':
    main()
    
