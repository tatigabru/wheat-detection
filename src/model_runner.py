import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import neptune

from src.helpers.metric import competition_metric


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class ModelRunner():
    def __init__(self, model, device):
        self.model = model
        #self.eval_model = eval_model
        self.device = device


    def train_epoch(self, optimizer, generator):
        self.model.train()
        tqdm_generator = tqdm(generator, mininterval=30)
        current_loss_mean = 0

        for batch_idx, (imgs, labels, image_id) in enumerate(tqdm_generator):
            loss = self.train_on_batch(optimizer, imgs, labels, batch_idx)
            
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


    def run_train(self, train_generator, val_generator, n_epoches, weights_file, factor, start_lr, min_lr,
                  lr_patience, overall_patience, loss_delta=0.):
        self.best_loss = 100
        self.best_metric = 0
        self.best_epoch = 0
        self.curr_lr_loss = 100
        self.best_lr_epoch = 0
        self.model.to(self.device)
        #params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params=self.model.parameters(), lr=start_lr)

        for epoch in range(n_epoches):
            print(f'Epoch {epoch}')
            train_loss = self.train_epoch(optimizer, train_generator)
            print('Train loss:', train_loss)
            neptune.log_metric('Train loss', train_loss)
            neptune.log_metric('Lr', get_lr(optimizer))

            if not self.on_epoch_end(epoch, optimizer, val_generator, weights_file, factor, min_lr, lr_patience, overall_patience, loss_delta):
                break


    def on_epoch_end(self, epoch, optimizer, val_generator, weights_file, factor, min_lr, lr_patience, overall_patience, loss_delta):
        
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
                # sliding average
                current_loss = (current_loss * batch_idx + loss_value) / (batch_idx + 1)

        # validate loss        
        print('\nValidation loss: ', current_loss)
        neptune.log_metric('Validation loss', current_loss)
        
        # validate metric
        nms_thr = 0.37
        #true_list, pred_boxes, pred_scores = self.predict(val_generator)
        #current_metric = competition_metric(true_list, pred_boxes, pred_scores, nms_thr)
        #print('\nValidation mAP', current_metric)
        #neptune.log_metric('Validation mAP', current_metric)
        #neptune.log_text('nms_threshold', str(nms_thr))

        if current_loss < self.best_loss - loss_delta:
            print(f'\nLoss has been improved from {self.best_loss} to {current_loss}')
            self.best_loss = current_loss
            self.best_epoch = epoch
            torch.save(self.model.model.state_dict(), f'{weights_file}_best_loss')
        else:
            print(f'\nLoss has not been improved from {self.best_loss}')            

        if epoch - self.best_epoch > overall_patience:
            print('\nEarly stop: training finished with patience!')
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