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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms
from tqdm import tqdm
sys.path.append("../timm-efficientdet-pytorch")
import neptune
from effdet import DetBenchTrain, DetBenchEval, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet
from typing import Optional

from src.helpers.boxes_helpers import (filter_box_area, filter_box_size,
                                   preprocess_boxes)
from src.helpers.metric import competition_metric, find_best_nms_threshold
from src.helpers.model_helpers import (collate_fn, get_effdet_pretrain_names)

#from src.datasets.dataset_sergey import WheatDataset
from src.datasets.get_transforms import (get_train_transforms, get_transforms,
                                     get_valid_transforms, set_augmentations)

warnings.filterwarnings('ignore')

SEED = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(SEED)

print(torch.__version__)
print(neptune.__version__)

neptune.init('ods/wheat')
DATA_DIR = '../../data'
DIR_TRAIN = f'{DATA_DIR}/train'
DIR_TEST = f'{DATA_DIR}/test'

DIR_NEGATIVE = f'{DATA_DIR}/negative'
DIR_SPIKE = f'{DATA_DIR}/spike'
fold_column = 'fold'
fold = 1
image_id_column = 'image_id'
negative_prefix = 'negative_'
spike_dataset_prefix = 'SPIKE_'

num_workers = 2
train_batch_size = 4
inf_batch_size = 16
effective_train_batch_size = 16
grad_accum = effective_train_batch_size // train_batch_size
our_image_size = 512
n_epochs = 60
factor = 0.5
start_lr = 2e-4
min_lr = 1e-6
lr_patience = 1
overall_patience = 10
loss_delta = 1e-4
gpu_number = 0

model_name = 'effdet5'
experiment_tag = 'hard'
experiment_name = f'{model_name}_fold{fold}_{our_image_size}_{experiment_tag}'

# Define parameters
PARAMS = {'fold' : fold,
          'num_workers': num_workers,
          'train_batch_size': train_batch_size,
          'effective_train_batch_size': effective_train_batch_size,
          'grad_accum': grad_accum,
          'our_image_size': our_image_size,
          'n_epochs': n_epochs, 
          'factor': factor, 
          'start_lr': start_lr, 
          'min_lr': min_lr, 
          'lr_patience': lr_patience, 
          'overall_patience': overall_patience, 
          'loss_delta': loss_delta, 
          'experiment_tag': experiment_tag,         
         }

# Create experiment with defined parameters
neptune.create_experiment (name=model_name,
                          params=PARAMS, 
                          tags=[experiment_name, experiment_tag, f'fold_{fold}'],
                          upload_source_files=['model29.py', 'src/datasets/get_transforms.py'])    

neptune.append_tags(['grad_accum'])

train_boxes_df = pd.read_csv(os.path.join(DATA_DIR, 'fixed_train.csv'))
train_images_df = pd.read_csv(os.path.join(DATA_DIR,'orig_alex_folds.csv'))
train_images_df = pd.read_csv(os.path.join(DATA_DIR, 'alex_folds_with_negative.csv'))

train_boxes_df = preprocess_boxes(train_boxes_df)
# filter tiny boxes as well
#train_boxes_df = filter_box_size(train_boxes_df, min_size = 10)
#train_boxes_df['area'] = train_boxes_df['w'] * train_boxes_df['h']

train_boxes_df = pd.read_csv(os.path.join(DATA_DIR, 'fixed_train.csv'))
print('boxes original: ', len(train_boxes_df))
#spike_df = pd.read_csv(os.path.join(DATA_DIR, 'spike_train.csv'))
#train_boxes_df = pd.concat((train_boxes_df, spike_df), axis=0, sort=False)
#print('boxes after spike: ', len(train_boxes_df))


class WheatDataset(Dataset):
    def __init__(self, image_ids, image_dir, boxes_df, transforms, is_test):
        super().__init__()

        self.image_ids = image_ids
        self.image_dir = image_dir
        self.boxes_df = boxes_df
        self.transforms = transforms
        self.is_test = is_test

    def __getitem__(self, index: int):
        is_generated = False
        if self.is_test or random.random() > 0.5:
            #print('use load_image_and_boxes')
            image, boxes = self.load_image_and_boxes(index)
            is_generated = self.is_test or len(boxes) > 0
        if not is_generated:
            #print('use load_cutmix_image_and_boxes')
            for _ in range(10):
                image, boxes = self.load_cutmix_image_and_boxes(index)
                if len(boxes) > 0:
                    break

        if self.is_test:
            original_boxes = np.array(boxes, copy=True, dtype=int)
            # back to coco format
            original_boxes[:, 2] = original_boxes[:, 2] - original_boxes[:, 0]
            original_boxes[:, 3] = original_boxes[:, 3] - original_boxes[:, 1]
        else:
            # doesn't matter
            original_boxes = []
        n_boxes = len(boxes)
        class_id = 1

        # there is only one class
        labels = np.full((n_boxes,), class_id)

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                })
                if n_boxes == 0:
                    # just change image
                    image = sample['image']
                    boxes = np.zeros((0, 4), dtype=int)
                else:
                   if len(sample['bboxes']) == 0:
                        # try another augmentation
                        #print('try another augmentation')
                        continue
                   image = sample['image']
                   boxes = np.array(sample['bboxes'])
                break
            #if n_boxes > 0:
            #    assert len(boxes) > 0

        if not self.is_test and len(boxes) == 0:
            # don't know what to do here...
            print('workaround for no boxes')
            boxes = np.array([[0, 0, 1, 1]], dtype=int)
            labels = np.full((1,), class_id)

        # to tensors
        # https://github.com/rwightman/efficientdet-pytorch/blob/814bb96c90a616a20424d928b201969578047b4d/data/dataset.py#L77
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
        boxes = torch.as_tensor(boxes, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.float)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['original_boxes'] = original_boxes

        return transforms.ToTensor()(image), target, self.image_ids[index]

    def __len__(self) -> int:
        return len(self.image_ids)

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]

        records = self.boxes_df[self.boxes_df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values

        if image_id.startswith(negative_prefix):
            image_id = image_id[len(negative_prefix):]
            image = cv2.imread(f'{DIR_NEGATIVE}/{image_id}.jpg', cv2.IMREAD_COLOR)
        elif image_id.startswith(spike_dataset_prefix):
            image_id = image_id[len(spike_dataset_prefix):]
            image = cv2.imread(f'{DIR_SPIKE}/{image_id}.jpg', cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # pad to square
        h, w = image.shape[:2]
        if h != w:
            new_size = max(h, w)
            result_image = np.zeros((new_size, new_size, 3), dtype=image.dtype)
            result_image[0:h, 0:w, :] = image[0:h, 0:w, :]
        else:
            result_image = image

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        h, w = result_image.shape[:2]
        if h != 1024 or w != 1024:
            labels = np.full((len(boxes),), 1)
            sample = get_resize_1024()(**{
                'image': result_image,
                'bboxes': boxes,
                'labels': labels
            })
            result_image = sample['image']
            boxes = np.array(sample['bboxes'])
            if len(boxes) == 0:
                # to correct shape
                boxes = np.zeros((0, 4), dtype=int)

        return result_image, boxes

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

            if len(boxes) > 0:
                padw = x1a - x1b
                padh = y1a - y1b

                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh

                result_boxes.append(boxes)

        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, axis=0)
            np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
            result_boxes = result_boxes.astype(np.int32)
            result_boxes = result_boxes[
                np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        else:
            result_boxes = np.zeros((0, 4), dtype=int)
        return result_image, result_boxes


def collate_fn(batch):
    return tuple(zip(*batch))

def get_lr(optimizer ):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def load_weights(model, weights_file):
    model.load_state_dict(torch.load(weights_file, map_location=f'cuda:{gpu_number}'))

class ModelManager():
    def __init__(self, train_model, eval_model, device):
        self.train_model = train_model
        self.eval_model = eval_model
        self.device = device

    def train_epoch(self, optimizer, generator):
        self.train_model.train()
        tqdm_generator = tqdm(generator, mininterval=15)
        current_loss_mean = 0

        for batch_idx, (imgs, labels, image_id) in enumerate(tqdm_generator):
            #if batch_idx == 0:
            #   print('first batch is', image_id)
            loss = self.train_on_batch(optimizer, imgs, labels, batch_idx)

            # just slide average
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)

            tqdm_generator.set_description('loss: {:.4} lr:{:.6}'.format(
                current_loss_mean, get_lr(optimizer)), refresh=False)
        return current_loss_mean

    def train_on_batch(self, optimizer, batch_imgs, batch_labels, batch_idx):
        batch_imgs = torch.stack(batch_imgs)
        batch_imgs = batch_imgs.to(self.device).float()
        batch_boxes = [target['boxes'].to(self.device) for target in batch_labels]
        batch_labels = [target['labels'].to(self.device) for target in batch_labels]
        loss, _, _ = self.train_model(batch_imgs, batch_boxes, batch_labels)

        # If you are using a loss which is averaged over the training samples (which is the case most of the time),
        # you have to divide by the number of gradient accumulation steps
        loss = loss / grad_accum

        loss.backward()

        # https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
        if batch_idx % grad_accum == grad_accum - 1:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item() * grad_accum

    def predict(self, generator):
        self.eval_model.eval()
        self.eval_model.to(self.device)
        
        true_list = []
        pred_boxes = []
        pred_scores = []
        # sorry, now just hardcoded :(
        original_image_size = 1024

        tqdm_generator = tqdm(generator, mininterval=15)
        tqdm_generator.set_description('predict')
        with torch.no_grad():
            for batch_idx, (imgs, true_targets, _) in enumerate(tqdm_generator):
                if not (true_targets is None):
                    true_list.extend([gt['original_boxes'] for gt in true_targets])
                imgs = torch.stack(imgs)
                imgs = imgs.to(self.device).float()
            
                predicted = self.eval_model(imgs, torch.tensor([2] * len(imgs)).float().to(self.device))
                for i in range(len(imgs)):
                    cur_boxes = predicted[i].detach().cpu().numpy()[:, :4]
                    cur_boxes = np.array(cur_boxes, dtype=int)
                    cur_scores = predicted[i].detach().cpu().numpy()[:, 4]

                    # to pascal format
                    cur_boxes[:, 2] = cur_boxes[:, 0] + cur_boxes[:, 2]
                    cur_boxes[:, 3] = cur_boxes[:, 1] + cur_boxes[:, 3]

                    # clip by image edges
                    cur_boxes[:, 0] = np.clip(cur_boxes[:, 0], 0, original_image_size)
                    cur_boxes[:, 2] = np.clip(cur_boxes[:, 2], 0, original_image_size)
                    cur_boxes[:, 1] = np.clip(cur_boxes[:, 1], 0, original_image_size)
                    cur_boxes[:, 3] = np.clip(cur_boxes[:, 3], 0, original_image_size)

                    # to coco format
                    cur_boxes[:, 2] = cur_boxes[:, 2] - cur_boxes[:, 0]
                    cur_boxes[:, 3] = cur_boxes[:, 3] - cur_boxes[:, 1]

                    # drop strange boxes
                    cur_filter = (cur_boxes[:, 2] > 0) & (cur_boxes[:, 3] > 0)
                    cur_boxes = cur_boxes[cur_filter]
                    cur_scores = cur_scores[cur_filter]
                    assert len(cur_boxes) == len(cur_scores)

                    pred_boxes.append(cur_boxes)
                    pred_scores.append(cur_scores)
        return true_list, pred_boxes, pred_scores

    def run_train(self, train_generator, val_generator, n_epoches, weights_file, factor, start_lr, min_lr,
                  lr_patience, overall_patience, loss_delta=0.):
        self.best_loss = 100
        self.best_metric = 0
        self.best_epoch = 0
        self.curr_lr_loss = 100
        self.best_lr_epoch = 0

        self.train_model.to(self.device)
        #params = [p for p in self.train_model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params=self.train_model.parameters(), lr=start_lr)

        for epoch in range(n_epoches):
            print('!!!! Epoch {}'.format(epoch))
            train_loss = self.train_epoch(optimizer, train_generator)
            print(f'Train loss: {train_loss}, lr: {get_lr(optimizer)}')
            neptune.log_metric('Train loss', train_loss)
            neptune.log_metric('Lr', get_lr(optimizer))
            
            if not self.on_epoch_end(epoch, optimizer, val_generator, weights_file, factor, min_lr, lr_patience, overall_patience, loss_delta):
                break
      
    def on_epoch_end(self, epoch, optimizer, val_generator, weights_file, factor, min_lr, lr_patience, overall_patience, loss_delta):
        self.train_model.eval()
        current_loss = 0

        tqdm_generator = tqdm(val_generator, mininterval=15)
        tqdm_generator.set_description('validation loss')

        with torch.no_grad():
            for batch_idx, (batch_imgs, batch_labels, image_id) in enumerate(tqdm_generator):
                batch_imgs = torch.stack(batch_imgs)
                batch_imgs = batch_imgs.to(self.device).float()
                batch_boxes = [target['boxes'].to(self.device) for target in batch_labels]
                batch_labels = [target['labels'].to(self.device) for target in batch_labels]
                loss, _, _ = self.train_model(batch_imgs, batch_boxes, batch_labels)

                loss_value = loss.item()
                # just slide average
                current_loss = (current_loss * batch_idx + loss_value) / (batch_idx + 1)
        
        # validate loss        
        print('\nValidation loss: ', current_loss)
        neptune.log_metric('Validation loss', current_loss)
        
        # validate metric
        nms_thr = 0.4
        true_list, pred_boxes, pred_scores = self.predict(val_generator)
        current_metric = competition_metric(true_list, pred_boxes, pred_scores, nms_thr)
        print('\nValidation mAP', current_metric)
        neptune.log_metric('Validation mAP', current_metric)
        neptune.log_text('nms_threshold', str(nms_thr))
        
        if current_loss < self.best_loss - loss_delta:
            print(f'\nLoss has been improved from {self.best_loss} to {current_loss}')
            self.best_loss = current_loss
            self.best_epoch = epoch
            torch.save(self.train_model.model.state_dict(), f'{weights_file}')
        else:
            print(f'\nLoss has not been improved from {self.best_loss}')            

        if current_metric > self.best_metric:
            print(f'\nmAP has been improved from {self.best_metric} to {current_metric}')
            self.best_metric = current_metric
            self.best_epoch = epoch
            torch.save(self.train_model.model.state_dict(), f'{weights_file}_best_map')

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
                print('\nEpoch %05d: ReduceLROnPlateau reducing learning rate to %s.' % (epoch, new_lr))
        else:
            print('curr_lr_loss improved')
            self.curr_lr_loss = current_loss
            self.best_lr_epoch = epoch

        return True

def enumerate_images(test_dir):
    return [f for root, _, files in os.walk(test_dir) for f in files if f.endswith('.jpg')]

def do_main():
    neptune.init('ods/wheat')
    # Create experiment with defined parameters
    neptune.create_experiment(name=model_name,
                              params=PARAMS,
                              tags=[experiment_name, experiment_tag],
                              upload_source_files=[os.path.basename(__file__)])

    neptune.append_tags(f'fold_{fold}')

    device = torch.device(f'cuda:{gpu_number}') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    print(len(train_boxes_df))
    print(len(train_images_df))

    # Leave only > 0
    print('Leave only train images with boxes (validation)')
    with_boxes_filter = train_images_df[image_id_column].isin(train_boxes_df[image_id_column].unique())

    negative_images = enumerate_images(DIR_NEGATIVE)
    negative_images = [(negative_prefix + filename[:-4]) for filename in negative_images]
    negative_images.sort()
    # take first 100 now...
    negative_images = negative_images[:100]

    """
    spike_images = enumerate_images(DIR_SPIKE)
    spike_images = [(spike_dataset_prefix + filename[:-4]) for filename in spike_images]
    spike_images.sort()
    assert len(spike_images) > 0
    """

    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    load_weights(net, '../timm-efficientdet-pytorch/efficientdet_d5-ef44aea8.pth')

    config.num_classes = 1
    config.image_size = our_image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    model_train = DetBenchTrain(net, config)
    model_eval = DetBenchEval(net, config)

    manager = ModelManager(model_train, model_eval, device)

    pretrained_weights_file = 'pretrained.pth'

    images_val = train_images_df.loc[
        (train_images_df[fold_column] == fold) & with_boxes_filter, image_id_column].values
    images_train = train_images_df.loc[
        (train_images_df[fold_column] != fold), image_id_column].values

    #images_train = list(images_train) + list(negative_images) + list(spike_images)
    images_train = list(images_train) + list(negative_images)
    print(len(images_train), len(images_val))

    train_dataset = WheatDataset(images_train, DIR_TRAIN, train_boxes_df,
                                 transforms=get_train_transform(), is_test=False)
    valid_dataset = WheatDataset(images_val, DIR_TRAIN, train_boxes_df,
                                 transforms=get_valid_transform(), is_test=True)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
     )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=inf_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    weights_file = f'{experiment_name}.pth'
    if os.path.exists(pretrained_weights_file):
        # continue training
        print('Continue training, loading weights: ' + pretrained_weights_file)
        load_weights(net, pretrained_weights_file)

    manager.run_train(train_data_loader, valid_data_loader, n_epoches=n_epochs, weights_file=weights_file,
                      factor=factor, start_lr=start_lr, min_lr=min_lr, lr_patience=lr_patience, overall_patience=overall_patience, loss_delta=loss_delta)
    
    # add tags 
    neptune.log_text('save checkpoints as', weights_file[:-4])
    neptune.stop()

if __name__ == '__main__':
    do_main()
