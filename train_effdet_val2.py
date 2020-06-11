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
from effdet import DetBenchTrain, DetBenchEval, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet
from typing import Optional, List, Tuple

from src.helpers.boxes_helpers import (filter_box_area, filter_box_size,
                                   preprocess_boxes)
from src.helpers.metric import competition_metric, find_best_nms_threshold
from src.helpers.model_helpers import (collate_fn, get_effdet_pretrain_names)

#from src.datasets.dataset_sergey import WheatDataset
from src.datasets.get_transforms import (get_train_transforms, get_transforms,
                                     get_valid_transforms, set_augmentations)

warnings.filterwarnings('ignore')


print(torch.__version__)
print(neptune.__version__)

neptune.init('ods/wheat')

DATA_DIR = '../Data'
DIR_TRAIN = f'{DATA_DIR}/train'
DIR_TEST = f'{DATA_DIR}/test'
fold_column = 'fold'
fold = 1
image_id_column = 'image_id'
num_workers = 2
train_batch_size = 4
inf_batch_size = 16
our_image_size = 512
n_epochs=60 
factor=0.2
start_lr=1e-3
min_lr=1e-6 
lr_patience=2
overall_patience=10 
loss_delta=1e-4
gpu_number=1

model_name = 'effdet4'
experiment_name = f'{model_name}_fold{fold}_{our_image_size}'
experiment_tag = 'v1'

# Define parameters
PARAMS = {'fold' : fold,
          'num_workers': num_workers,
          'train_batch_size': train_batch_size,
          'our_image_size': our_image_size,
          'n_epochs': n_epochs, 
          'factor': factor, 
          'start_lr': start_lr, 
          'min_lr': min_lr, 
          'lr_patience': lr_patience, 
          'overall_patience': overall_patience, 
          'loss_delta': loss_delta,          
         }

# Create experiment with defined parameters
neptune.create_experiment (name=model_name,
                          params=PARAMS, 
                          tags=[experiment_name, experiment_tag],
                          upload_source_files=['train_effdet_val-fix.py'])    

neptune.append_tags(f'fold_{fold}')

train_boxes_df = pd.read_csv(os.path.join(DATA_DIR, 'fixed_train.csv'))
train_images_df = pd.read_csv(os.path.join(DATA_DIR,'orig_alex_folds.csv'))

train_boxes_df = preprocess_boxes(train_boxes_df)
# filter tiny boxes as well
#train_boxes_df = filter_box_size(train_boxes_df, min_size = 10)


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


class WheatDataset(Dataset):
    def __init__(self, image_ids, image_dir, box_callback, transforms, is_test):
        super().__init__()

        self.image_ids = image_ids
        self.image_dir = image_dir
        self.box_callback = box_callback
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
            image, boxes = self.load_cutmix_image_and_boxes(index)
            assert len(boxes) > 0

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
            if n_boxes > 0:
                assert len(boxes) > 0

        # to tensors
        # https://github.com/rwightman/efficientdet-pytorch/blob/814bb96c90a616a20424d928b201969578047b4d/data/dataset.py#L77
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
        boxes = torch.as_tensor(boxes, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.float)
        #if n_boxes == 0:
        #   labels = torch.LongTensor([])

        #print('boxes', repr(boxes))
        #print('labels', repr(labels))

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
        boxes = self.box_callback(image_id)

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return image, boxes

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
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes



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
            overall_maps = overall_maps + contrib
            if (cur_image_pred_boxes.shape[0] > 0):  # true positive
                ntps = ntps + 1  # track number of true positive cases, for curiosity
    overall_maps = overall_maps / (ns + 1e-7)
    print("\nSamples:  ", ns)
    print("\nFalse positive cases:  ", nfps)
    print("\nTrue positive cases: ", ntps)
    print("\nOverall evaluation score: ", overall_maps)

    return overall_maps


def collate_fn(batch):
    return tuple(zip(*batch))

def get_lr(optimizer ):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def load_weights(model, weights_file):
    model.load_state_dict(torch.load(weights_file))


class ModelManager():
    def __init__(self, train_model, eval_model, device):
        self.train_model = train_model
        self.eval_model = eval_model
        self.device = device

    def train_epoch(self, optimizer, generator):
        self.train_model.train()
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
        loss, _, _ = self.train_model(batch_imgs, batch_boxes, batch_labels)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def predict(self, generator):
        self.eval_model.eval()
        self.eval_model.to(self.device)
        
        tqdm_generator = tqdm(generator)
        true_list = []
        pred_boxes = []
        pred_scores = []

        # sorry, now just hardcoded :(
        original_image_size = our_image_size # not 1024 as we resize train and val the same way

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
            tqdm_generator.set_description('predict')
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
        tqdm_generator = tqdm(val_generator, mininterval=30)
        current_loss = 0
        self.train_model.eval()

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


def do_main():
    device = torch.device(f'cuda:{gpu_number}') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    print(len(train_boxes_df))
    print(len(train_images_df))

    # Leave only > 0
    print('Leave only train images with boxes')
    with_boxes_filter = train_images_df[image_id_column].isin(train_boxes_df[image_id_column].unique())

    images_val = train_images_df.loc[
        (train_images_df[fold_column] == fold) & with_boxes_filter, image_id_column].values
    images_train = train_images_df.loc[
        (train_images_df[fold_column] != fold) & with_boxes_filter, image_id_column].values     
    print(f'\nTrain images:{len(images_train)}, validation images {len(images_val)}')

    # get datasets
    augs_dict = set_augmentations(our_image_size)

    train_dataset = WheatDataset(
                                image_ids = images_train[:16], 
                                image_dir = DIR_TRAIN, 
                                box_callback = train_box_callback,                                
                                transforms = get_transforms(augs_dict["hard"]), 
                                is_test = False
                                )
    valid_dataset = WheatDataset(
                                image_ids = images_val[:16], 
                                image_dir = DIR_TRAIN,                                 
                                box_callback = train_box_callback,
                                transforms=get_transforms(augs_dict["resize"]), 
                                is_test=True
                                )

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

    config = get_efficientdet_config(f'tf_efficientdet_d{model_name[-1]}')
    net = EfficientDet(config, pretrained_backbone=False)
    #load_weights(net, '../timm-efficientdet-pytorch/efficientdet_d4-5b370b7a.pth')
    #load_weights(net, '../timm-efficientdet-pytorch/efficientdet_d5-ef44aea8.pth')
    
    config.num_classes = 1
    config.image_size = our_image_size
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

    model_train = DetBenchTrain(net, config)
    model_eval = DetBenchEval(net, config)

    manager = ModelManager(model_train, model_eval, device)
    weights_file = f'{experiment_name}.pth'     

    manager.run_train(train_data_loader, valid_data_loader, n_epoches=n_epochs, weights_file=weights_file,
                      factor=factor, start_lr=start_lr, min_lr=min_lr, lr_patience=lr_patience, overall_patience=overall_patience, loss_delta=loss_delta)
    
    # add tags 
    neptune.log_text('save checkpoints as', weights_file[:-4])
    neptune.stop()


if __name__ == '__main__':
    do_main()
    
