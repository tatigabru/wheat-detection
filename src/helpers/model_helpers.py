import json
import logging
import os
import random
from datetime import datetime
from typing import Tuple
import sys 
#sys.path.append("../timm-efficientdet-pytorch")

import numpy as np
import torch
from torch import nn

from effdet import (DetBenchEval, DetBenchTrain, EfficientDet,
                    get_efficientdet_config)
from effdet.efficientdet import HeadNet


def fix_seed(seed: int=1234) -> None:
    """
    Fix all random seeds for reproducibility
    PyTorch
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def fix_seeds_tf(seed: int=1234) -> None:
    """
    Fix all random seeds for reproducibility
    for Tensorflow
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    

def load_optim(optimizer: torch.optim, checkpoint_path: str, device: torch.device) -> torch.optim:
    """
    Load optimizer to continuer training
        Args:
            optimizer      : initialized optimizer
            checkpoint_path: path to the checkpoint
            device         : device to send optimizer to (must be the same as in the model)
            
        Note: must be called after initializing the model    

        Output: optimizer with the loaded state
    """  
    checkpoint = torch.load(checkpoint_path)    
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)    

    for param_group in optimizer.param_groups:
        print('learning_rate: {}'.format(param_group['lr']))    

    print('Loaded optimizer {} state from {}'.format(optimizer, checkpoint_path))    
    
    return optimizer


def save_ckpt(model: nn.Module, optimizer: torch.optim, checkpoint_path: str) -> dict:
    """
    Save model and optimizer checkpoint to continuer training
    """  
    torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
                checkpoint_path
            )
    print("Saved model and optimizer state to {}".format(checkpoint_path))


def load_ckpt(checkpoint_path: str) -> dict:
    """
    Load checkpoint to continuer training
        Args:
            checkpoint_path: path to the checkpoint

        Output: (dict) 0f the checkpoint state    

    """  
    checkpoint = torch.load(checkpoint_path)
        
    return checkpoint


def load_model(model: nn.Module, checkpoint_path: str) -> tuple:
    """
    Load model weigths to continuer training
        Args:
            model          : nn model
            checkpoint_path: path to the checkpoint  

        Output: 
            (nn.Module) nn model with weights
            (dict) 0f the checkpoint state
    """  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    return model, checkpoint    


def collate_fn(batch):
    return tuple(zip(*batch))


def load_weights(model: nn.Module, weights_file: str):
    model.load_state_dict(torch.load(weights_file))
    return model


def set_train_effdet(config, num_classes: int = 1, device: torch.device = 'cuda:0'):
    """Init EfficientDet to train mode"""
    model = EfficientDet(config, pretrained_backbone=False)    
    model.class_net = HeadNet(config, num_outputs=num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    model = DetBenchTrain(model, config)
    model = model.train()

    return model.to_device(device)     
   

def set_eval_effdet(checkpoint_path: str, config, num_classes: int = 1, device: torch.device = 'cuda:0'):
    """Init EfficientDet to validation mode"""
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint)

    net = DetBenchEval(net, config)
    net = net.eval()

    return net.to_device(device)  


def get_effdet_pretrain_names(alias: str = 'effdet4') -> str:
    """Returns pretrains names for differne efficient dets"""
    pretrains = { 
        'effdet0': 'efficientdet_d0-d92fd44f.pth',
        'effdet1': 'efficientdet_d1-4c7ebaf2.pth',
        'effdet2': 'efficientdet_d2-cb4ce77d.pth',
        'effdet3': 'efficientdet_d3-b0ea2cbc.pth',
        'effdet4': 'efficientdet_d4-5b370b7a.pth',
        'effdet5': 'efficientdet_d5-ef44aea8.pth',
        'effdet6': 'efficientdet_d6-51cb0132.pth',
        'effdet7': 'efficientdet_d7-f05bf714.pth',
    }
    return pretrains[alias] 