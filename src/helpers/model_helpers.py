import os
import torch
import random
import numpy as np
from torch import nn
import json
from datetime import datetime
import logging
from typing import Tuple


def set_seed(seed: int=1234) -> None:
    """Fix all random seeds for reproductibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def fix_seeds_tf(seed: int=1234) -> None:
    """
    Fix all random seeds for reproductibility
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


def get_lr(optimizer ):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def load_weights(model: nn.Module, weights_file: str):
    model.load_state_dict(torch.load(weights_file))

    return model

