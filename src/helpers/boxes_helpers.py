"""
Helpers for bboxes preprocessing

"""
import pandas as pd
import re
import numpy as np
from typing import List, Union, Tuple, Optional


def expand_bbox(x: np.array) -> np.array:
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]

    return r


def preprocess_boxes(train_boxes_df: pd.DataFrame) -> pd.DataFrame:

    train_boxes_df['x'], train_boxes_df['y'], train_boxes_df['w'], train_boxes_df['h'] = -1, -1, -1, -1

    train_boxes_df[['x', 'y', 'w', 'h']] = np.stack(train_boxes_df['bbox'].apply(lambda x: expand_bbox(x)))
    train_boxes_df.drop(columns=['bbox'], inplace=True)
    train_boxes_df['x'] = train_boxes_df['x'].astype(np.float)
    train_boxes_df['y'] = train_boxes_df['y'].astype(np.float)
    train_boxes_df['w'] = train_boxes_df['w'].astype(np.float)
    train_boxes_df['h'] = train_boxes_df['h'].astype(np.float)

    return train_boxes_df


def filter_box_size(train_boxes_df: pd.DataFrame, min_size: Optional[int] = None, max_size: Optional[int] = 700) -> pd.DataFrame:
    """
    Apply filtering for boxes by size
        Args:
            boxes_df: pd.DataFrame with train boxes coordinates
            min_size: boxes with the h, w below minimum are removed
            max_size: boxes with the h, w above maximum are removed

        Output:
            pd.DataFrame with filtered boxes
    """
    train_boxes_df['area'] = train_boxes_df['w'] * train_boxes_df['h']  
    if min_size:      
        size_filter = train_boxes_df['w'] > min_size & train_boxes_df['h'] > min_size
        train_boxes_df = train_boxes_df[size_filter] 
    if max_size:      
        size_filter = train_boxes_df['w'] < max_size & train_boxes_df['h'] < max_size
        train_boxes_df = train_boxes_df[size_filter] 

    return train_boxes_df


def filter_box_area(train_boxes_df: pd.DataFrame, min_area: Optional[int] = 10, max_area: Optional[int] = 200000) -> pd.DataFrame:
    """
    Apply filtering for boxes by area
        Args:
            boxes_df: pd.DataFrame with train boxes coordinates
            min_area: boxes with the area below minimum are removed
            max_area: boxes with the area above maximum are removed

        Output:
            pd.DataFrame with filtered boxes
    """
    train_boxes_df['area'] = train_boxes_df['w'] * train_boxes_df['h']    
    if min_area:
        area_filter = (train_boxes_df['area'] > min_area)
        train_boxes_df = train_boxes_df[area_filter]
    if max_area:
        area_filter = (train_boxes_df['area'] < max_area) 
        train_boxes_df = train_boxes_df[area_filter]

    return train_boxes_df       


def get_boxes(boxes_df: pd.DataFrame, image_id: str) -> np.array:
    records = boxes_df[boxes_df['image_id'] == image_id]
    return records[['x', 'y', 'w', 'h']].values


def split_prediction_string(preds: str) -> np.array:
    """ 
    Get bboxes coordinaes (locations) from predicitons 
    """
    parts = preds.split()
    if len(parts) % 5 != 0: raise ValueError("Predicitons should have lens of 5: 4 coords and a label")
    locations = []
    for ind in range(len(parts) // 5):
        score = float(parts[ind * 5])
        location = int(float(parts[ind * 5 + 1])), int(float(parts[ind * 5 + 2])), \
                   int(float(parts[ind * 5 + 3])), int(float(parts[ind * 5 + 4]))
        locations.append(np.array(location))
    if len(locations) <= 0: raise ValueError("No locations")

    return np.array(locations) 


def format_prediction_string(boxes: list, scores: list) -> str:
    """ 
    Creates a string of bboxes preditions: confidence scores and coordinates 
    """
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)