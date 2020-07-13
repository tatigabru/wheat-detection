# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import pickle
import os
import time
import pandas as pd
import json
from multiprocessing import Pool, Process, cpu_count, Manager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from itertools import repeat
from ensemble_boxes import *
from map_boxes import mean_average_precision_for_boxes


def get_annotations_data_test():
    file_in = 'E:/Projects_M2/2019_06_Google_Open_Images/input/COCO2017/annotations/image_info_test-dev2017.json'
    images = dict()
    with open(file_in) as json_file:
        data = json.load(json_file)
        for i in range(len(data['images'])):
            image_id = data['images'][i]['id']
            images[image_id] = data['images'][i]

    return images


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def process_single_id(id, res_boxes, weights, params):
    run_type = params['run_type']
    verbose = params['verbose']

    # print('Go for ID: {}'.format(id))
    boxes_list = []
    scores_list = []
    labels_list = []
    labels_to_use_forward = dict()
    labels_to_use_backward = dict()

    for i in range(len(res_boxes[id])):
        boxes = []
        scores = []
        labels = []

        dt = res_boxes[id][i]

        for j in range(0, len(dt)):
            lbl = dt[j][5]
            scr = float(dt[j][4])
            box_x1 = float(dt[j][0])
            box_y1 = float(dt[j][1])
            box_x2 = float(dt[j][2])
            box_y2 = float(dt[j][3])

            if box_x1 >= box_x2:
                if verbose:
                    print('Problem with box x1 and x2: {}. Skip it'.format(dt[j]))
                continue
            if box_y1 >= box_y2:
                if verbose:
                    print('Problem with box y1 and y2: {}. Skip it'.format(dt[j]))
                continue
            if scr <= 0:
                if verbose:
                    print('Problem with box score: {}. Skip it'.format(dt[j]))
                continue

            boxes.append([box_x1, box_y1, box_x2, box_y2])
            scores.append(scr)
            if lbl not in labels_to_use_forward:
                cur_point = len(labels_to_use_forward)
                labels_to_use_forward[lbl] = cur_point
                labels_to_use_backward[cur_point] = lbl
            labels.append(labels_to_use_forward[lbl])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # Empty predictions for all models
    if len(boxes_list) == 0:
        return np.array([]), np.array([]), np.array([])

    if run_type == 'wbf':
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                                       weights=weights, iou_thr=params['intersection_thr'],
                                                                       skip_box_thr=params['skip_box_thr'],
                                                                           conf_type=params['conf_type'])
    elif run_type == 'nms':
        iou_thr = params['iou_thr']
        merged_boxes, merged_scores, merged_labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    elif run_type == 'soft-nms':
        iou_thr = params['iou_thr']
        sigma = params['sigma']
        thresh = params['thresh']
        merged_boxes, merged_scores, merged_labels = soft_nms(boxes_list, scores_list, labels_list,
                                                              weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=thresh)
    elif run_type == 'nmw':
        merged_boxes, merged_scores, merged_labels = non_maximum_weighted(boxes_list, scores_list, labels_list,
                                                                       weights=weights, iou_thr=params['intersection_thr'],
                                                                       skip_box_thr=params['skip_box_thr'])

    # print(len(boxes_list), len(merged_boxes))
    if 'limit_boxes' in params:
        limit_boxes = params['limit_boxes']
        if len(merged_boxes) > limit_boxes:
            merged_boxes = merged_boxes[:limit_boxes]
            merged_scores = merged_scores[:limit_boxes]
            merged_labels = merged_labels[:limit_boxes]

    # Rename labels back
    merged_labels_string = []
    for m in merged_labels:
        merged_labels_string.append(labels_to_use_backward[m])
    merged_labels = np.array(merged_labels_string, dtype=np.str)

    # Create IDs array
    ids_list = [id] * len(merged_labels)

    return merged_boxes.copy(), merged_scores.copy(), merged_labels.copy(), ids_list.copy()


def process_part_of_data(proc_number, return_dict, ids_to_use, res_boxes, weights, params):
    print('Start process: {} IDs to proc: {}'.format(proc_number, len(ids_to_use)))
    result = []
    for id in ids_to_use:
        merged_boxes, merged_scores, merged_labels, ids_list = process_single_id(id, res_boxes, weights, params)
        # print(merged_boxes.shape, merged_scores.shape, merged_labels.shape, len(ids_list))
        result.append((merged_boxes, merged_scores, merged_labels, ids_list))
    return_dict[proc_number] = result.copy()


def ensemble_predictions(pred_filenames, weights, params):
    verbose = False
    if 'verbose' in params:
        verbose = params['verbose']

    start_time = time.time()
    procs_to_use = max(cpu_count() // 2, 1)
    procs_to_use = 6
    print('Use processes: {}'.format(procs_to_use))

    res_boxes = dict()
    ref_ids = None
    for j in range(len(pred_filenames)):
        s = pd.read_csv(pred_filenames[j], dtype={'img_id': np.str, 'label': np.str})
        s.sort_values('img_id', inplace=True)
        s.reset_index(drop=True, inplace=True)
        ids = s['img_id'].values
        if ref_ids is None:
            ref_ids = tuple(ids)
        else:
            if ref_ids != tuple(ids):
                print('Different IDs in ensembled CSVs!')
                exit()
        preds = s[['x1', 'y1', 'x2', 'y2', 'score', 'label']].values
        single_res = dict()
        for i in range(len(ids)):
            id = ids[i]
            if id not in single_res:
                single_res[id] = []
            single_res[id].append(preds[i])
        for el in single_res:
            if el not in res_boxes:
                res_boxes[el] = []
            res_boxes[el].append(single_res[el])

    ids_to_use = sorted(list(res_boxes.keys()))
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(procs_to_use):
        start = i * len(ids_to_use) // procs_to_use
        end = (i+1) * len(ids_to_use) // procs_to_use
        if i == procs_to_use - 1:
            end = len(ids_to_use)
        p = Process(target=process_part_of_data, args=(i, return_dict, ids_to_use[start:end], res_boxes, weights, params))
        jobs.append(p)
        p.start()

    for i in range(len(jobs)):
        jobs[i].join()

    results = []
    for i in range(len(jobs)):
        results += return_dict[i]

    # p = Pool(processes=procs_to_use)
    # results = p.starmap(process_single_id, zip(ids_to_use, repeat(weights), repeat(params)))

    all_ids = []
    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, labels, ids_list in results:
        if boxes is None:
            continue
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_ids.append(ids_list)

    all_ids = np.concatenate(all_ids)
    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    if verbose:
        print(all_ids.shape, all_boxes.shape, all_scores.shape, all_labels.shape)

    res = pd.DataFrame(all_ids, columns=['img_id'])
    res['label'] = all_labels
    res['score'] = all_scores
    res['x1'] = all_boxes[:, 0]
    res['x2'] = all_boxes[:, 2]
    res['y1'] = all_boxes[:, 1]
    res['y2'] = all_boxes[:, 3]
    print('Run time: {:.2f}'.format(time.time() - start_time))
    return res


def get_annotations_data():
    file_in = 'E:/Projects_M2/2019_06_Google_Open_Images/input/COCO2017/annotations/instances_val2017.json'
    images = dict()
    with open(file_in) as json_file:
        data = json.load(json_file)
        for i in range(len(data['images'])):
            image_id = data['images'][i]['id']
            images[image_id] = data['images'][i]

    return images


def convert_csv_predictions_to_coco(csv_path, out_path=None):
    images = get_annotations_data()
    s = pd.read_csv(csv_path, dtype={'img_id': np.str, 'label': np.str})

    out = np.zeros((len(s), 7), dtype=np.float64)
    out[:, 0] = s['img_id']
    ids = s['img_id'].astype(np.int32).values
    x1 = s['x1'].values
    x2 = s['x2'].values
    y1 = s['y1'].values
    y2 = s['y2'].values
    for i in range(len(s)):
        width = images[ids[i]]['width']
        height = images[ids[i]]['height']
        out[i, 1] = x1[i] * width
        out[i, 2] = y1[i] * height
        out[i, 3] = (x2[i] - x1[i]) * width
        out[i, 4] = (y2[i] - y1[i]) * height
    out[:, 5] = s['score'].values
    out[:, 6] = s['label'].values
    if out_path:
        save_in_file(out, out_path)

    filename = 'E:/Projects_M2/2019_06_Google_Open_Images/input/COCO2017/annotations/instances_val2017.json'
    coco_gt = COCO(filename)
    detections = out
    # print(detections.shape)
    # print(detections[:5])
    image_ids = list(set(detections[:, 0]))
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats
    print(coco_metrics)
    return coco_metrics, detections


def save_to_json(detections, output_path):
    box_result_list = []
    for det in detections:
        box_result_list.append({
            'image_id': int(det[0]),
            'category_id': int(det[6]),
            'bbox': np.around(
                det[1:5].astype(np.float64), decimals=2).tolist(),
            'score': float(np.around(det[5], decimals=4)),
        })
    json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    with open(output_path, 'w') as fid:
        json.dump(box_result_list, fid)


if __name__ == '__main__':
    ensemble_test = True
    if 1:
        params = {
            'run_type': 'nms',
            'iou_thr': 0.5,
            'verbose': False,
            'out_folder': "experiment/EffDetB7_nms/"
        }
    if 0:
        params = {
            'run_type': 'soft-nms',
            'iou_thr': 0.5,
            'thresh': 0.0001,
            'sigma': 0.1,
            'verbose': True,
            'out_folder': "experiment/EffDetB7_softnms/"
        }
    if 0:
        params = {
            'run_type': 'nmw',
            'skip_box_thr': 0.000000001,
            'intersection_thr': 0.5,
            'limit_boxes': 30000,
            'verbose': True,
            'out_folder': "experiment/EffDetB7_nmw/"
        }
    if 0:
        params = {
            'run_type': 'wbf',
            'skip_box_thr': 0.001,
            'intersection_thr': 0.7,
            'conf_type': 'avg',
            'limit_boxes': 30000,
            'verbose': False,
            'out_folder': "experiment/EffDetB7_wbf/"
        }

    pred_list = [
        'predictions/EffNetB7-preds.csv',
        'predictions/EffNetB7-mirror-preds.csv',
    ]
    weights = [1, 1]

    best_metric = -1
    best_params = None
    for iou_thr in range(40, 80, 1):
        p = params
        p['iou_thr'] = iou_thr / 100
        ensemble_preds = ensemble_predictions(pred_list, weights, params)

        save_prefix = 'ens'
        for f in pred_list:
            nm = os.path.basename(f)
            nm = nm.replace('EffNet', '')
            nm = nm.replace('-preds.csv', '')
            nm = nm.replace('-mirror', 'm')
            save_prefix += '_' + nm
        for el in sorted(list(params.keys())):
            if el == 'out_folder':
                continue
            save_prefix += '_' + str(params[el])
        save_prefix += '.csv'

        save_path1 = "{}/{}".format(params['out_folder'], save_prefix)
        if os.path.isfile(save_path1):
            print('File already exists: {}. Skip'.format(save_path1))
            continue

        ensemble_preds.to_csv(save_path1, index=False)
        ensemble_preds = ensemble_preds[['img_id', 'label', 'score', 'x1', 'x2', 'y1', 'y2']].values
        ann_path = 'E:/Projects_M2/2019_06_Google_Open_Images/input/COCO2017/annotations/instances_val2017.csv'
        ann = pd.read_csv(ann_path, dtype={'img_id': np.str, 'label': np.str})
        ann_numpy = ann[['img_id', 'label', 'x1', 'x2', 'y1', 'y2']].values
        mean_ap, average_precisions = mean_average_precision_for_boxes(ann_numpy, ensemble_preds, iou_threshold=0.5, verbose=False)
        coco_preds, detections = convert_csv_predictions_to_coco("{}/{}".format(params['out_folder'], save_prefix))
        print("Ensemble [{}] Weights: {} Params: {} mAP: {:.6f}".format(len(weights), weights, params, mean_ap))
        print("Coco preds: {}".format(coco_preds))

        out = open("{}/{}.txt".format(params['out_folder'], save_prefix[:-4]), 'w')
        out.write('{}\n'.format(pred_list))
        out.write('{}\n'.format(weights))
        out.write('{}\n'.format(params))
        out.write('{}\n'.format(mean_ap))
        out.write('{}\n'.format(coco_preds))
        out.close()

        if coco_preds[0] > best_metric:
            best_metric = coco_preds[0]
            best_params = p.copy()

    print('Best metric found: {}'.format(best_metric))
    print('Best params: {}'.format(best_params))

