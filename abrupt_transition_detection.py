import os
import re
from typing import Tuple

import cv2
import numpy as np
from sklearn.metrics import jaccard_score

from common import FRAME_FOLDER
from utils import time_span_to_timestamps_list

MASK_WIDTH = 160
MASK_HEIGHT = 180


def abs_diff(img_1: np.ndarray, img_2: np.ndarray) -> np.float64:
    img_area = img_1.shape[0] * img_1.shape[1]
    return np.sum(np.absolute(img_2 - img_1)) / img_area


def hist_diff(img_1: np.ndarray, img_2: np.ndarray, region_diff=False) \
        -> np.float64:
    all_masks = []
    for i in range(2):
        for j in range(4):
            mask = np.zeros(img_1.shape[:2], np.uint8)
            top_left_x = (0 + j) * MASK_WIDTH
            top_left_y = (0 + i) * MASK_HEIGHT
            bot_right_x = top_left_x + MASK_WIDTH
            bot_right_y = top_left_y + MASK_HEIGHT
            mask[top_left_y:bot_right_y, top_left_x: bot_right_x] = 255
            all_masks.append(mask)

    def get_hist(img, mask_idx, chan_idx):
        if mask_idx == -1:
            return cv2.calcHist([img], [chan_idx], None, [256], [0, 256])
        else:
            return cv2.calcHist(
                [img], [chan_idx], all_masks[mask_idx], [256], [0, 256]
            )

    def get_chan_hist_diff(mask_idx, chan_idx):
        return np.abs(get_hist(img_1, mask_idx, chan_idx) -
                      get_hist(img_2, mask_idx, chan_idx))

    if region_diff:
        score = np.float64(0)
        for i in range(8):
            b_diff = get_chan_hist_diff(i, 0)
            g_diff = get_chan_hist_diff(i, 1)
            r_diff = get_chan_hist_diff(i, 2)
            score += np.sum(b_diff + g_diff + r_diff, dtype=np.float64)
    else:
        b_diff = get_chan_hist_diff(-1, 0)
        g_diff = get_chan_hist_diff(-1, 1)
        r_diff = get_chan_hist_diff(-1, 2)
        score = np.sum(b_diff + g_diff + r_diff, dtype=np.float64)
    return score


def gen_pixel_diff(data: dict, use_time_span: bool = False, score_method='abs'):
    assert score_method in ['abs', 'hist-all', 'hist-reg']

    vid_folder = FRAME_FOLDER + data['vid_name'] + '/'
    l = list(filter(lambda f: f.endswith('.jpg'), os.listdir(vid_folder)))
    l.sort(key=lambda f: int(re.sub('\D', '', f)))

    if use_time_span:
        timestamps = time_span_to_timestamps_list(data)
    else:
        timestamps = list(range(1, len(l) + 1))

    imgs = [cv2.cvtColor(cv2.imread(vid_folder + l[i]), cv2.COLOR_BGR2RGB)
            for i in range(len(l)) if (i + 1) in timestamps]

    scores = []

    for i in range(len(imgs) - 1):
        img_1 = imgs[i]
        img_2 = imgs[i + 1]
        if score_method == 'abs':
            diff = abs_diff(img_1, img_2)
        elif score_method == 'hist-all':
            diff = hist_diff(img_1, img_2)
        else:
            diff = hist_diff(img_1, img_2, True)
        scores.append(diff)

    return {
        'timestamps': timestamps,
        'pixel_diff_score': scores
    }


def non_max_suppression(scores: list, preds: list, ignore_class_idx: int):
    def nms_compare(target, n1, n2):
        if target < n1 or target < n2:
            return 0
        else:
            return target

    new_scores = []

    for i in range(len(scores)):
        if preds[i] == ignore_class_idx:
            new_scores.append(0)
        elif i == 0:
            new_scores.append(nms_compare(scores[i], 0, scores[i + 1]))
        elif i == len(scores) - 1:
            new_scores.append(nms_compare(scores[i], 0, scores[i - 1]))
        else:
            new_scores.append(
                nms_compare(scores[i], scores[i - 1], scores[i + 1]))

    return new_scores


def ab_change_jacc_score(pred_change: list, gt_change: list,
                         no_change_class_index: int) -> float:
    change_class_index = int(not bool(no_change_class_index))
    lower_bound = np.min(pred_change + gt_change)
    upper_bound = np.max(pred_change + gt_change)
    all_pairs = [[i, i + 1] for i in range(lower_bound, upper_bound)]
    encoded_pred = [
        change_class_index if pair in pred_change else no_change_class_index for
        pair in all_pairs]
    encoded_gt = [
        change_class_index if pair in gt_change else no_change_class_index for
        pair in all_pairs]
    return jaccard_score(encoded_gt, encoded_pred)


def ab_change_bin_acc(pred_change: list, gt_change: list) -> Tuple[int, int]:
    num_pos = len(gt_change)
    pred_neg_pairs = [pair for pair in pred_change if pair not in gt_change]
    pred_pos_pairs = [pair for pair in gt_change if pair in pred_change]
    num_neg = len(pred_neg_pairs)

    total = num_pos * 2
    if num_pos <= num_neg:
        correct = len(pred_pos_pairs)
    else:
        correct = len(pred_pos_pairs) + (num_pos - num_neg)
    return correct, total
