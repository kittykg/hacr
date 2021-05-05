import os
import re

import cv2
import numpy as np

from common import FRAME_FOLDER
from utils import time_span_to_timestamps_list


def gen_pixel_diff(data: dict, use_time_span: bool = False):
    vid_folder = FRAME_FOLDER + data['vid_name'] + '/'
    l = list(filter(lambda f: f.endswith('.jpg'), os.listdir(vid_folder)))
    l.sort(key=lambda f: int(re.sub('\D', '', f)))

    if use_time_span:
        timestamps = time_span_to_timestamps_list(data)
    else:
        timestamps = list(range(1, len(l) + 1))

    imgs = [cv2.cvtColor(cv2.imread(vid_folder + l[i]), cv2.COLOR_BGR2RGB)
            for i in range(len(l)) if (i + 1) in timestamps]

    img_area = imgs[0].shape[0] * imgs[0].shape[1]
    scores = []

    for i in range(len(imgs) - 1):
        img_1 = imgs[i]
        img_2 = imgs[i + 1]
        diff = np.sum(np.absolute(img_2 - img_1)) / img_area
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
