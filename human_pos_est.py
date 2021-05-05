import copy
import math
from operator import itemgetter
from typing import Union, Dict, Tuple

from PIL import Image
import sys

import cv2
import numpy as np

import torch.optim
import torchvision.transforms as T

sys.path.append('../human-pose-estimation.pytorch/lib/')
import models
from core.config import config
from core.config import update_config
from common import BoundingBox

CONFIG_FILE = \
    '../human-pose-estimation.pytorch/experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml'
MODEL_PATH = '../human-pose-estimation.pytorch/pose_resnet_50_256x256.pth.tar'

update_config(CONFIG_FILE)

model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(config,
                                                              is_train=False)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device('cuda:0')))

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

JOINTS = ['r-ankle', 'r-knee', 'r-hip', 'l-hip', 'l-knee', 'l-ankle', 'pelvis',
          'thorax', 'upper-neck', 'head-top', 'r-wrist', 'r-elbow',
          'r-shoulder', 'l-shoulder', 'l-elbow', 'l-wrist']

POSE_PAIRS = [
    # UPPER BODY
    [9, 8],
    [8, 7],
    [7, 6],

    # # LOWER BODY
    [6, 2],
    [2, 1],
    [1, 0],

    [6, 3],
    [3, 4],
    [4, 5],

    # ARMS
    [7, 12],
    [12, 11],
    [11, 10],

    [7, 13],
    [13, 14],
    [14, 15]
]


def get_centered_padding_img(bbox: BoundingBox, image_file: str) \
        -> Tuple[np.ndarray, Image.Image]:
    # b is bbox
    x0, y0, x1, y1 = bbox.get_coords()
    person_bgr = cv2.imread(image_file)[y0: y1, x0: x1]
    person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)

    ht, wd, cc = person_rgb.shape

    hh = max(ht, wd)
    ww = hh
    color = (0, 0, 0)
    result = np.full((hh, ww, cc), color, dtype=np.uint8)

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    result[yy:yy + ht, xx:xx + wd] = person_rgb

    return result, Image.fromarray(result)


def get_detached(x):
    return copy.deepcopy(x.cpu().detach().numpy())


def get_key_points(pose_layers):
    return list(map(itemgetter(1, 3),
                    [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers]))


def get_angle(v_a: np.ndarray, v_b: np.ndarray) -> float:
    d_p = np.dot(v_a, v_b)
    m_a = np.linalg.norm(v_a)
    m_b = np.linalg.norm(v_b)
    cos = d_p / m_a / m_b
    ang = math.degrees(math.acos(cos)) % 360
    if ang - 180 >= 0:
        return 360 - ang
    else:
        return ang


def get_angle_between_lines(key_points: list,
                            l1_end_idx: int, l2_end_idx: int, joining_idx: int,
                            threshold: float = 0.5) -> Union[int, None]:
    line1_thr, (line1_x, line1_y) = key_points[l1_end_idx]
    line2_thr, (line2_x, line2_y) = key_points[l2_end_idx]
    joining_thr, (joining_x, joining_y) = key_points[joining_idx]

    if line1_thr > threshold and line2_thr > threshold and \
            joining_thr > threshold:
        va = np.array([(line1_x - joining_x), (line1_y - joining_y)])
        vb = np.array([(line2_x - joining_x), (line2_y - joining_y)])

        return int(get_angle(va, vb))
    return None


def get_legs_movement(bbox: BoundingBox, img_file: str) \
        -> Dict[str, Union[int, None]]:
    _, centered_img = get_centered_padding_img(bbox, img_file)
    output = model(transform(centered_img).unsqueeze(0)).squeeze(0)

    pose_layers = get_detached(output)
    key_points = get_key_points(pose_layers)

    # Left hip and lap
    l_h_l = get_angle_between_lines(key_points, 6, 4, 3)
    # Left lap and calf
    l_l_c = get_angle_between_lines(key_points, 3, 5, 4)
    # Right hip and lap
    r_h_l = get_angle_between_lines(key_points, 6, 1, 2)
    # Right lap and calf
    r_l_c = get_angle_between_lines(key_points, 2, 0, 1)

    return {'l_h_l': l_h_l, 'l_l_c': l_l_c, 'r_h_l': r_h_l, 'r_l_c': r_l_c}


def draw_pose_sticks(bbox: BoundingBox, img_file: str, threshold: float = 0.5):
    result_arr, centered_img = get_centered_padding_img(bbox, img_file)
    img_h, img_w, _ = result_arr.shape

    output = model(transform(centered_img).unsqueeze(0)).squeeze(0)
    _, out_h, out_w = output.shape

    pose_layers = get_detached(output)
    key_points = get_key_points(pose_layers)

    is_joint_plotted = [False] * len(JOINTS)

    x_ratio = img_h / out_h
    y_ratio = img_w / out_w

    for pose_pair in POSE_PAIRS:
        from_j, to_j = pose_pair

        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j) = key_points[to_j]

        from_x_j *= x_ratio
        to_x_j *= x_ratio
        from_y_j *= y_ratio
        to_y_j *= y_ratio

        from_x_j, to_x_j = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j = int(from_y_j), int(to_y_j)

        # Plot joints if they haven't been plotted
        if from_thr > threshold and not is_joint_plotted[from_j]:
            cv2.ellipse(result_arr, (from_x_j, from_y_j), (4, 4), 0, 0, 360,
                        (255, 255, 255), cv2.FILLED)
            is_joint_plotted[from_j] = True
        if to_thr > threshold and not is_joint_plotted[to_j]:
            cv2.ellipse(result_arr, (to_x_j, to_y_j), (4, 4), 0, 0, 360,
                        (255, 255, 255), cv2.FILLED)
            is_joint_plotted[to_j] = True

        # Plot the body stick
        if from_thr > threshold and to_thr > threshold:
            cv2.line(result_arr, (from_x_j, from_y_j), (to_x_j, to_y_j),
                     (255, 74, 0), 3)

    return Image.fromarray(cv2.cvtColor(result_arr, cv2.COLOR_RGB2BGR))
