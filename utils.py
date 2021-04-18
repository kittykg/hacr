import itertools
import random
from typing import List

from common import BoundingBox, BBoxIntersectionPred


def get_all_intersections(boxes: List[BoundingBox]) \
        -> List[BBoxIntersectionPred]:
    intersections = []
    for b1, b2 in itertools.combinations(boxes, 2):
        if b1.img_id != b2.img_id:
            continue

        intersection_area = b1.get_intersection_area(b2)
        b1_area = b1.get_area()
        b2_area = b2.get_area()

        frac_wrt_b1 = min(100, int(intersection_area / b1_area * 100))
        frac_wrt_b2 = min(100, int(intersection_area / b2_area * 100))

        if frac_wrt_b1 > 0:
            intersections.append(BBoxIntersectionPred(b1.img_id,
                                                      b1.label,
                                                      b2.label,
                                                      frac_wrt_b1))

        if frac_wrt_b2 > 0:
            intersections.append(BBoxIntersectionPred(b2.img_id,
                                                      b2.label,
                                                      b1.label,
                                                      frac_wrt_b2))

    return intersections


def json_to_bounding_box(bbox: dict) -> BoundingBox:
    img_id = bbox['img_id']
    tl_x = bbox['left']
    tl_y = bbox['top']
    width = bbox['width']
    height = bbox['height']
    label = bbox['label']
    return BoundingBox(img_id, tl_x, tl_y, width, height, label)


def split_data_set(proportion: float, data: list):
    random.shuffle(data)
    train_size = int(len(data) * proportion)
    return data[:train_size], data[train_size:]
