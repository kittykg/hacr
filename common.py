from dataclasses import dataclass

from typing import List

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


@dataclass
class GroundingBox:
    tl_x: int
    tl_y: int
    width: int
    height: int


@dataclass
class QaObject:
    obj_class: str
    bbox: GroundingBox
    score: float
    sample_name: str
    img_id: str


@dataclass
class ActionPred:
    action: str
    subj: str
    obj: str

    def gen_pred(self):
        return F'{self.action}({self.subj.lower()}, {self.obj.lower()})'

    def __str__(self):
        return self.gen_pred()


def torchvision_bbox_to_coco_bbox(bbox: List[float]) -> GroundingBox:
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]

    tl_x = int(min(x0, x1))
    tl_y = int(min(y0, y1))
    width = int(abs(x1 - x0))
    height = int(abs(y1 - y0))

    return GroundingBox(tl_x, tl_y, width, height)
