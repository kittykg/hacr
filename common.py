from dataclasses import dataclass

from typing import List

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


@dataclass
class GroundingBox:
    img_id: int
    tl_x: int
    tl_y: int
    width: int
    height: int
    label: str

    def get_coords(self):
        x0 = self.tl_x
        x1 = x0 + self.width
        y0 = self.tl_y
        y1 = y0 + self.height

        return x0, y0, x1, y1

    def get_area(self):
        return self.height * self.width

    def get_intersection_area(self, b2) -> float:
        b1_x0, b1_y0, b1_x1, b1_y1 = self.get_coords()
        b2_x0, b2_y0, b2_x1, b2_y1 = b2.get_coords()

        inter_x1 = max(b1_x0, b2_x0)
        inter_x2 = min(b1_x1, b2_x1)
        inter_y1 = max(b1_y0, b2_y0)
        inter_y2 = min(b1_y1, b2_y1)

        return max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    def gen_pred(self):
        return F'bbox({self.img_id}, {self.label}, {self.tl_x}, ' \
               F'{self.tl_y}, {self.width}, {self.height})'


@dataclass
class BBoxIntersectionPred:
    frame_id: int
    b1_label: str
    b2_label: str
    frac_wrt_b1: int

    def gen_pred(self):
        return F'bbox_intersec({self.frame_id}, {self.b1_label}, ' \
               F'{self.b2_label}, {self.frac_wrt_b1})'


@dataclass
class QaObject:
    obj_class: str
    bbox: GroundingBox
    score: float
    sample_name: str
    timestamp: int


@dataclass
class ActionPred:
    action: str
    subj: str
    obj: str

    def gen_pred(self):
        return F'{self.action}({self.subj.lower()}, {self.obj.lower()})'

    def __str__(self):
        return self.gen_pred()


@dataclass
class Example:
    qid: int
    vid_name: str


@dataclass
class PositiveExample(Example):
    curr_time: int
    facts: List[str]

    def gen_example(self):
        eg_id = F'p_{self.qid}_{self.curr_time}'
        facts_list = '\n'.join(map(lambda f: F'    {f}', self.facts))
        return F'#pos({eg_id}@10, {{}}, {{}}, {{\n' + facts_list + '\n}).'


@dataclass
class NegativeExample(Example):
    curr_time: int
    facts: List[str]
