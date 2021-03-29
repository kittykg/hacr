import os
from PIL import Image
import re

import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from typing import List

from common import COCO_INSTANCE_CATEGORY_NAMES
from common import QaObject, torchvision_bbox_to_coco_bbox


class ObjectDetector:
    def __init__(self):
        self.frcnn_model = fasterrcnn(pretrained=True)

        self.transforms = T.Compose([T.ToTensor()])

    def get_example_rcnn_outs(self, folder_path: str, threshold: float, transforms=None) -> List[QaObject]:
        if transforms is None:
            transforms = self.transforms

        # Get all images from the dir
        l = os.listdir(folder_path)
        l = list(filter(lambda f: f.endswith('.jpg'), l))
        l.sort(key=lambda f: int(re.sub('\D', '', f)))

        # Open the images and transform them
        imgs = [transforms(Image.open(folder_path + f).convert('RGB')) for f in l]

        # Feed to the model
        self.frcnn_model.eval()
        with torch.no_grad():
            outs = self.frcnn_model(imgs)

        detected_objs = []

        # Filter out objects with low score, convert to
        for j, out in enumerate(outs):
            image_name = l[j]
            boxes = out['boxes']
            labels = out['labels']
            scores = out['scores']

            for i in range(len(boxes)):
                score = scores[i].item()
                if score >= threshold:
                    obj_class = COCO_INSTANCE_CATEGORY_NAMES[labels[i].item()]
                    box = torchvision_bbox_to_coco_bbox(boxes[i].tolist())
                    rcnn_obj = QaObject(obj_class, box, score, folder_path, image_name)

                    detected_objs.append(rcnn_obj)

        return detected_objs
