import logging
import os
from PIL import Image
import re

import numpy as np

import cv2
from dlib import face_recognition_model_v1
from face_recognition_models import face_recognition_model_location
from scipy.optimize import linear_sum_assignment
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from typing import List, Set, Tuple

from common import COCO_INSTANCE_CATEGORY_NAMES, BBT_PEOPLE
from common import QaObject, BoundingBox
import utils


def torchvision_bbox_to_coco_bbox(img_id,
                                  bbox: List[float],
                                  label: str) -> BoundingBox:
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]

    tl_x = int(min(x0, x1))
    tl_y = int(min(y0, y1))
    width = int(abs(x1 - x0))
    height = int(abs(y1 - y0))

    return BoundingBox(img_id, tl_x, tl_y, width, height, label)


def draw_bounding_box(img_path, bboxes: List[BoundingBox]):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox.get_coords()
        image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = re.sub('\d', '', bbox.label)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        l_x0 = x0
        l_y0 = y0
        l_x1 = l_x0 + label_size[0][0]
        l_y1 = l_y0 - int(label_size[0][1])
        image = cv2.rectangle(image, (l_x0, l_y0), (l_x1, l_y1),
                              (0, 255, 0), cv2.FILLED)
        image = cv2.putText(image, label, (l_x0, l_y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def match_person_bounding_boxes(qa_objects: List[QaObject], data: dict) \
        -> List[Tuple[QaObject, str, float]]:
    matched_tuple = []
    for img_id in data['bbox']:
        targets = [(idx, q.bbox) for idx, q in enumerate(qa_objects) if
                   q.bbox.img_id == int(img_id) and 'person' in q.bbox.label]
        target_indices = [idx for idx, _ in targets]
        target_bboxes = [b for _, b in targets]

        gt_bboxes = [utils.json_to_bounding_box(f) for f in data['bbox'][img_id]
                     if f['label'].lower() in BBT_PEOPLE]

        # Construct IOU matrix
        targets_len = len(target_bboxes)
        gt_len = len(gt_bboxes)
        iou_matrix = np.zeros((targets_len, gt_len))
        for i in range(targets_len):
            for j in range(gt_len):
                iou_matrix[i, j] = target_bboxes[i].get_iou_score(gt_bboxes[j])

        # Compute assignment that max the overall IOU score
        row_idx, col_idx = linear_sum_assignment(iou_matrix, maximize=True)

        matched_tuple += [
            (qa_objects[target_indices[i]],
             gt_bboxes[j].label,
             iou_matrix[i, j])
            for i, j in zip(row_idx, col_idx)]

    return matched_tuple


class ObjectDetector:
    def __init__(self, use_gpu=True):
        cuda_available = torch.cuda.is_available()
        if not cuda_available or not use_gpu:
            self.device = 'cpu'
        else:
            self.device = 'cuda:0'

        self.frcnn_model = fasterrcnn(pretrained=True)
        self.frcnn_model.to(self.device)

        self.transforms = T.Compose([T.ToTensor()])

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_encoder = face_recognition_model_v1(
            face_recognition_model_location())

    def get_frame_qa_objects(self,
                             folder_path: str,
                             threshold: float,
                             timestamp: int,
                             transforms=None) -> List[QaObject]:
        if transforms is None:
            transforms = self.transforms

        image_list = list(
            filter(lambda f: int(re.sub('\D', '', f)) == timestamp,
                   os.listdir(folder_path))
        )
        if len(image_list) != 1:
            logging.warning(
                F'Try to find {timestamp} in {folder_path}, '
                F'found {len(image_list)} img(s)')
            return []

        img_name = folder_path + image_list[0]
        detected_objs = []

        self.frcnn_model.eval()
        with torch.no_grad():
            img = transforms(Image.open(img_name).convert('RGB')).to(
                self.device).unsqueeze(0)

            # Feed to the model
            outs = self.frcnn_model(img)

            seen_classes = dict()
            for out in outs:
                boxes = out['boxes']
                labels = out['labels']
                scores = out['scores']

                for i in range(len(boxes)):
                    score = scores[i].item()

                    # Filter out objects with low score
                    if score < threshold:
                        continue

                    obj_class = COCO_INSTANCE_CATEGORY_NAMES[
                        labels[i].item()]

                    if obj_class == 'person':
                        obj_label = 'person'
                    else:
                        if obj_class not in seen_classes:
                            seen_classes[obj_class] = 1
                        else:
                            seen_classes[obj_class] += 1

                        obj_label = \
                            F'{obj_class}{seen_classes[obj_class]}'.replace(
                                ' ', '_')

                    box = torchvision_bbox_to_coco_bbox(timestamp,
                                                        boxes[i].tolist(),
                                                        obj_label)

                    detected_objs.append(
                        QaObject(obj_class, box, score, img_name, timestamp)
                    )

            # Clear CUDA cache after each frame
            torch.cuda.empty_cache()

        return detected_objs

    def get_rcnn_qa_objects(self,
                            folder_path: str,
                            threshold: float,
                            specific_imgs: Set[int] = None,
                            transforms=None) -> List[QaObject]:
        if transforms is None:
            transforms = self.transforms

        # Get images from the dir
        l = list(filter(lambda f: f.endswith('.jpg'), os.listdir(folder_path)))
        if specific_imgs:
            l = list(
                filter(lambda f: int(re.sub('\D', '', f)) in specific_imgs, l)
            )
        l.sort(key=lambda f: int(re.sub('\D', '', f)))

        detected_objs = []
        seen_classes = dict()

        self.frcnn_model.eval()
        with torch.no_grad():
            # Process frame by frame
            for img_name in l:
                # Open the img and transform
                img = transforms(
                    Image.open(folder_path + img_name).convert('RGB')
                ).to(self.device).unsqueeze(0)

                # Feed to the model
                outs = self.frcnn_model(img)

                for out in outs:
                    boxes = out['boxes']
                    labels = out['labels']
                    scores = out['scores']

                    for i in range(len(boxes)):
                        score = scores[i].item()

                        # Filter out objects with low score
                        if score < threshold:
                            continue

                        obj_class = COCO_INSTANCE_CATEGORY_NAMES[
                            labels[i].item()]

                        if obj_class not in seen_classes:
                            seen_classes[obj_class] = 1
                        else:
                            seen_classes[obj_class] += 1

                        obj_label = \
                            F'{obj_class}{seen_classes[obj_class]}'.replace(
                                ' ', '_')

                        timestamp = int(re.sub('\D', '', img_name))

                        box = torchvision_bbox_to_coco_bbox(
                            timestamp,
                            boxes[i].tolist(),
                            obj_label)

                        rcnn_obj = QaObject(obj_class, box, score,
                                            folder_path + img_name,
                                            timestamp)

                        detected_objs.append(rcnn_obj)

                # Clear CUDA cache after each frame
                torch.cuda.empty_cache()

        return detected_objs

    def get_human_faces(self, qa_objects: List[QaObject]) -> \
            Tuple[List[np.ndarray], List[int]]:
        human_faces = []
        valid_qa_objects_idx = []

        for i, obj in enumerate(qa_objects):
            # Ignore non-person objects
            if obj.obj_class != 'person':
                continue

            # Get person in the image
            x0, y0, x1, y1 = obj.bbox.get_coords()
            person_bgr = cv2.imread(obj.sample_name)[y0: y1, x0: x1]
            person_gray = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2GRAY)
            person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)

            # Detect faces on the person region and extract faces
            faces = self.face_cascade.detectMultiScale(person_gray, 1.1, 4)
            if len(faces) < 1:
                continue
            (x, y, w, h) = faces[0]
            human_faces.append(person_rgb[y:y + h, x:x + w])
            valid_qa_objects_idx.append(i)

        # Resample the image to the same size
        target_size = (150, 150)
        resized_faces = [np.array(Image.fromarray(face).resize(target_size))
                         for face in human_faces]

        return resized_faces, valid_qa_objects_idx

    def encode_faces(self, faces: List[np.ndarray]) -> np.ndarray:
        return np.array(self.face_encoder.compute_face_descriptor(faces))
