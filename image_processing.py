import os
from PIL import Image
import pickle
import re

import numpy as np

import cv2
from dlib import face_recognition_model_v1
from face_recognition_models import face_recognition_model_location
from sklearn.cluster import MiniBatchKMeans
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from typing import List, Tuple

from common import COCO_INSTANCE_CATEGORY_NAMES
from common import QaObject, GroundingBox


def torchvision_bbox_to_coco_bbox(img_id,
                                  bbox: List[float],
                                  label: str) -> GroundingBox:
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]

    tl_x = int(min(x0, x1))
    tl_y = int(min(y0, y1))
    width = int(abs(x1 - x0))
    height = int(abs(y1 - y0))

    return GroundingBox(img_id, tl_x, tl_y, width, height, label)


def bbox_to_xy_coordinates(bbox: GroundingBox) -> (int, int, int, int):
    return bbox.tl_x, bbox.tl_y, bbox.tl_x + bbox.width, bbox.tl_y + bbox.height


def draw_bounding_box(img_path, bboxes: List[GroundingBox]):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox_to_xy_coordinates(bbox)
        image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return image


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

    def get_rcnn_qa_object_outs(self, folder_path: str, threshold: float,
                                transforms=None) -> List[QaObject]:
        if transforms is None:
            transforms = self.transforms

        # Get all images from the dir
        l = os.listdir(folder_path)
        l = list(filter(lambda f: f.endswith('.jpg'), l))
        l.sort(key=lambda f: int(re.sub('\D', '', f)))

        # Open the images and transform them
        imgs = [transforms(Image.open(folder_path + f)
                           .convert('RGB')).to(self.device) for f in l]

        # Feed to the model
        self.frcnn_model.eval()
        with torch.no_grad():
            outs = self.frcnn_model(imgs)

        detected_objs = []

        # Filter out objects with low score, convert to to QA Object
        for j, out in enumerate(outs):
            image_name = l[j]
            boxes = out['boxes']
            labels = out['labels']
            scores = out['scores']
            seen_classes = dict()

            for i in range(len(boxes)):
                score = scores[i].item()
                if score >= threshold:
                    obj_class = COCO_INSTANCE_CATEGORY_NAMES[labels[i].item()]
                    if obj_class not in seen_classes:
                        seen_classes[obj_class] = 1
                    else:
                        seen_classes[obj_class] += 1
                    obj_label = F'{obj_class}{seen_classes[obj_class]}' \
                        .replace(' ', '_')
                    timestamp = int(re.sub('[^0-9]', '', image_name))
                    box = torchvision_bbox_to_coco_bbox(timestamp,
                                                        boxes[i].tolist(),
                                                        obj_label)
                    rcnn_obj = QaObject(obj_class, box, score,
                                        folder_path + image_name, timestamp)

                    detected_objs.append(rcnn_obj)

        return detected_objs

    def get_rcnn_human_faces(self,
                             folder_path: str,
                             detect_threshold: float = 0.7,
                             target_size: Tuple[int, int] = (64, 64)) -> List[
        np.ndarray]:
        qa_objects = self.get_rcnn_qa_object_outs(folder_path, detect_threshold)

        human_faces = []

        for obj in qa_objects:
            # Ignore non-person objects
            if obj.obj_class != 'person':
                continue

            # Get person in the image
            x0, y0, x1, y1 = bbox_to_xy_coordinates(obj.bbox)
            person_bgr = cv2.imread(obj.sample_name)[y0: y1, x0: x1]
            person_gray = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2GRAY)
            person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)

            # Detect faces on the person region and extract faces
            faces = self.face_cascade.detectMultiScale(person_gray, 1.1, 4)
            for (x, y, w, h) in faces:
                human_faces.append(person_rgb[y:y + h, x:x + w])

        # Resample the image to the same size
        resized_faces = [np.array(Image.fromarray(face).resize(target_size)) for
                         face in human_faces]

        return resized_faces


class FaceCluster:
    def __init__(self, num_cluster: int, fit_threshold: int = 20):
        self.num_cluster = num_cluster
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_cluster,
                                      random_state=np.random.RandomState(0))
        self.face_encoder = face_recognition_model_v1(
            face_recognition_model_location())
        self.buffer = None
        self.fit_threshold = fit_threshold

    def _add_to_buffer(self, enc: np.ndarray):
        if self.buffer is None:
            self.buffer = enc
        else:
            self.buffer = np.concatenate((self.buffer, enc))

    def _encode(self, faces: List[np.ndarray]) -> np.ndarray:
        return np.array(self.face_encoder.compute_face_descriptor(faces))

    def _partial_fit(self):
        if self.buffer.shape[0] >= self.fit_threshold:
            self.kmeans.partial_fit(self.buffer)
            self.buffer = None

    def encode_and_partial_fit(self, faces: List[np.ndarray]):
        self._add_to_buffer(self._encode(faces))
        self._partial_fit()

    def end_fitting(self):
        self._partial_fit()
        self.buffer = None
