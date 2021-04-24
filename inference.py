import math
import regex
import subprocess
import sys
from typing import List

import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.neighbors import KNeighborsClassifier
import spacy

from common import BBT_PEOPLE, BoundingBox
from image_processing import ObjectDetector
import language_processing as lp
import utils

frame_folder = '../TVQA_frames/frames_hq/bbt_frames/'
face_collection_path = './face_collection_v2.npz'

od = ObjectDetector()
nlp = spacy.load('en_core_web_sm')

arr = np.load(face_collection_path)
faces = arr['encoded_faces']
labels = arr['labels']

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(faces, labels)


def time_span_to_timestamps_list(test: dict) -> List[int]:
    start, end = test['ts']
    s_ts = max(1, math.floor(start) * 3)
    e_ts = math.ceil(end) * 3

    return list(range(s_ts, e_ts + 1))


def get_gt_timestamps_list(test: dict) -> List[int]:
    return sorted([int(k) for k in test['bbox'].keys()])


def _get_facts_helper(bboxes: List[BoundingBox], file):
    all_labels = dict()
    for box in bboxes:
        curr_l = box.label.lower()
        if curr_l in BBT_PEOPLE:
            box.label = curr_l
        elif curr_l in all_labels:
            all_labels[curr_l] += 1
            box.label = F'{curr_l}_{all_labels[curr_l]}'
        else:
            all_labels[curr_l] = 1
            box.label = F'{curr_l}_{all_labels[curr_l]}'
        print(box.gen_pred() + '.', file=file)

    all_intersections = utils.get_all_intersections(bboxes)

    all_labels = set()
    for inter in all_intersections:
        all_labels.add(inter.b1_label)
        all_labels.add(inter.b2_label)
        print(inter.gen_pred() + '.', file=file)

    for label in all_labels:
        if label in BBT_PEOPLE:
            print(F'person({label}).', file=file)
        elif label == 'person':
            print(F'person(unknown).', file=file)
        else:
            print(F'object({label}).', file=file)


def get_all_facts(test: dict, threshold: float, timestamp: int,
                  file=sys.stdout):
    print(F'time({timestamp}).', file=file)
    print(F'time({timestamp + 1}).', file=file)

    vid_folder = frame_folder + test['vid_name'] + '/'
    qa_objects = od.get_frame_qa_objects(vid_folder, threshold, timestamp)

    human_faces, valid_idx = od.get_human_faces(qa_objects)
    test_encoded_human_faces = od.encode_faces(human_faces)
    if len(test_encoded_human_faces) == 0:
        return
    neigh_predictions = neigh.predict(test_encoded_human_faces)

    for n_i, v_i in enumerate(valid_idx):
        qa_objects[v_i].bbox.label = neigh_predictions[n_i].lower()

    bboxes = [o.bbox for o in qa_objects]

    _get_facts_helper(bboxes, file)


def get_gt_facts(test: dict, timestamp: int, file=sys.stdout):
    print(F'time({timestamp}).', file=file)
    print(F'time({timestamp + 1}).', file=file)

    bboxes = [utils.json_to_bounding_box(bbox)
              for bbox in test['bbox'][str(timestamp)]]

    for bbox in bboxes:
        bbox.label = bbox.label.lower()

    _get_facts_helper(bboxes, file)


def inference(test: dict, gt_object: bool = False) -> List[int]:
    search_results = []

    # Run clingo with learnt rules and facts from the object detection
    with open('temp.lp', 'w') as f:

        timestamps_list = get_gt_timestamps_list(test) if gt_object \
            else time_span_to_timestamps_list(test)

        for t in timestamps_list:
            f.seek(0)

            if gt_object:
                get_gt_facts(test, t, file=f)
            else:
                get_all_facts(test, 0.7, t, file=f)

            f.truncate()

            cp = subprocess.run(['clingo', 'base.lp', 'temp.lp'],
                                capture_output=True)

            s = cp.stdout.decode('utf-8')
            search_results.append(
                regex.search(
                    '.*Answer: [\d]+\\n(?<hold>[^\n]*)\\nSATISFIABLE\\n.*',
                    s).group(1))

    # Get relevant object from the answer set predicates
    action_pred = lp.get_action_pred(test['q'])
    if action_pred is None:
        print(F'qid: {test["qid"]} parsing error')
        print()
        return [-1]

    possible_objects = set()
    for r in list(filter(lambda x: x != '', search_results)):
        for h in r.split(' '):
            s = regex.search(
                'holdsAt\((?<action>.*)\((?<subj>.*),(?<obj>.*)\),[\d]+\)', h)
            if s.group(1).lower() == action_pred.action.lower() and \
                    s.group(2).lower() == action_pred.subj.lower():
                obj = regex.sub('\d', '', s.group(3))
                possible_objects.add(regex.sub('_', ' ', obj))

    gt_ans_idx = test['answer_idx']
    gt_answer = test[F'a{gt_ans_idx}']
    # Match objects with multiple choices
    answer_index = []
    for i in range(5):
        answer = test[F'a{i}']

        root_ans = lp.get_root_obj_token(answer)
        action_ans = lp.get_action_obj_token(answer)

        if root_ans:
            for p_o in possible_objects:
                if lp.check_synonyms(p_o, root_ans.text):
                    answer_index.append(i)
        elif action_ans:
            for p_o in possible_objects:
                if lp.check_synonyms(p_o, action_ans.text):
                    answer_index.append(i)
        else:
            for p_o in possible_objects:
                if p_o in answer.lower():
                    answer_index.append(i)

    print(F'qid: {test["qid"]}   ans_idx: {gt_ans_idx}    gt_ans: {gt_answer}')
    print(F'poss_obj: {possible_objects}   pred_ans_idx: {answer_index}')

    return answer_index


def get_jaccard_score(pred_idx_list: List[int], test: dict) -> float:
    y_true = np.zeros(5)
    y_true[int(test['answer_idx'])] = 1

    y_pred = np.zeros(5)
    y_pred[pred_idx_list] = 1

    return jaccard_score(y_true, y_pred)
