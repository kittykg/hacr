import math
import regex
import subprocess
import sys
from typing import List

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import spacy

from common import BBT_PEOPLE
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
    for box in bboxes:
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


def inference(test: dict) -> List[int]:
    search_results = []

    # Run clingo with learnt rules and facts from the object detection
    with open('temp.lp', 'w') as f:
        for t in time_span_to_timestamps_list(test):
            f.seek(0)

            get_all_facts(test, 0.7, t, file=f)
            cp = subprocess.run(['clingo', 'base.lp', 'temp.lp'],
                                capture_output=True)

            f.truncate()

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
        return []

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
        answer_object = test[F'a{i}']
        doc = nlp(answer_object)
        for w in doc:
            if w.pos_ == 'NOUN' and w not in BBT_PEOPLE:
                for p_o in possible_objects:
                    if lp.check_synonyms(p_o, w.lemma_):
                        answer_index.append(i)

    print(F'qid: {test["qid"]}   ans_idx: {gt_ans_idx}    gt_ans: {gt_answer}')
    print(F'poss_obj: {possible_objects}   pred_ans_idx: {answer_index}')
    print()

    return answer_index
