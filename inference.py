import regex
import subprocess
import sys
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from sklearn.neighbors import KNeighborsClassifier
import spacy

import abrupt_transition_detection as atd
from common import BBT_PEOPLE, BoundingBox, FRAME_FOLDER, \
    FACE_COLLECTION_V2_NPZ, SCENE_TRANSITION_NPZ
from image_processing import ObjectDetector
import language_processing as lp
import utils

od = ObjectDetector()
nlp = spacy.load('en_core_web_sm')

# KNN classifier for face detection
arr = np.load(FACE_COLLECTION_V2_NPZ)
faces = arr['encoded_faces']
labels = arr['labels']

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(faces, labels)

# K-means cluster for abrupt transition detection
arr = np.load(SCENE_TRANSITION_NPZ)
all_diff = arr['hist_all']
kmeans = KMeans(n_clusters=2, random_state=0).fit(
    np.expand_dims(all_diff, axis=1))


def get_gt_timestamps_list(test: dict) -> List[int]:
    return sorted([int(k) for k in test['bbox'].keys()])


def __get_facts_helper_h(bboxes: List[BoundingBox], file):
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


def get_all_facts_h(test: dict, threshold: float, timestamp: int,
                    file=sys.stdout):
    print(F'time({timestamp}).', file=file)
    print(F'time({timestamp + 1}).', file=file)

    vid_folder = FRAME_FOLDER + test['vid_name'] + '/'
    qa_objects = od.get_frame_qa_objects(vid_folder, threshold, timestamp)

    human_faces, valid_idx = od.get_human_faces(qa_objects)
    test_encoded_human_faces = od.encode_faces(human_faces)
    if len(test_encoded_human_faces) == 0:
        return
    neigh_predictions = neigh.predict(test_encoded_human_faces)

    for n_i, v_i in enumerate(valid_idx):
        qa_objects[v_i].bbox.label = neigh_predictions[n_i].lower()

    bboxes = [o.bbox for o in qa_objects]

    __get_facts_helper_h(bboxes, file)


def get_gt_facts_h(test: dict, timestamp: int, file=sys.stdout):
    print(F'time({timestamp}).', file=file)
    print(F'time({timestamp + 1}).', file=file)

    bboxes = [utils.json_to_bounding_box(bbox)
              for bbox in test['bbox'][str(timestamp)]]

    for bbox in bboxes:
        bbox.label = bbox.label.lower()

    __get_facts_helper_h(bboxes, file)


def inference_h(test: dict, gt_object: bool = False, log: bool = False) \
        -> List[int]:
    """
    HACR-H inference
    :param test: dict, json data
    :param gt_object: bool, whether to use ground truth objects
    :param log: bool, whether to log errors etc.
    :return: List[int], index of answer(s)
    """
    search_results = []

    # Run clingo with learnt rules and facts from the object detection
    with open('temp.lp', 'w') as f:

        timestamps_list = get_gt_timestamps_list(test) if gt_object \
            else utils.time_span_to_timestamps_list(test)

        for t in timestamps_list:
            f.seek(0)

            if gt_object:
                get_gt_facts_h(test, t, file=f)
            else:
                get_all_facts_h(test, 0.7, t, file=f)

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
        if log:
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
                obj = regex.sub('_\d+', '', s.group(3))
                obj = regex.sub('\d+$', '', obj)
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

    if log:
        print(
            F'qid: {test["qid"]}   ans_idx: {gt_ans_idx}    gt_ans: {gt_answer}')
        print(F'poss_obj: {possible_objects}   pred_ans_idx: {answer_index}')

    return answer_index


def __get_subclause_people(text):
    subclause_words = ['when', 'after', 'before']

    for i, w in enumerate(text.split()):
        if i != 0 and w.lower() in subclause_words:
            subclause = ' '.join(text.split()[i + 1:])
            return lp.get_all_people_in_ans(subclause)


def get_facts_e(data: dict, gt_human_face: bool = False, gt_abt: bool = False,
                diff_score_method: str = 'hist-all', file=sys.stdout):
    vid_folder = FRAME_FOLDER + data['vid_name'] + '/'

    # Get time and abrupt transition
    if gt_abt:
        times = utils.time_span_to_timestamps_list(data)

        for pair in data['scene_change_pairs']:
            print(F'abrupt_transition({pair[0]}, {pair[1]}).', file=file)
    else:
        res = atd.gen_pixel_diff(data, use_time_span=True,
                                 score_method=diff_score_method)

        times = res['timestamps']
        print(F'time({min(times)}..{max(times)}).', file=file)

        diff_score = res['pixel_diff_score']
        no_change_class_index = kmeans.predict([[0]])[0]
        preds = kmeans.predict(np.expand_dims(diff_score, axis=1))
        new_score = atd.non_max_suppression(diff_score, preds,
                                            no_change_class_index)
        new_pred = kmeans.predict(np.expand_dims(new_score, axis=1))
        pred_ab_change_list = [[times[j], times[j + 1]] for j, pred in
                               enumerate(new_pred) if
                               pred != no_change_class_index]
        for t1, t2 in pred_ab_change_list:
            print(F'abrupt_transition({t1}, {t2}).', file=file)

    # Get in_scene
    all_people = []
    if gt_human_face:
        for person in data['in_scene']:
            all_people.append(person)
            for pair in data['in_scene'][person]:
                print(F'holdsAt(in_scene({person}), {pair[0]}..{pair[1]}).',
                      file=file)
    else:
        for time in times:
            qa_objects = od.get_frame_qa_objects(vid_folder, 0.7, time)
            people = [o for o in qa_objects if o.obj_class == 'person']
            human_faces, valid_idx = od.get_human_faces(people)
            if len(human_faces) == 0:
                continue
            neigh_predictions = neigh.predict(od.encode_faces(human_faces))
            for pred in neigh_predictions:
                print(F'holdsAt(in_scene({pred.lower()}), {time}).', file=file)
                if pred.lower() not in all_people:
                    all_people.append(pred.lower())

    for p in all_people:
        print(F'person({p}).', file=file)

    # Assume the people mentioned in the subclauses would be at the
    # current scene at the start of the time span
    for p in __get_subclause_people(data['q']):
        p_name = p.lower()
        print(F'holdsAt(at_curr_location({p_name}), {min(times)}).', file=file)
        if p_name not in all_people:
            print(F'person({p_name}).', file=file)
            all_people.append(p_name)


def inference_e(test: dict, gt_human_face: bool = False, gt_abt: bool = False,
                log: bool = False) -> List[int]:
    """
    HACR-E inference
    :param test: dict, json data
    :param gt_human_face: bool, whether to use ground truth human in scene
    :param gt_abt: bool, whether to use ground truth abrupt transition
    :param log: bool, whether to log errors etc.
    :return: List[int], index of answer(s)
    """
    search_results = []

    # Run clingo with learnt rules and facts from the object detection
    with open('temp.lp', 'w') as f:

        f.seek(0)
        get_facts_e(test, file=f, gt_human_face=gt_human_face, gt_abt=gt_abt)
        f.truncate()

        cp = subprocess.run(['clingo', 'base_enter.lp', 'temp.lp'],
                            capture_output=True)

        s = cp.stdout.decode('utf-8')
        search_results.append(
            regex.search(
                '.*Answer: [\d]+\\n(?<enter>[^\n]*)\\nSATISFIABLE\\n.*',
                s).group(1))

    possible_people = set()
    for r in list(filter(lambda x: x != '', search_results)):
        for e in r.split(' '):
            s = regex.search(
                'initiates\(enter\((?<person>[a-z]+)\),.*,[\d]+\)', e)
            possible_people.add(s.group(1).lower())

    gt_ans_idx = test['answer_idx']
    gt_answer = test[F'a{gt_ans_idx}']
    # Match objects with multiple choices
    answer_index = []
    single_match = False
    for i in range(5):
        if not single_match:
            answer = test[F'a{i}']
            answer_people_list = set(lp.get_all_people_in_ans(answer))
            if answer_people_list == possible_people:
                single_match = True
                answer_index = [i]
            elif answer_people_list.intersection(possible_people):
                answer_index.append(i)

    if log:
        print(
            F'qid: {test["qid"]}   ans_idx: {gt_ans_idx}    gt_ans: {gt_answer}')
        print(F'poss_people: {possible_people}   pred_ans_idx: {answer_index}')

    return answer_index


def get_jaccard_score(pred_idx_list: List[int], test: dict) -> float:
    y_true = np.zeros(5)
    y_true[int(test['answer_idx'])] = 1

    y_pred = np.zeros(5)
    y_pred[pred_idx_list] = 1

    return jaccard_score(y_true, y_pred)
