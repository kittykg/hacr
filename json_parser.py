import logging
from typing import List, Union, Set

from utils import get_all_intersections

import language_processing as lp
from common import PositiveExample, BoundingBox, ActionPred, BBT_PEOPLE


def get_img_ids(data: dict) -> Set[int]:
    return set([int(k) for k in data['bbox'].keys()])


def get_answer_obj(ans_text: str) -> Union[str, None]:
    obj = None

    ans_root = lp.get_root_obj_token(ans_text)
    if ans_root is not None:
        obj = ans_root.lemma_
    else:
        ans_act_obj = lp.get_action_obj_token(ans_text)
        if ans_act_obj is not None:
            obj = ans_act_obj.lemma_

    return obj


class Parser:
    def __init__(self):
        self.relational_object_pairs = {
            'piece': 'clothing',
            'remote': 'tv'
        }

    def match_ans_obj_with_bbox_obj(self, ans_obj: str,
                                    bbox_obj_set: Set[str]) -> Union[str, None]:
        for bbox_obj in bbox_obj_set:
            noun_tag = 'NOUN'
            if lp.check_synonyms(ans_obj, bbox_obj, noun_tag) or \
                    lp.check_hyponyms(ans_obj, bbox_obj, noun_tag) or \
                    lp.check_hypernyms(ans_obj, bbox_obj, noun_tag):
                return bbox_obj
            elif ans_obj in self.relational_object_pairs:
                return self.relational_object_pairs[ans_obj]
            elif bbox_obj in self.relational_object_pairs:
                return self.relational_object_pairs[bbox_obj]
        return None

    def get_goal_action_obj(self, data: dict, bbox_obj_set: Set[str],
                            log: bool = False) -> \
            Union[str, None]:

        answer_idx = data['answer_idx']
        ans_text = data['a' + answer_idx]
        ans_obj = get_answer_obj(ans_text)

        # Stop if we can't parse the correct answer
        if ans_obj is None:
            if log:
                logging.warning(F'Cannot parse correct answer {ans_text}')
            return None

        obj = self.match_ans_obj_with_bbox_obj(ans_obj, bbox_obj_set)

        # Stop if we can't match the object in the correct answer with
        # an object in our grounding box
        if obj is None:
            if log:
                logging.warning(
                    F'Cannot match {ans_obj} with set {bbox_obj_set}')

        return obj

    def get_pos_example(self, data: dict) -> List[PositiveExample]:
        question = data['q']
        action_pred = lp.get_action_pred(question)

        # Stop if we can't parse the question
        if action_pred is None:
            logging.warning(
                F'{data["qid"]}: Cannot parse question {question}')
            return []

        action_subj = action_pred.subj
        pos_eg_list = []

        qid = data['qid']
        vid_name = data['vid_name']

        bbox_list = data['bbox']
        for id in bbox_list:
            frame = bbox_list[id]

            boxes = []
            obj_set = set()
            ppl_set = set()

            id = int(id)
            for box in frame:
                label = box['label'].lower()

                if label in BBT_PEOPLE:
                    ppl_set.add(label)
                else:
                    obj_set.add(label)

                box = BoundingBox(id, box['left'], box['top'], box['width'],
                                  box['height'], label)
                boxes.append(box)

            action_obj = self.get_goal_action_obj(data, obj_set)
            intersections = get_all_intersections(boxes)

            # Cannot generate positive example if we can't get goal action
            # predicate, or there is no intersection
            if action_obj is None or len(intersections) == 0:
                logging.warning(
                    F'{data["qid"]}: Cannot get positive example at frame {id}')
                continue

            action_pred.obj = action_obj
            inclusion = [F'goal(holdsAt({action_pred.gen_pred()}, {id + 1})).']
            exclusion = []
            for p in ppl_set:
                if p.lower() != action_subj.lower():
                    exclude_action = ActionPred(action_pred.action,
                                                p,
                                                action_obj)
                    exclusion += [
                        F'goal(holdsAt({exclude_action.gen_pred()}, {id + 1})).'
                    ]
            context = [F'current_time({id}).']
            context += list(map(lambda p: F'person({p}).', ppl_set))
            context += list(map(lambda o: F'object({o}).', obj_set))
            context += list(map(lambda b: b.gen_pred() + '.', boxes))
            context += list(map(lambda i: i.gen_pred() + '.', intersections))

            pos_eg_list.append(
                PositiveExample(qid, vid_name, id, context, inclusion,
                                exclusion))

        if len(pos_eg_list) == 0:
            logging.warning(F'{data["qid"]}: No example from this question')

        return pos_eg_list
