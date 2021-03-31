from dataclasses import dataclass
import itertools
import json
import logging
from typing import List, Union, Set

import language_processing as lp
from common import PositiveExample, GroundingBox, ActionPred

PEOPLE_NAMES = {'penny', 'sheldon', 'leonard', 'howard', 'raj', 'amy'}


@dataclass
class BBoxIntersectionPred:
    frame_id: int
    b1_label: str
    b2_label: str
    frac_wrt_b1: int

    def gen_pred(self):
        return F'bbox_intersec({self.frame_id}, {self.b1_label}, {self.b2_label}, {self.frac_wrt_b1})'


class Parser:
    def __init__(self):
        self.relational_object_pairs = {
            'piece': 'clothing',
            'remote': 'tv'
        }

    def match_ans_obj_with_bbox_obj(self, ans_obj: str, bbox_obj_set: Set[str]) -> Union[str, None]:
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
        logging.warning(F'Cannot match {ans_obj} with set {bbox_obj_set}')
        return None

    @staticmethod
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

    def get_goal_action_pred(self, data: dict, bbox_obj_set: Set[str]) -> Union[ActionPred, None]:
        question = data['q']
        action_pred = lp.get_action_pred(question)

        # Stop if we can't parse the question
        if action_pred is None:
            return None

        answer_idx = data['answer_idx']
        ans_text = data['a' + answer_idx]
        ans_obj = Parser.get_answer_obj(ans_text)

        # Stop if we can't parse the correct answer
        if ans_obj is None:
            return None

        obj = self.match_ans_obj_with_bbox_obj(ans_obj, bbox_obj_set)

        # Stop if we can't match the object in the correct answer with an object in our grounding box
        if obj is None:
            return None

        action_pred.obj = obj
        return action_pred

    def get_pos_example(self, data: dict) -> List[PositiveExample]:
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

                if label in PEOPLE_NAMES:
                    ppl_set.add(label)
                else:
                    obj_set.add(label)

                box = GroundingBox(id, box['top'], box['left'], box['width'], box['height'], label)
                boxes.append(box)

            if obj_set == set():
                continue

            action_pred = self.get_goal_action_pred(data, obj_set)
            intersections = Parser.get_all_intersections(boxes)

            # Cannot generate positive example if we can't get goal action predicate, or
            # there is no intersection
            if action_pred is None or len(intersections) == 0:
                continue

            facts = [F'goal(holdsAt({action_pred.gen_pred()}, {id + 1})).']
            facts += [F'current_time({id}).']
            facts += list(map(lambda p: F'person({p}).', ppl_set))
            facts += list(map(lambda o: F'object({o}).', obj_set))
            facts += list(map(lambda b: b.gen_pred() + '.', boxes))
            facts += list(map(lambda i: i.gen_pred() + '.', intersections))

            pos_eg_list.append(PositiveExample(qid, vid_name, id, facts))

        return pos_eg_list

    @staticmethod
    def get_all_intersections(boxes: List[GroundingBox]) -> List[BBoxIntersectionPred]:
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
                intersections.append(BBoxIntersectionPred(b1.img_id, b1.label, b2.label, frac_wrt_b1))

            if frac_wrt_b2 > 0:
                intersections.append(BBoxIntersectionPred(b2.img_id, b2.label, b1.label, frac_wrt_b2))

        return intersections
