import logging
from typing import List, Union, Set

from common import PositiveExample, BoundingBox, ActionPred, BBT_PEOPLE
import language_processing as lp
from utils import get_all_intersections, time_span_start_end


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
    @staticmethod
    def match_ans_obj_with_bbox_obj(ans_obj: str,
                                    bbox_obj_set: Set[str]) -> Union[str, None]:
        for bbox_obj in bbox_obj_set:
            noun_tag = 'NOUN'
            if lp.check_synonyms(ans_obj, bbox_obj, noun_tag) or \
                    lp.check_hyponyms(ans_obj, bbox_obj, noun_tag) or \
                    lp.check_hypernyms(ans_obj, bbox_obj, noun_tag):
                return bbox_obj
        return None

    @staticmethod
    def get_goal_action_obj(data: dict, bbox_obj_set: Set[str],
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

        obj = Parser.match_ans_obj_with_bbox_obj(ans_obj, bbox_obj_set)

        # Stop if we can't match the object in the correct answer with
        # an object in our grounding box
        if obj is None:
            if log:
                logging.warning(
                    F'Cannot match {ans_obj} with set {bbox_obj_set}')

        return obj

    @staticmethod
    def get_pos_example_h(data: dict, log: bool = False) -> \
            List[PositiveExample]:
        """
        HACR-H get positive example
        :param data: dict, json data
        :param log: bool, whether or not to log errors etc.
        :return: list of PositiveExamples
        """
        question = data['q']
        action_pred = lp.get_action_pred(question)

        # Stop if we can't parse the question
        if action_pred is None:
            if log:
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

            action_obj = Parser.get_goal_action_obj(data, obj_set)
            intersections = get_all_intersections(boxes)

            # Cannot generate positive example if we can't get goal action
            # predicate, or there is no intersection
            if action_obj is None or len(intersections) == 0:
                if log:
                    logging.warning(
                        F'{data["qid"]}: Cannot get positive example at frame {id}')
                continue

            action_pred.obj = action_obj
            exclusion = []
            for p in ppl_set:
                if p.lower() != action_subj.lower():
                    exclude_action = ActionPred(action_pred.action,
                                                p,
                                                action_obj)
                    exclusion += [
                        F'holdsAt({exclude_action.gen_pred()}, {id + 1})'
                    ]
            inclusion = [F'holdsAt({action_pred.gen_pred()}, {id + 1})']
            context = [F'goal(holdsAt({action_pred.gen_pred()}, {id + 1})).']
            context += [F'current_time({id}).']
            context += list(map(lambda p: F'person({p}).', ppl_set))
            context += list(map(lambda o: F'object({o}).', obj_set))
            context += list(map(lambda b: b.gen_pred() + '.', boxes))
            context += list(map(lambda i: i.gen_pred() + '.', intersections))

            pos_eg_list.append(
                PositiveExample(qid, vid_name, id, context, inclusion,
                                exclusion))

        if len(pos_eg_list) == 0:
            if log:
                logging.warning(F'{data["qid"]}: No example from this question')

        return pos_eg_list

    @staticmethod
    def get_pos_example_e(data: dict, log: bool = False) \
            -> Union[None, PositiveExample]:
        """
        HACR-E get positive example
        :param data: dict, json data
        :param log: bool, whether or not to log errors etc.
        :return: None if there's error, otherwise return PositiveExample
        """
        qid = data['qid']
        vid_name = data['vid_name']

        ans_idx = data['answer_idx']
        answers = lp.get_all_people_in_ans(data[F'a{ans_idx}'])
        if not answers:
            if log:
                logging.warning(F'{qid}: Cannot get names from answer')
            return None

        inclusion_list = []
        gt_list = []
        exclusion_list = []
        s_t, e_t = time_span_start_end(data)

        for answer in answers:
            person = answer.lower()
            enter_time = data['in_scene'][person][0][0]
            inclusion_list.append(
                F'initiates(enter({person}), at_curr_location({person}), '
                F'{enter_time})')

        gt_list.append(F'time({s_t}..{e_t}).')

        for pair in data['scene_change_pairs']:
            gt_list.append(F'abrupt_transition({pair[0]}, {pair[1]}).')
        for person in data['in_scene']:
            gt_list.append(F'person({person}).')
            for pair in data['in_scene'][person]:
                gt_list.append(
                    F'holdsAt(in_scene({person}), {pair[0]}..{pair[1]}).')
        for person in data['initial_in_scene']:
            gt_list.append(F'holdsAt(at_curr_location({person}), {s_t}).')
            exclusion_list.append(
                F'initiates(enter({person}), at_curr_location({person}), {s_t}..{e_t})')

        return PositiveExample(qid=qid,
                               vid_name=vid_name,
                               # don't need curr_time for enter learning
                               curr_time=0,
                               context=gt_list,
                               inclusions=inclusion_list,
                               exclusions=exclusion_list)
