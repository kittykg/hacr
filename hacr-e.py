import json
import time
import subprocess

from sklearn.model_selection import KFold

import common
import inference as inf
from json_parser import Parser

# Background knowledge of event calculus etc. for enter learning
with open(common.ENTER_ASP_BASE) as f:
    background_knowledge = f.read().splitlines()


def inference_test_enter(train_set, test_set, gt_human_face=False,
                         gt_abt=False):
    ############################################################################
    # 1. Generate examples based on train_set                                  #
    ############################################################################

    with open('pos_eg_enter', 'w') as out:
        for t in train_set:
            example = Parser.get_pos_example_e(t)
            if example:
                print(example.gen_example(), file=out)

    ############################################################################
    # 2. Learn with ILASP and write new rules to base.lp                       #
    ############################################################################

    # Since one example per question, there are len(train_set) of examples
    print(f'Running ILASP on {len(train_set)} training examples...')
    cp = subprocess.run(['./ILASP', '--version=4', '-ml=4',
                         common.ENTER_ILASP_BK, 'pos_eg_enter'],
                        capture_output=True)
    learnt_rule = cp.stdout.decode('utf-8').split('\n')[0]
    print(F'Learnt rule: {learnt_rule}')

    with open('base_enter.lp', 'w') as f:
        for line in background_knowledge:
            print(line, file=f)
        print(learnt_rule, file=f)

    ############################################################################
    # 3. Test on test set                                                      #
    ############################################################################

    jacc_score = 0
    num_questions = 0
    not_full_score = []
    zero_score = []

    for test in test_set:
        pred_ans_idx = inf.inference_e(test, gt_human_face=gt_human_face,
                                       gt_abt=gt_abt, log=True)
        score = inf.get_jaccard_score(pred_ans_idx, test)

        if 0 < score < 1:
            not_full_score.append(test['qid'])
        elif score == 0:
            zero_score.append(test['qid'])

        jacc_score += score
        num_questions += 1

    ############################################################################
    # 4. Print out result                                                      #
    ############################################################################

    print(F'# tests:     {num_questions}')
    print(F'jacc score:   {jacc_score}')
    print(F'norm jacc score: {jacc_score / num_questions}')
    print()

    print('NOT FULL SCORE')
    print(not_full_score)
    print()

    print('ZERO SCORE')
    print(zero_score)

    return jacc_score, num_questions


def kfold_enter_test(gt_human_face: bool = False, gt_abt: bool = False):
    print(F'gt_human_face: {gt_human_face}')
    print(F'gt_abt:        {gt_abt}')
    print()

    with open(common.ENTER_QUESTIONS) as f:
        s_e_data = json.load(f)

    run_id = 0
    total_jacc_score = 0
    total_questions = 0
    start_time = time.time()

    kf = KFold(n_splits=5)
    kf.get_n_splits(s_e_data)

    for train_index, test_index in kf.split(s_e_data):
        print(F'-------Run {run_id}-------')
        train_set = [s_e_data[i] for i in train_index]
        test_set = [s_e_data[i] for i in test_index]

        jacc, n_q = inference_test_enter(train_set, test_set,
                                         gt_human_face=gt_human_face,
                                         gt_abt=gt_abt)
        total_jacc_score += jacc
        total_questions += n_q
        run_id += 1

        print(F'----------------------------')
        print()

    print(F'Total runtime: {time.time() - start_time}')
    print(F'Avg norm jacc score across folds: '
          F'{total_jacc_score / total_questions}')


if __name__ == '__main__':
    kfold_enter_test()
