import json
import subprocess
import time

from sklearn.model_selection import KFold

import common
import inference as inf
from json_parser import Parser

# Background knowledge of event calculus etc. for hold learning
with open(common.HOLD_ASP_BASE) as f:
    background_knowledge = f.read().splitlines()


def inference_test_hold(train_set, test_set, gt_od=False):
    ############################################################################
    # 1. Generate examples based on train_set                                  #
    ############################################################################

    total_examples = 0
    with open('pos_eg_od', 'w') as out:
        for t in train_set:
            examples = Parser.get_pos_example_h(t)
            total_examples += len(examples)
            for e in examples:
                print(e.gen_example(), file=out)

    ############################################################################
    # 2. Learn with ILASP and write new rules to base.lp                       #
    ############################################################################

    print(F'Running ILASP on {total_examples} training examples..')
    cp = subprocess.run(['./ILASP', '--version=4', '--override-default-sm',
                         common.HOLD_ILASP_BK, common.ILASP_OVR, 'pos_eg_od'],
                        capture_output=True)

    learnt_rule = cp.stdout.decode('utf-8').split('\n')[0]
    print(F'Learnt rule: {learnt_rule}')

    with open('base.lp', 'w') as f:
        for line in background_knowledge:
            print(line, file=f)
        print(learnt_rule, file=f)

    ############################################################################
    # 3. Test on test set                                                      #
    ############################################################################

    jacc_score = 0
    num_questions = 0
    parsing_error = []
    not_full_score = []
    zero_score = []

    for test in test_set:
        pred_ans_idx = inf.inference_h(test, gt_object=gt_od)
        if pred_ans_idx == [-1]:
            parsing_error.append(test['qid'])
            continue

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

    print('PARSING ERROR')
    print(parsing_error)
    print()

    print('NOT FULL SCORE')
    print(not_full_score)
    print()

    print('ZERO SCORE')
    print(zero_score)

    return jacc_score, num_questions


def kfold_hold_test(gt_od=False):
    with open(common.OD_HOLD_QUESTIONS) as f:
        attempted_question = json.load(f)
    print(F'# total tests: {len(attempted_question)}')

    kf = KFold(n_splits=5)
    kf.get_n_splits(attempted_question)

    start_time = time.time()

    run_id = 0
    total_jacc_score = 0
    total_questions = 0

    for train_index, test_index in kf.split(attempted_question):
        print(F'--------Run {run_id}--------')
        train_set = [attempted_question[i] for i in train_index]
        test_set = [attempted_question[i] for i in test_index]

        jacc, n_q = inference_test_hold(train_set, test_set, gt_od)
        total_jacc_score += jacc
        total_questions += n_q
        run_id += 1

        print(F'----------------------------')
        print()

    print(F'Total runtime: {time.time() - start_time}')
    print(F'Avg norm jacc score across folds: '
          F'{total_jacc_score / total_questions}')


if __name__ == '__main__':
    kfold_hold_test()
