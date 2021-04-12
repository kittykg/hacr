import json
import time
import spacy

from common import COCO_INSTANCE_CATEGORY_NAMES, BBT_PEOPLE
import language_processing as lp

if __name__ == '__main__':
    # Some parameters
    frame_folder = '../TVQA_frames/frames_hq/bbt_frames/'
    hold_question_json_path = './hold_questions.json'
    npz_file_path = './face_collection.npz'

    write_to_file = True
    json_ver = 4  # Used for labelling the output json file if write to file

    start_time = time.time()

    nlp = spacy.load('en_core_web_sm')

    # Step 1: First filter
    # Check if the question ROOT word is 'hold' and the
    # question starts with 'what'
    # This step is commented. In hold_questions.json, the questions are
    # corrected

    # train_json_file_path = '../tvqa_plus_train_prettified.json'
    # with open(train_json_file_path) as f:
    #     all_data = json.load(f)
    # valid_data = []
    # for d in all_data:
    #     question = d['q']
    #     if lp.valid_hold_questions(question):
    #         valid_data.append(d)

    with open(hold_question_json_path) as f:
        valid_data = json.load(f)

    # Step 2: Get all objects directly/indirectly in COCO dataset
    # Check the object is in COCO or its synonym/hypernym
    all_objects = set()
    for v_q in valid_data:
        bboxes = v_q['bbox']

        for f_id in bboxes:
            frames = bboxes[f_id]
            for f in frames:
                label = f['label']
                if label.lower() not in BBT_PEOPLE:
                    doc = nlp(label)
                    if doc[0].pos_ == 'PROPN':
                        continue
                    all_objects.add(label)

    might_objects = set()
    might_not = set()
    for a_o in all_objects:
        o = nlp(a_o)[0].lemma_
        might = False
        for coco in COCO_INSTANCE_CATEGORY_NAMES:
            if lp.check_synonyms(o, coco):
                might_objects.add(a_o)
                might = True
                break
            elif lp.check_hyponyms(o, coco):
                might_objects.add(a_o)
                might = True
                break
            elif lp.check_hypernyms(o, coco):
                might_objects.add(a_o)
                might = True
                break
        if not might:
            might_not.add(a_o)

    # Step 3: Second filter
    # Filter the questions where the ground truth bbox label can be detected
    # i.e. in the might_objects set
    valid_data_v2 = []
    for v_q in valid_data:
        bboxes = v_q['bbox']
        answer_index = v_q['answer_idx']

        possible = False
        for f_id in bboxes:
            frames = bboxes[f_id]
            for f in frames:
                label = f['label']
                if label.lower() not in BBT_PEOPLE:
                    if label in might_objects:
                        possible = True

        if possible:
            valid_data_v2.append(v_q)

    if write_to_file:
        with open(F'hold_questions_v{json_ver}.json', 'w') as outfile:
            json.dump(valid_data_v2, outfile, indent=4, sort_keys=True)

    print(F'Running time:   {time.time() - start_time}')
