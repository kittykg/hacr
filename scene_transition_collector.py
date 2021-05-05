import json
import time

import numpy as np

import abrupt_transition_detection as atd
from common import TVQA_PLUS_TRAIN_JSON

start_time = time.time()

npz_file_path = './transition_collection.npz'

with open(TVQA_PLUS_TRAIN_JSON) as f:
    all_json_data = json.load(f)

all_diff_score = []
for i in range(len(all_json_data)):
    data = all_json_data[i]
    all_diff_score += atd.gen_pixel_diff(data)['pixel_diff_score']

print(F'Time:            {time.time() - start_time}')
print(F'Len diff scores: {len(all_diff_score)}')

np.savez_compressed(npz_file_path,
                    pixel_diff_scores=np.array(all_diff_score))

