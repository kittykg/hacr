import json
import multiprocessing as mp
import time

import numpy as np

import abrupt_transition_detection as atd
from common import TVQA_PLUS_TRAIN_JSON

start_time = time.time()

npz_file_path = './transition_collection_hist_v2.npz'

pool = mp.Pool(mp.cpu_count())

with open(TVQA_PLUS_TRAIN_JSON) as f:
    all_json_data = json.load(f)

all_hist_list = [
    pool.apply(
        atd.gen_pixel_diff,
        args=(all_json_data[i], False, 'hist-all')
    )['pixel_diff_score']
    for i in range(10000)
]

all_hist_all_diff_score = []
for l in all_hist_list:
    all_hist_all_diff_score += l

all_hist_list = [
    pool.apply(
        atd.gen_pixel_diff,
        args=(all_json_data[i], False, 'hist-reg')
    )['pixel_diff_score']
    for i in range(10000)
]

all_hist_reg_diff_score = []
for l in all_hist_list:
    all_hist_reg_diff_score += l

# all_hist_all_diff_score = []
# all_hist_reg_diff_score = []
# for i in range(15):
#     data = all_json_data[i]
#     all_hist_all_diff_score += \
#         atd.gen_pixel_diff(data, score_method='hist-all')['pixel_diff_score']
#     all_hist_reg_diff_score += \
#         atd.gen_pixel_diff(data, score_method='hist-reg')['pixel_diff_score']

print(F'Time:                {time.time() - start_time}')
print(F'Len hist-all scores: {len(all_hist_all_diff_score)}')
print(F'Len hist-reg scores: {len(all_hist_reg_diff_score)}')

np.savez_compressed(npz_file_path,
                    hist_all_diff_scores=np.array(all_hist_all_diff_score)
                    ,
                    hist_reg_diff_scores=np.array(all_hist_reg_diff_score)
                    )
