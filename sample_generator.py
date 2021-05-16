import json
import time

from json_parser import Parser
from utils import split_data_set

if __name__ == '__main__':
    start_time = time.time()
    parser = Parser()

    with open('questions/train_hold.json') as f:
        all_data = json.load(f)

    train_set, test_set = split_data_set(0.8, all_data)

    total_examples = 0
    with open('pos_eg_v1', 'w') as out:
        for t in train_set:
            examples = parser.get_pos_example_h(t)
            total_examples += len(examples)
            for e in examples:
                print(e.gen_example(), file=out)

    print(F'Running time:   {time.time() - start_time}')
    print(F'Total examples: {total_examples}')
