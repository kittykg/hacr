import json
import random

from json_parser import Parser


def split_data_set(proportion: float, data: list):
    random.shuffle(data)
    train_size = int(len(data) * proportion)
    return data[:train_size], data[train_size:]


if __name__ == '__main__':
    parser = Parser()

    with open('train_hold.json') as f:
        all_data = json.load(f)

    train_set, test_set = split_data_set(0.8, all_data)

    with open('pos_eg_v1', 'w') as out:
        for t in train_set:
            for e in parser.get_pos_example(t):
                print(e.gen_example(), file=out)
