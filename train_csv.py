import argparse
from itertools import zip_longest

from train import train
import csv
import os

def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data and model checkpoints directories
    parser.add_argument('--data_dir', type=str, default='data/wine/input.csv',
                        help='path to training csv')

    return parser


if __name__ == '__main__':
    args = arg_parser().parse_args()

    with open(args.data_dir, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        columns = zip_longest(*csv_reader)

        for i, column in enumerate(columns):
            if i not in [2]:
                continue

            data_dir = "data/wip" + str(i)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            with open(data_dir + '/input.txt', "w+", encoding='utf-8') as text_file:
                text_file.write(u'\n'.join(column))

            train(save_dir='models/model_' + str(i), data_dir=data_dir)
            # oc = column
