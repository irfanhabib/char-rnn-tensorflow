import argparse
from itertools import zip_longest

from train import train
import csv


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data and model checkpoints directories
    parser.add_argument('--training_csv', type=str, default='data/csv/input.csv',
                        help='path to training csv')

    return parser


if __name__ == '__main__':
    args = arg_parser().parse_args()

    with open(args.training_csv, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        columns = zip_longest(*csv_reader)

        for i, column in enumerate(columns):
            train(save_dir='models/model_' + str(i))
