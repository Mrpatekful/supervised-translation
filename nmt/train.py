"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=import-error
# pylint: disable=no-name-in-module

import argparse
import torch

from tqdm import tqdm
from data import create_datasets, setup_data_args
from model import create_model, setup_model_args


def setup_train_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-me',
        '--max_epochs',
        type=int,
        default=100,
        help='Maximum number of epochs for training.')

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def train(model, datasets, args):
    """"""
    train, val, test = datasets

    for epoch_idx in range(args.max_epochs):
        for batch_idx, batch in enumerate(tqdm(train)):
            pass

        for batch_idx, batch in enumerate(tqdm(val)):
            pass

    for batch_idx, batch in enumerate(test):
        print(batch_idx)


def main():
    args = setup_train_args()

    datasets = create_datasets(args)
    model = create_model(args)

    train(model, datasets, args)


if __name__ == '__main__':
    main()
