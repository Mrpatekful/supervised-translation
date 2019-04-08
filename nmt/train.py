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

from model import (
    create_model, 
    greedy_decode, 
    create_criterion,
    create_optimizer,
    setup_model_args)

from beam import beam_search_decode


def setup_train_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of epochs for training.')
    parser.add_argument(
        '--cuda',
        type=str,
        default='cpu',
        help='Device for training.')

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def train(model, datasets, args):
    """"""
    device = torch.device(  # pylint: disable=no-member
        'cuda' if args.cuda else 'cpu') 
    model.to(device)

    train, val, test = datasets
    criterion = create_criterion(args)
    optimizer = create_optimizer(args, model.parameters())

    for epoch_idx in range(args.epochs):

        # Training
        with tqdm(total=len(train)) as pbar:
            pbar.set_description('epoch {}'.format(epoch_idx))
            model.train()

            for batch in train:
                optimizer.zero_grad()
                outputs = greedy_decode(model, criterion, batch)
                
                optimizer.step()
                pbar.set_postfix()
                pbar.update()

        # Validation
        with tqdm(total=len(train)) as pbar:
            pbar.set_description('epoch {}'.format(epoch_idx))
            model.eval()

            for batch in val:
                outputs = beam_search_decode(model, batch)

                pbar.set_postfix()
                pbar.update()

    # Testing
    model.eval()
    for batch in test:
        _ = beam_search_decode(model, batch)
            

def main():
    args = setup_train_args()

    datasets = create_datasets(args)
    model = create_model(args)

    train(model, datasets, args)


if __name__ == '__main__':
    main()
