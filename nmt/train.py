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
import random

from torch.optim import Adam

from tqdm import tqdm
from data import create_datasets, setup_data_args

from model import (
    create_model, 
    create_criterion,
    setup_model_args)

from beam import decode_beam_search


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
        type=bool,
        default=False,
        help='Device for training.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate for the model.')


    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def create_optimizer(args, parameters):
    """"""
    return Adam(lr=args.learning_rate, params=parameters)


def compute_loss(inputs, targets, criterion):
    pass


def train_step(model, criterion, optimizer, batch):
    """"""
    optimizer.zero_grad()

    inputs, input_lengths = batch.src
    targets, target_lengths = batch.trg

    outputs = model(inputs=inputs, targets=targets)
    # Applying teacher forcing 50% of the time.
    # if random.random() > 0.5:
    #     outputs = model(inputs=inputs)
    
    # else:
    #     outputs = model(inputs=inputs, targets=targets)

    # loss = compute_loss(outputs)
    # loss.backward()

    optimizer.step()


def eval_step(model, criterion, batch, use_beam_search=False):
    """"""
    inputs, input_lengths = batch.src
    targets, target_lengths = batch.trg

    if use_beam_search:
        preds, scores = decode_beam_search(model=model, inputs=inputs)

    else:
        preds, scores = model(inputs=inputs)


def log_results(results):
    """"""


def train(model, datasets, args):
    """"""
    device = torch.device(  # pylint: disable=no-member
        'cuda' if args.cuda else 'cpu') 
    model.to(device)

    train, val, test = datasets
    criterion = create_criterion(args)
    optimizer = create_optimizer(args, model.parameters())

    for epoch_idx in range(args.epochs):

        with tqdm(total=len(train)) as pbar:
            pbar.set_description('epoch {}'.format(epoch_idx))
            model.train()

            for batch in train:
                results = train_step(
                    model=model, 
                    criterion=criterion, 
                    optimizer=optimizer, 
                    batch=batch)

                log_results(results=results)

                pbar.set_postfix()
                pbar.update()

        with tqdm(total=len(val)) as pbar:
            pbar.set_description('epoch {}'.format(epoch_idx))
            model.eval()

            with torch.no_grad():
                for batch in val:
                    results = eval_step(
                        model=model,
                        criterion=criterion,
                        batch=batch)

                    log_results(results=results)

                    pbar.set_postfix()
                    pbar.update()

    with torch.no_grad():
        model.eval()
        for batch in test:
            _ = beam_search_decode(model, batch)
            

def main():
    args = setup_train_args()

    datasets, vocab_sizes = create_datasets(args)
    model = create_model(args, vocab_sizes)

    train(model, datasets, args)


if __name__ == '__main__':
    main()
