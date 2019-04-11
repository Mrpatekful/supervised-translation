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
        default=torch.cuda.is_available(),
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


def compute_loss(outputs, targets, criterion, pad_token):
    """"""
    scores, preds = outputs

    scores_view = scores.view(-1, scores.size(-1))
    targets_view = targets.view(-1)
    loss = criterion(scores_view, targets_view)
    
    notnull = targets.ne(pad_token)
    target_tokens = notnull.long().sum().item()
    print(targets.size(), preds.size(), notnull.size())
    correct = ((targets == preds) * notnull).sum().item()

    return loss


def train_step(model, criterion, optimizer, batch, pad_token):
    """"""
    optimizer.zero_grad()

    inputs, targets = batch.src, batch.trg

    outputs = model(inputs=inputs, max_len=targets.size(1))
    # Applying teacher forcing 50% of the time.
    # if random.random() > 0.5:
    #     outputs = model(inputs=inputs)
    
    # else:
    #     outputs = model(inputs=inputs, targets=targets)

    loss = compute_loss(
        outputs=outputs, 
        targets=targets, 
        criterion=criterion,
        pad_token=pad_token)

    loss.backward()

    optimizer.step()

    return loss


def eval_step(model, criterion, batch, pad_token, 
              use_beam_search=False):
    """"""
    inputs, targets = batch.src, batch.trg

    if use_beam_search:
        outputs = decode_beam_search(model=model, inputs=inputs)

    else:
        outputs = model(inputs=inputs)

    loss = compute_loss(
        outputs=outputs, 
        targets=targets, 
        criterion=criterion,
        pad_token=pad_token)

    return loss


def train(model, datasets, pad_token, args):
    """"""
    device = torch.device(  # pylint: disable=no-member
        'cuda' if args.cuda else 'cpu') 
    model.to(device)

    train, valid, test = datasets
    criterion = create_criterion(args, pad_token)
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
                    pad_token=pad_token, 
                    batch=batch)

                pbar.set_postfix()
                pbar.update()

        with tqdm(total=len(valid)) as pbar:
            pbar.set_description('epoch {}'.format(epoch_idx))
            model.eval()

            with torch.no_grad():
                for batch in valid:
                    results = eval_step(
                        model=model,
                        criterion=criterion,
                        pad_token=pad_token,
                        batch=batch)

                    pbar.set_postfix()
                    pbar.update()

    with tqdm(total=len(test)) as pbar:
        with torch.no_grad():
            model.eval()
            for batch in test:
                results = eval_step(
                    model=model,
                    criterion=criterion,
                    pad_token=pad_token,
                    use_beam_search=True,
                    batch=batch)
                
                pbar.set_postfix()
                pbar.update()
            

def main():
    args = setup_train_args()

    datasets, vocab_sizes, tokens = create_datasets(args)
    pad_token, start_token, end_token = tokens
    
    model = create_model(args, vocab_sizes, (start_token, end_token))

    train(model, datasets, pad_token, args)


if __name__ == '__main__':
    main()
