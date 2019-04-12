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
import os

import numpy as np

from os.path import exists, join, dirname, abspath
from datetime import datetime
from torch.optim import Adam
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
from data import create_datasets, setup_data_args

from model import (
    create_model, 
    create_criterion,
    setup_model_args)

from beam import decode_beam_search, setup_beam_args


project_dir = join(abspath(dirname(__file__)), '..')


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
    parser.add_argument(
        '--model_dir',
        type=str,
        default=join(project_dir, 'model.{}'.format(
            datetime.today().strftime('%j%H%m'))),
        help='Path of the model checkpoints.')
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Number of epochs without progress.')

    setup_data_args(parser)
    setup_model_args(parser)
    setup_beam_args(parser)

    return parser.parse_args()


def save_state(epoch, model, optimizer, loss, patience, path):
    """"""
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'patience': patience
    }
    torch.save(state, join(path, 'state-{:.4}-{}.pt'.format(
        loss, epoch)))


def load_last_state(path, model, optimizer, patience):
    """"""
    try:
        file_path = sorted((f for f in os.listdir(path)), 
            key=lambda x: float(x.split('-')[-2]))[-1]

        state = torch.load(join(path, file_path))
        epoch = state['epoch']
        loss = state['loss']
        patience = state['patience']

        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

        return epoch, loss, patience

    except IndexError:
        return 0, np.inf, patience


def create_optimizer(args, parameters):
    """"""
    optimizer = Adam(lr=args.learning_rate, params=parameters)
    return optimizer


def compute_bleu(outputs, targets):
    """"""
    # TODO
    return 0


def compute_loss(outputs, targets, criterion, pad_index):
    """"""
    scores, preds = outputs

    scores_view = scores.view(-1, scores.size(-1))
    targets_view = targets.view(-1)
    loss = criterion(scores_view, targets_view)
    
    notnull = targets.ne(pad_index)
    target_tokens = notnull.long().sum().item()
    correct = ((targets == preds) * notnull).sum().item()

    accuracy = correct / target_tokens
    loss = loss / target_tokens
    bleu = compute_bleu(outputs, targets)

    return loss, accuracy, bleu


def train_step(model, criterion, optimizer, batch, trg_pad_index):
    """"""
    optimizer.zero_grad()

    inputs, targets = batch.src, batch.trg
    max_len=targets.size(1)

    # Applying teacher forcing 50% of the time.
    if random.random() > 0.5:
        outputs = model(inputs=inputs, max_len=max_len)
    else:
        outputs = model(inputs=inputs, targets=targets, max_len=max_len)

    loss, accuracy, bleu = compute_loss(
        outputs=outputs, 
        targets=targets, 
        criterion=criterion,
        pad_index=trg_pad_index)

    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy(), accuracy, bleu


def eval_step(model, criterion, batch, src_pad_index, 
              trg_pad_index, use_beam_search=False):
    """"""
    inputs, targets = batch.src, batch.trg

    if use_beam_search:
        outputs = decode_beam_search(
            model=model, inputs=inputs, max_len=targets.size(1))
    else:
        outputs = model(inputs=inputs, max_len=targets.size(1))

    scores, preds = outputs
    batch_size, sequence_length = targets.size()

    # Handling the case when greedy or beam search decoding
    # terminates before target sequence length.
    if scores.size(1) < sequence_length:
        size_diff = sequence_length - scores.size(1)
        scores = torch.cat([scores, torch.zeros( # pylint: disable=no-member
            [batch_size, size_diff, scores.size(2)])], dim=1) 
        preds = torch.cat([preds, src_pad_index.expand( # pylint: disable=no-member
            batch_size, size_diff
        )], dim=1)

        outputs = scores, preds

    loss, accuracy, bleu = compute_loss(
        outputs=outputs, 
        targets=targets, 
        criterion=criterion,
        pad_index=trg_pad_index)

    return loss.cpu().numpy(), accuracy, bleu


def train(model, datasets, pad_indices, args, device):
    """"""
    src_pad_index, trg_pad_index = pad_indices
    src_pad_index = torch.tensor(src_pad_index) # pylint: disable=not-callable
    src_pad_index.to(device)

    train, valid, test = datasets
    criterion = create_criterion(args, trg_pad_index)
    optimizer = create_optimizer(args, model.parameters())

    initial_epoch, best_loss, patience = load_last_state(
        model=model, 
        optimizer=optimizer, 
        path=args.model_dir,
        patience=args.patience)

    for epoch in range(initial_epoch, args.epochs):

        # Running training loop.
        with tqdm(total=len(train)) as pbar:
            pbar.set_description('epoch {}'.format(epoch))
            model.train()

            for batch in train:
                loss, accuracy, bleu = train_step(
                    model=model, 
                    criterion=criterion, 
                    optimizer=optimizer,
                    trg_pad_index=trg_pad_index, 
                    batch=batch)

                pbar.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy, bleu=bleu))
                pbar.update()

        # Running validation loop.
        with tqdm(total=len(valid)) as pbar:
            pbar.set_description('epoch {}'.format(epoch))
            model.eval()
            avg_loss = 0

            with torch.no_grad():
                for batch in valid:
                    loss, accuracy, bleu = eval_step(
                        model=model,
                        criterion=criterion,
                        src_pad_index=src_pad_index,
                        trg_pad_index=trg_pad_index,
                        batch=batch)

                    avg_loss += loss
                    pbar.set_postfix(ordered_dict=OrderedDict(
                        loss=loss, acc=accuracy, bleu=bleu))
                    pbar.update()

        # Perform early stopping after `patience` number
        # of epochs without progress.
        avg_loss /= len(valid)
        if best_loss > avg_loss:
            best_loss = avg_loss
            save_state(
                epoch=epoch, model=model, 
                optimizer=optimizer, loss=best_loss, 
                patience=patience, path=args.model_dir)
            patience = args.patience
        else:
            patience -= 1

        if patience == 0:
            break

    # Running testing loop.
    with tqdm(total=len(test)) as pbar:
        with torch.no_grad():
            model.eval()
            for batch in test:
                loss, accuracy, bleu = eval_step(
                    model=model,
                    criterion=criterion,
                    src_pad_index=src_pad_index,
                    trg_pad_index=trg_pad_index,
                    use_beam_search=True,
                    batch=batch)
                
                pbar.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy, bleu=bleu))
                pbar.update()
            

def main():
    args = setup_train_args()
    device = torch.device( # pylint: disable=no-member
        'cuda' if args.cuda else 'cpu') 

    datasets, vocab_sizes, indices = create_datasets(args, device)
    pad_indices, start_index, end_index = indices
    
    model = create_model(
        args, vocab_sizes, (start_index, end_index), device)

    train(model, datasets, pad_indices, args, device)


if __name__ == '__main__':
    main()
