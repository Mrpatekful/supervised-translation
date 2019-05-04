"""

@author:    Patrik Purgai
@copyright: Copyright 2019, nmt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=import-error
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=not-callable

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

import argparse
import os
import random

random.seed(0)

import numpy as np

np.random.seed(0)

from os.path import exists, join, dirname, abspath
from torch.optim import Adam

from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
from nltk.translate import bleu

from data import (
    create_datasets, 
    setup_data_args, 
    get_special_indices,
    ids2text)

from model import (
    create_model, 
    create_criterion,
    setup_model_args)

from beam import (
    beam_search, 
    setup_beam_args)


PROJECT_DIR = join(abspath(dirname(__file__)), '..')


def setup_train_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
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
        default=join(PROJECT_DIR, 'model.{}'.format(
            datetime.today().strftime('%j%H%m'))),
        help='Path of the model checkpoints.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Size of the batches during training.')

    setup_data_args(parser)
    setup_model_args(parser)
    setup_beam_args(parser)

    return parser.parse_args()


def save_state(model, optimizer, avg_loss, path):
    """
    Saves the model and optimizer state.
    """
    model_path = join(path, 'model.pt')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'avg_loss': avg_loss
    }
    torch.save(state, model_path)
    print('Saving model to {}'.format(model_path))


def load_state(model, optimizer, path, device):
    """
    Loads the model and optimizer state.
    """
    try:
        model_path = join(path, 'model.pt')
        state_dict = torch.load(
            model_path, map_location=device)

        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        print('Loading model from {}'.format(model_path))
        return state_dict['avg_loss']

    except FileNotFoundError:
        return np.inf


def create_optimizer(args, parameters):
    """
    Creates an ADAM optimizer.
    """
    optimizer = Adam(lr=args.learning_rate, params=parameters)
    return optimizer


def compute_loss(outputs, targets, criterion, pad_index):
    """
    Computes the loss and accuracy with masking.
    """
    scores, preds = outputs
    scores_view = scores.view(-1, scores.size(-1))
    targets_view = targets.view(-1)

    loss = criterion(scores_view, targets_view)
    notpad = targets.ne(pad_index)
    target_tokens = notpad.long().sum().item()
    correct = ((targets == preds) * notpad).sum().item()

    accuracy = correct / target_tokens
    loss = loss / target_tokens

    return loss, accuracy


def train_step(model, criterion, optimizer, batch, indices):
    """
    Performs a single step of training.
    """
    optimizer.zero_grad()

    _, _, _, trg_pad_index, _ = indices
    inputs, targets = batch.src, batch.trg

    # targets[:, 0] is the sos token
    targets = targets[:, 1:].contiguous()
    max_len = targets.size(1)
    
    outputs = model(inputs=inputs, targets=targets, 
                    max_len=max_len)
    
    loss, accuracy = compute_loss(
        outputs=outputs, 
        targets=targets, 
        criterion=criterion,
        pad_index=trg_pad_index)

    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy(), accuracy


def eval_step(model, criterion, batch, indices, device, beam_size):
    """
    Performs a single step of evaluation.
    """
    inputs, targets = batch.src, batch.trg
    _, _, _, trg_pad_index, _ = indices

    targets = targets[:, 1:].contiguous()
    batch_size, max_len = targets.size()
    
    if beam_size > 1:
        outputs = beam_search(
            model=model, 
            inputs=inputs,
            indices=indices,
            beam_size=beam_size,
            device=device,
            max_len=max_len)
    else:
        outputs = model(inputs=inputs, max_len=max_len)

    scores, preds = outputs
    
    # handling the case when greedy or beam search decoding
    # terminates before target sequence length
    if scores.size(1) < max_len:
        size_diff = max_len - scores.size(1)
        zeros = torch.zeros(
            [batch_size, size_diff, scores.size(2)]).to(device)
        scores = torch.cat([scores, zeros], 1)
        paddings = torch.tensor(trg_pad_index).expand(
            batch_size, size_diff).to(device)
        preds = torch.cat([preds, paddings], 1)

        outputs = scores, preds

    loss, accuracy = compute_loss(
        outputs=outputs, 
        targets=targets, 
        criterion=criterion,
        pad_index=trg_pad_index)

    return loss.cpu().numpy(), accuracy, preds


def train(model, datasets, indices, args, device):
    """
    Performs training, validation and testing.
    """
    _, _, _, trg_pad_index, _ = indices

    train, valid, test = datasets
    criterion = create_criterion(args, trg_pad_index)
    optimizer = create_optimizer(args, model.parameters())

    prev_avg_loss = load_state(
        model, optimizer, args.model_dir, device)

    for epoch in range(args.epochs):

        # running training loop
        loop = tqdm(train)
        loop.set_description('epoch {}'.format(epoch))
        model.train()

        for batch in loop:
            loss, accuracy = train_step(
                model=model, 
                criterion=criterion, 
                optimizer=optimizer,
                indices=indices, 
                batch=batch)

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=accuracy))

        # running validation loop
        loop = tqdm(valid)
        loop.set_description('validation')
        model.eval()
        avg_loss = []

        with torch.no_grad():
            for batch in loop:
                loss, accuracy, _ = eval_step(
                    model=model,
                    criterion=criterion,
                    indices=indices,
                    device=device,
                    beam_size=1,
                    batch=batch)

                avg_loss.append(loss)
                loop.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy))

        avg_loss = sum(avg_loss) / len(avg_loss)
        tqdm.write(
            'avg validation loss: {:.4}'.format(avg_loss))
        if avg_loss < prev_avg_loss:
            save_state(
                model, optimizer, avg_loss, args.model_dir)
            prev_avg_loss = avg_loss

    # running testing loop
    loop = tqdm(test)
    loop.set_description('testing')
    model.eval()

    with torch.no_grad():
        for batch in loop:
            loss, accuracy, _ = eval_step(
                model=model,
                criterion=criterion,
                indices=indices,
                device=device,
                beam_size=args.beam_size,
                batch=batch)
            
            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=accuracy))


def main():
    args = setup_train_args()
    device = torch.device('cuda' if args.cuda else 'cpu') 

    datasets, vocabs = create_datasets(args, device)
    indices = get_special_indices(vocabs)
    model = create_model(args, vocabs, indices, device)

    train(model, datasets, indices, args, device)


if __name__ == '__main__':
    main()
