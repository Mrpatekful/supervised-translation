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

from beam import beam_search, setup_beam_args
from model import create_model, setup_model_args

from data import (
    create_datasets,
    setup_data_args,
    get_special_indices,
    ids2text)

from nltk.translate.bleu_score import sentence_bleu
from apex import amp
from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime

from torch.nn.functional import kl_div
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from os.path import exists, join, dirname, abspath

import torch
import argparse
import os
import random
import numpy as np


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True


def setup_train_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='Maximum number of epochs for training.')
    parser.add_argument(
        '--cuda',
        type=bool,
        default=torch.cuda.is_available(),
        help='Device for training.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1,
        help='Learning rate for the model.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=join(
            abspath(dirname(__file__)), '..', 'model.{}'.format(
                datetime.today().strftime('%j%H%m'))),
        help='Path of the model checkpoints.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Size of the batches during training.')
    parser.add_argument(
        '--mixed',
        type=bool,
        default=True,
        help='Use mixed precision training.')

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


def create_optimizer(args, parameters, finetune=False):
    """
    Creates an adam or swats optimizer with cyclical learning rate.
    """
    optimizer = SGD(lr=args.learning_rate, weight_decay=1e-6,
                    params=parameters, momentum=0.9, 
                    nesterov=True)

    return optimizer


def create_criterion(pad_idx, vocab_size, device, smoothing=0.1):
    """
    Creates label smoothing loss with kl-divergence for the
    seq2seq model.
    """
    confidence = 1.0 - smoothing
    # initializes the target distribution vector with smoothing
    # value divided by the number of other valid tokens
    smoothed = torch.full(
        (vocab_size, ), smoothing / (vocab_size - 2)).to(device)
    smoothed[pad_idx] = 0

    def label_smoothing(outputs, targets):
        """
        Computes the kl-divergence between the preds and the 
        smoothed target distribution.
        """
        smoothed_targets = smoothed.repeat(targets.size(0), 1)
        smoothed_targets.scatter_(
            1, targets.unsqueeze(1), confidence)
        smoothed_targets.masked_fill_(
            (targets == pad_idx).unsqueeze(1), 0)

        return kl_div(outputs, smoothed_targets, reduction='sum')

    return label_smoothing


def compute_loss(outputs, targets, criterion, pad_idx):
    """
    Computes the loss and accuracy with masking.
    """
    scores, preds = outputs
    scores_view = scores.view(-1, scores.size(-1))
    targets_view = targets.view(-1)

    loss = criterion(scores_view, targets_view)

    # computing accuracy without including the pad tokens
    notpad = targets.ne(pad_idx)
    target_tokens = notpad.long().sum().item()
    correct = ((targets == preds) * notpad).sum().item()
    accuracy = correct / target_tokens

    loss = loss / target_tokens

    return loss, accuracy


def compute_bleu(outputs, targets, indices, fields):
    """
    Computes the bleu score for a batch of output 
    target pairs.
    """
    _, trg = fields
    _, end_idx, _, trg_pad_idx, _ = indices
    ignored = end_idx, trg_pad_idx

    references = ids2text(targets, trg, ignored)
    hypotesis = ids2text(outputs, trg, ignored)

    return [sentence_bleu([ref], hyp) for
            ref, hyp, in zip(references, hypotesis)]


def train_step(model, criterion, optimizer, batch, indices):
    """
    Performs a single step of training 
    """
    optimizer.zero_grad()

    inputs, targets = batch.src, batch.trg

    # the first token is the sos which is also
    # created by the decoder internally
    targets = targets[1:]
    max_len = targets.size(0)

    outputs = model(
        inputs=inputs,
        targets=targets,
        max_len=max_len)

    loss, accuracy = compute_loss(
        outputs=outputs,
        targets=targets,
        criterion=criterion,
        pad_idx=indices[3])

    loss.backward()

    # clipping gradients enhances sgd performance
    # and prevents exploding gradient problem
    clip_grad_norm_(model.parameters(), 0.25)

    optimizer.step()

    return loss.item(), accuracy


def eval_step(model, criterion, batch, indices, device):
    """
    Performs a single step of evaluation.
    """
    inputs, targets = batch.src, batch.trg

    targets = targets[1:]
    max_len = targets.size(0)

    outputs = model(inputs=inputs, max_len=max_len)

    loss, accuracy = compute_loss(
        outputs=outputs,
        targets=targets,
        criterion=criterion,
        pad_idx=indices[3])

    _, preds = outputs

    return loss.item(), accuracy, preds


def train(model, datasets, fields, args, device):
    """
    Performs training, validation and testing.
    """
    _, trg = fields
    indices = get_special_indices(fields)

    train, val, test = datasets
    criterion = create_criterion(
        pad_idx=indices[3], 
        vocab_size=len(trg.vocab), 
        device=device)

    # creating optimizer with learning rate schedule
    optimizer = create_optimizer(args, model.parameters())
    
    prev_avg_loss = load_state(
        model, optimizer, args.model_dir, device)

    # TODO apex bug with weight drop
    if args.mixed and args.cuda and False:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O1')

    for epoch in range(args.epochs):
        # running training loop
        loop = tqdm(train)
        loop.set_description('{}'.format(epoch))
        model.train()

        scheduler = CosineAnnealingLR(
            optimizer, len(train), eta_min=1e-6)

        for batch in loop:
            loss, accuracy = train_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                indices=indices,
                batch=batch)

            scheduler.step()

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=accuracy))

        # running validation loop
        loop = tqdm(val)
        model.eval()
        avg_loss = []

        with torch.no_grad():
            for batch in loop:
                loss, accuracy, _ = eval_step(
                    model=model,
                    criterion=criterion,
                    indices=indices,
                    device=device,
                    batch=batch)

                avg_loss.append(loss)
                loop.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy))

        avg_loss = sum(avg_loss) / len(avg_loss)
        print('avg val loss: {:.4}'.format(avg_loss))
        if avg_loss < prev_avg_loss:
            save_state(
                model, optimizer, avg_loss, args.model_dir)
            prev_avg_loss = avg_loss

    # running testing loop
    loop = tqdm(test)
    model.eval()
    bleu_scores = []

    with torch.no_grad():
        for batch in loop:
            loss, accuracy, outputs = eval_step(
                model=model,
                criterion=criterion,
                indices=indices,
                device=device,
                batch=batch)

            bleu_scores.extend(compute_bleu(
                outputs, batch.trg, indices, fields))

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=accuracy))

    print('test bleu score: {:.4}'.format(
        sum(bleu_scores) / len(bleu_scores)))


def main():
    args = setup_train_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    datasets, fields = create_datasets(args, device)
    model = create_model(args, fields, device)

    train(model, datasets, fields, args, device)


if __name__ == '__main__':
    main()
