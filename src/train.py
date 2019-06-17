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

from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime

from torch.nn.functional import kl_div
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from torchnlp.metrics.bleu import get_moses_multi_bleu

from os.path import (
    exists, join,
    dirname, abspath)

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
torch.backends.cudnn.benchmark = False

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
        default=64,
        help='Size of the batches during training.')

    setup_data_args(parser)
    setup_model_args(parser)
    setup_beam_args(parser)

    return parser.parse_args()


def save_state(model, optimizer, avg_acc, epoch, path):
    """
    Saves the model and optimizer state.
    """
    model_path = join(path, 'model.pt')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'avg_acc': avg_acc,
        'epoch': epoch
    }
    print('Saving model to {}'.format(model_path))
    # making sure the model saving is not left in a
    # corrupted state after a keyboard interrupt
    while True:
        try:
            torch.save(state, model_path)
            break
        except KeyboardInterrupt:
            pass


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
        return state_dict['avg_acc'], state_dict['epoch']

    except FileNotFoundError:
        return -np.inf, 0


def create_optimizer(args, parameters):
    """
    Creates an adam or swats optimizer with cyclical 
    learning rate.
    """
    optimizer = SGD(
        lr=args.learning_rate, weight_decay=1e-6,
        params=parameters, momentum=0.9, nesterov=True)

    return optimizer


def create_criterion(pad_idx, vocab_size, device, 
                     smoothing=0.1):
    """
    Creates label smoothing loss with kl-divergence for the
    seq2seq model.
    """
    confidence = 1.0 - smoothing
    # initializes the target distribution vector with 
    # smoothing value divided by the number of other 
    # valid tokens
    smoothed = torch.full(
        (vocab_size, ), smoothing / (vocab_size - 2))
    smoothed = smoothed.to(device)
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

        return kl_div(outputs, smoothed_targets, 
                      reduction='sum')

    return label_smoothing


def compute_lr(step, factor=5e-2, warmup_steps=3):
    """
    Calculates learning rate with warm up.
    """
    if step < warmup_steps:
        return (1 + factor) ** step
    else:
        # after reaching maximum number of steps
        # the lr is decreased by factor as well
        return ((1 + factor) ** warmup_steps) * \
            ((1 - factor) ** (step - warmup_steps))


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
    _, TRG = fields
    _, end_idx, _, trg_pad_idx, _ = indices
    ignored = end_idx, trg_pad_idx

    hypotheses = ids2text(outputs, TRG, ignored)
    references = ids2text(targets, TRG, ignored)
    print(len(hypotheses))

    return get_moses_multi_bleu(hypotheses, references)


def main():
    """
    Performs training, validation and testing.
    """
    args = setup_train_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # creating dataset and storing dataset splits, fields
    # and special indices as individual variables
    # for convenience
    (train, val, test), (SRC, TRG) = create_datasets(
        args=args, device=device)

    indices = get_special_indices((SRC, TRG))
    _, _, _, trg_pad_idx, _ = indices

    model = create_model(
        args=args, fields=(SRC, TRG), device=device)

    optimizer = create_optimizer(
        args=args, parameters=model.parameters())

    criterion = create_criterion(
        pad_idx=trg_pad_idx, vocab_size=len(TRG.vocab), 
        device=device)

    best_avg_acc, init_epoch = load_state(
        model, optimizer, args.model_dir, device)

    def train_step(batch):
        """
        Performs a single step of training 
        """
        optimizer.zero_grad()

        inputs, targets = batch.src, batch.trg

        # the first token is the sos which is also
        # created by the decoder internally
        targets = targets[:, 1:]
        targets = targets.contiguous()
        max_len = targets.size(1)

        outputs = model(
            inputs=inputs,
            targets=targets,
            max_len=max_len)

        loss, accuracy = compute_loss(
            outputs=outputs,
            targets=targets,
            criterion=criterion,
            pad_idx=trg_pad_idx)

        loss.backward()

        # clipping gradients enhances sgd performance
        # and prevents exploding gradient problem
        # clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

        return loss.item(), accuracy

    def eval_step(batch):
        """
        Performs a single step of evaluation.
        """
        inputs, targets = batch.src, batch.trg

        targets = targets[:, 1:]
        targets = targets.contiguous()
        max_len = targets.size(1)

        outputs = model(
            inputs=inputs,
            max_len=max_len)

        loss, accuracy = compute_loss(
            outputs=outputs,
            targets=targets,
            criterion=criterion,
            pad_idx=trg_pad_idx)

        _, preds = outputs

        return loss.item(), accuracy, preds

    scheduler = LambdaLR(optimizer, compute_lr)
        
    for epoch in range(init_epoch, args.epochs):
        # running training loop
        loop = tqdm(train)
        loop.set_description('{}'.format(epoch))
        model.train()

        for batch in loop:
            loss, accuracy = train_step(batch)

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=accuracy))

        # running validation loop
        loop = tqdm(val)
        model.eval()
        avg_acc = []

        with torch.no_grad():
            for batch in loop:
                loss, accuracy, _ = eval_step(batch)

                avg_acc.append(accuracy)

                loop.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy))

        avg_acc = sum(avg_acc) / len(avg_acc)
        print('avg val acc: {:.4}'.format(avg_acc))
        if avg_acc > best_avg_acc:
            save_state(model, optimizer, avg_acc, 
                       epoch, args.model_dir)
            best_avg_acc = avg_acc
        
        scheduler.step()

    # running testing loop
    loop = tqdm(test)
    model.eval()
    hypotheses, references = [], []

    with torch.no_grad():
        for batch in loop:
            loss, accuracy, outputs = eval_step(batch)

            hypotheses.extend(outputs)
            references.extend(batch.trg)

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=accuracy))

    bleu_score = compute_bleu(
        hypotheses, references, indices, (SRC, TRG))

    print('test bleu score: {:.4}'.format(bleu_score))


if __name__ == '__main__':
    main()
