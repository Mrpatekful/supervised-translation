"""
@author:    Patrik Purgai
@copyright: Copyright 2019, supervised-translation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=import-error
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=not-callable

import torch
import argparse
import os
import logging
import random

import numpy as np

from beam import (
    beam_search, 
    setup_beam_args)

from model import (
    create_model, 
    setup_model_args)

from data import (
    create_dataset,
    setup_data_args)

from tensorboardX import SummaryWriter
from collections import OrderedDict
from tqdm import tqdm
from math import ceil
from datetime import datetime
from statistics import mean

try:
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

from torch.nn.functional import (
    cross_entropy, softmax,
    kl_div, log_softmax)

from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    LambdaLR)

from torch.nn.parallel import (
    DistributedDataParallel)

from torchnlp.metrics.bleu import (
    get_moses_multi_bleu)

from os.path import (
    exists, join,
    dirname, abspath)


def setup_train_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=1000,
        help='Maximum number of epochs for training.')
    parser.add_argument(
        '--cuda',
        type=bool,
        default=torch.cuda.is_available(),
        help='Device for training.')
    parser.add_argument(
        '--mixed',
        type=bool,
        default=APEX_INSTALLED,
        help='Use mixed precision training.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1,
        help='Learning rate for the model.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Size of the batches during training.')
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Patience value for early stopping.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=join(
            abspath(dirname(__file__)), 
            '..', 'model.{}'.format(
                datetime.today().strftime('%j%H%m'))),
        help='Path of the model checkpoints.')
    parser.add_argument(
        '--grad_accum_steps',
        type=int,
        default=4,
        help='Number of steps for grad accum.')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training.')

    setup_data_args(parser)
    setup_model_args(parser)
    setup_beam_args(parser)

    return parser.parse_args()


def load_state(model_dir, model, optimizer, logger, 
               device):
    """
    Loads the model and optimizer state.
    """
    try:
        model_path = join(model_dir, 'model.pt')
        state_dict = torch.load(
            model_path, map_location=device)

        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        logger.info('Loading model from {}'.format(
                model_path))

        return (
            state_dict['val_loss'],
            state_dict['epoch'],
            state_dict['step']
        )

    except FileNotFoundError:
        return np.inf, 0, 0


def create_logger(args):
    """
    Creates a logger that outputs information to a
    file and the standard output as well.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # setting up logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # setting up logging to a file
    filename = '{date}.log'.format(
        date=datetime.today().strftime(
            '%m-%d-%H-%M'))

    log_path = join(args.model_dir, filename)
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


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


def compute_lr(step, factor=3e-3, warmup=50, eps=1e-7):
    """
    Calculates learning rate with warm up.
    """
    if step < warmup:
        return (1 + factor) ** step
    else:
        # after reaching maximum number of steps
        # the lr is decreased by factor as well
        return max(((1 + factor) ** warmup) *
                   ((1 - factor) ** (step - warmup)), eps)


def compute_loss(outputs, targets, criterion, ignore_idx):
    """
    Computes the loss and accuracy with masking.
    """
    log_probs = outputs[0]

    log_probs_view = log_probs.view(-1, log_probs.size(-1))
    targets_view = targets.view(-1)

    loss = criterion(log_probs_view, targets_view)

    _, preds = log_probs.max(dim=-1)

    # computing accuracy without including the
    # values at the ignore indices
    not_ignore = targets_view.ne(ignore_idx)
    target_tokens = not_ignore.long().sum().item()
    
    correct = (targets_view == preds) * not_ignore
    correct = correct.sum().item()

    accuracy = correct / target_tokens
    loss = loss / target_tokens

    return loss, accuracy


def compute_bleu(outputs, targets, tokenizer):
    """
    Computes the bleu score for a batch of output 
    target pairs.
    """
    outputs = tokenizer.decode_ids(outputs)
    targets = tokenizer.decode_ids(targets)

    return get_moses_multi_bleu(
        outputs, targets)


def main():
    """
    Performs training, validation and testing.
    """
    args = setup_train_args()
    master_process = args.local_rank in [0, -1]

    if args.local_rank != -1 and args.cuda:
        # use distributed training if local rank is given
        # and GPU training is requested
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)

        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    else:
        device = torch.device(
            'cuda' if args.cuda else 'cpu')

    # creating dataset and storing dataset splits
    # as individual variables for convenience
    datasets, tokenizers = create_dataset(
        args=args, device=device)

    _, target_tokenizer = tokenizers
    target_vocab_size = len(target_tokenizer)

    model = create_model(
        args=args, tokenizers=tokenizers, 
        device=device)

    optimizer = create_optimizer(
        args=args, parameters=model.parameters())

    writer = SummaryWriter(
        logdir=args.model_dir,
        flush_secs=100)

    logger = create_logger(args=args)

    pad_idx = target_tokenizer.pad_id()

    criterion = create_criterion(
        pad_idx=pad_idx, vocab_size=target_vocab_size,
        device=device)

    # loading previous state of the training
    best_val_loss, init_epoch, step = load_state(
        model_dir=args.model_dir, model=model, 
        optimizer=optimizer, logger=logger,
        device=device)

    if args.mixed and args.cuda:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if args.local_rank != -1:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], 
            output_device=args.local_rank)

    world_size = 1 if not args.cuda \
        else torch.cuda.device_count()

    train, valid, test = [
        (split, ceil(
            size / args.batch_size / world_size)) 
        for split, size in datasets]

    # computing the sizes of the dataset splits
    train_dataset, num_train_steps = train
    valid_dataset, num_valid_steps = valid
    test_dataset, num_test_steps = test

    patience = 0

    def convert_to_tensor(ids):
        """
        Convenience function for converting int32
        ndarray to torch int64.
        """
        return torch.as_tensor(ids).long().to(device)

    def forward_step(batch):
        """
        Applies forward pass with the given batch.
        """
        inputs, attn_mask, targets = batch

        inputs = convert_to_tensor(inputs)
        attn_mask = convert_to_tensor(attn_mask)
        targets = convert_to_tensor(targets)

        outputs = model(
            inputs=inputs,
            attn_mask=attn_mask.byte())

        loss, accuracy = compute_loss(
            outputs=outputs,
            targets=targets,
            criterion=criterion,
            ignore_idx=pad_idx)

        return loss, accuracy, outputs

    def train_step(batch):
        """
        Performs a single step of training.
        """
        nonlocal step

        loss, accuracy, _ = forward_step(batch)

        if torch.isnan(loss).item():
            if master_process:
                logger.warn('skipping step (nan)')
            # returning None values when a NaN loss
            # is encountered and skipping backprop
            # so model grads will not be corrupted
            return None, None

        loss /= args.grad_accum_steps

        backward(loss)
        clip_grad_norm(1.0)

        step += 1

        if step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item(), accuracy

    def backward(loss):
        """
        Backpropagates the loss in either mixed or
        normal precision mode.
        """
        # cuda is required for mixed precision training.
        if args.mixed and args.cuda:
            with amp.scale_loss(loss, optimizer) as scaled:
                scaled.backward()
        else:
            loss.backward()

    def clip_grad_norm(max_norm):
        """
        Applies gradient clipping.
        """
        if args.mixed and args.cuda:
            clip_grad_norm_(
                amp.master_params(optimizer), max_norm)
        else:
            clip_grad_norm_(model.parameters(), max_norm)

    @torch.no_grad()
    def evaluate(dataset, num_steps):
        """
        Constructs a validation loader and evaluates
        the model.
        """
        loop = tqdm(
            dataset(), 
            total=num_steps,
            disable=not master_process,
            desc='Eval')

        model.eval()

        for batch in loop:
            loss, acc, _ = forward_step(batch)

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss.item(), acc=acc))

            yield loss.item()

    def save_state():
        """
        Saves the model and optimizer state.
        """
        model_path = join(args.model_dir, 'model.pt')

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'epoch': epoch + 1,
            'step': step
        }
        
        logger.info('Saving model to {}'.format(model_path))
        # making sure the model saving is not left in a
        # corrupted state after a keyboard interrupt
        while True:
            try:
                torch.save(state, model_path)
                break
            except KeyboardInterrupt:
                pass

    scheduler = LambdaLR(optimizer, compute_lr)

    if master_process:
        logger.info(str(vars(args)))

    for epoch in range(init_epoch, args.max_epochs):
        # running training loop
        loop = tqdm(
            train_dataset(), 
            total=num_train_steps,
            disable=not master_process,
            desc='Train {}'.format(epoch))

        train_loss = []

        model.train()

        for batch in loop:
            try:
                loss, acc = train_step(batch)

                if master_process and loss is not None:
                    train_loss.append(loss)

                    # logging to tensorboard    
                    writer.add_scalar('train/loss', loss, step)
                    writer.add_scalar('train/acc', acc, step)

                loop.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=acc))

                if not step % args.eval_every_step:
                    val_loss = mean(evaluate(
                        dataset=valid_dataset,
                        num_steps=num_valid_steps))
                    
                    # switching back to training
                    model.train()

                    if master_process:
                        logger.info('val loss: {:.4}'.format(
                            val_loss))

                        # logging to tensorboard    
                        writer.add_scalar('val/loss', loss, step)
                        writer.add_scalar('val/acc', acc, step)

                    if val_loss < best_val_loss:
                        patience = 0
                        best_val_loss = val_loss

                        if master_process:
                            save_state()

                    else:
                        patience += 1
                        if patience == args.patience:
                            # terminate when max patience 
                            # level is hit
                            break

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.warn('skipping step (oom)')

        if len(train_loss) > 0:
            train_loss = mean(train_loss)
        else:
            train_loss = 0.0

        if master_process:
            logger.info('train loss: {:.4}'.format(
                train_loss))

        scheduler.step()

    writer.close()

    loop = tqdm(
        test_dataset(), 
        total=num_test_steps,
        disable=not master_process,
        desc='Test')

    model.eval()
    test_loss, hypotheses, references = [], [], []

    # running testing loop
    with torch.no_grad():
        for batch in loop:
            loss, acc, outputs = forward_step(batch)

            hypotheses.extend(outputs)
            references.extend(batch.trg)

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=acc))

            test_loss.append(loss)

    test_loss = mean(test_loss)

    bleu_score = compute_bleu(
        hypotheses, references, target_tokenizer)

    if master_process:
        logger.info('test loss: {:.4}'.format(
            test_loss))

        logger.info('test bleu score: {:.4}'.format(
            bleu_score))


if __name__ == '__main__':
    main()
