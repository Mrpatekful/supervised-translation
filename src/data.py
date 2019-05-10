"""

@author:    Patrik Purgai
@copyright: Copyright 2019, nmt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import torch
import sys
import re

from os.path import exists, join, dirname, abspath
from torchtext.datasets import Multi30k
from torchtext.data import Field, HarvardBucketIterator
from nltk.translate import bleu


START = '<sos>'
END = '<eos>'
PAD = '<pad>'
UNK = '<unk>'


def setup_data_args(parser):
    """
    Sets up the data arguments.
    """
    parser.add_argument(
        '--data_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the data root directory.')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=20000,
        help='Maximum size of the vocabulary.')
    parser.add_argument(
        '--min_freq',
        type=int,
        default=2,
        help='Minimum frequency of a word in the vocab.')
    parser.add_argument(
        '--max_len',
        type=int,
        default=50,
        help='Maximum length of a sequence.')


def text2ids(text, field):
    """
    Converts a text to list of ids.
    """
    tokenized = field.preprocess(text)
    return field.numericalize([tokenized])


def ids2text(ids, field, ignored=None):
    """
    Converts a list of ids to text.
    """
    if ignored is None:
        ignored = []
    return [[field.vocab.itos[i] for 
        i in s if i not in ignored] for s in ids]


def create_datasets(args, device):
    """
    Either loads the fields or creates the datasets.
    """
    fields_path = join(args.model_dir, 'fields.pt')
    if not exists(fields_path):
        src = Field(
            batch_first=True,
            pad_token=PAD,
            unk_token=UNK,
            tokenize='spacy',
            tokenizer_language='en_core_web_sm',
            lower=True)

        trg = Field(
            init_token=START, 
            eos_token=END,
            pad_token=PAD,
            unk_token=UNK,
            batch_first=True,
            tokenize='spacy',
            tokenizer_language='de_core_news_sm',
            lower=True,
            is_target=True)
    else:
        print('Loading fields from {}'.format(fields_path))
        fields = torch.load(fields_path)
        src = fields['src']
        trg = fields['trg']

    train, valid, test = Multi30k.splits(
        exts=('.en', '.de'),
        fields=(src, trg),
        root=args.data_dir)

    if not exists(fields_path):
        # including specials first, so the last indices
        # can be truncated when creating
        # the output layer of the vocab.
        src.build_vocab(
            train, valid, 
            specials_first=False,
            min_freq=args.min_freq,
            max_size=args.vocab_size)

        trg.build_vocab(
            train, valid, 
            specials_first=False,
            min_freq=args.min_freq,
            max_size=args.vocab_size)

        print('Saving fields to {}'.format(fields_path))
        torch.save({'src': src, 'trg': trg}, fields_path)

    fields = src, trg

    iterators = HarvardBucketIterator.splits(
        (train, valid, test),
        batch_sizes=[args.batch_size] * 3,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        device=device)

    return iterators, fields


def get_special_indices(fields):
    """
    Returns the special token indices from the vocab.
    """
    src, trg = fields

    start_idx = trg.vocab.stoi[START]
    end_idx = trg.vocab.stoi[END]

    unk_idx = src.vocab.stoi[UNK]

    src_pad_idx = src.vocab.stoi[PAD]
    trg_pad_idx = trg.vocab.stoi[PAD]

    return start_idx, end_idx, src_pad_idx, \
        trg_pad_idx, unk_idx
