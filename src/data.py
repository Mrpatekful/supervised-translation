"""

@author:    Patrik Purgai
@copyright: Copyright 2019, nmt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import spacy
import pickle

from os.path import join, dirname, abspath
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from nltk.translate import bleu

START_TOKEN = '<sos>'
END_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


def setup_data_args(parser):
    """"""
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Size of the batches during training.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the data root directory.')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=3000,
        help='Maximum size of the vocabulary.')
    

def tokenize(sentence, tokenizer):
    """"""
    return [tok.text for tok in tokenizer(sentence)]


def create_datasets(args, device):
    """"""
    src = Field(
        batch_first=True)

    trg = Field(
        init_token=START_TOKEN, 
        eos_token=END_TOKEN,
        batch_first=True,
        is_target=True)

    train, valid, test = Multi30k.splits(
        exts=('.en', '.de'),
        fields=(src, trg),
        root=args.data_dir)

    src.build_vocab(train, valid, max_size=args.vocab_size)
    trg.build_vocab(train, valid, max_size=args.vocab_size)

    vocabs = src.vocab, trg.vocab

    iterators = BucketIterator.splits(
        (train, valid, test), batch_sizes=[args.batch_size] * 3,
        sort_key=lambda x: len(x.src), device=device)

    return iterators, vocabs


def get_indices(vocabs):
    """"""
    src_vocab, trg_vocab = vocabs

    start_index = trg_vocab.stoi[START_TOKEN]
    end_index = trg_vocab.stoi[END_TOKEN]

    src_pad_index = src_vocab.stoi[PAD_TOKEN]
    trg_pad_index = trg_vocab.stoi[PAD_TOKEN]

    return start_index, end_index, src_pad_index, trg_pad_index
