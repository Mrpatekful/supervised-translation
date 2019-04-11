"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import spacy

from os.path import join, dirname, abspath
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


START_TOKEN = '<sos>'
END_TOKEN = '<eos>'


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


def create_datasets(args):
    """"""
    english = Field(batch_first=True)

    german = Field(
        init_token=START_TOKEN, 
        eos_token=END_TOKEN,
        batch_first=True,
        is_target=True)

    train, valid, test = Multi30k.splits(
        exts=('.en', '.de'),
        fields=(english, german),
        root=args.data_dir)

    iterators = BucketIterator.splits(
        (train, valid, test), batch_sizes=[args.batch_size] * 3,
        sort_key=lambda x: len(x.src), device=0)

    english.build_vocab(train, valid, max_size=args.vocab_size)
    german.build_vocab(train, valid, max_size=args.vocab_size)

    pad_token = german.vocab.stoi['<pad>']
    start_token = german.vocab.stoi[START_TOKEN]
    end_token = german.vocab.stoi[END_TOKEN]

    tokens = pad_token, start_token, end_token

    return iterators, (len(english.vocab), len(german.vocab)), tokens
