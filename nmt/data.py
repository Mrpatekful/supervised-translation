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


def setup_data_args(parser):
    """"""
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Size of the batches during training.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the data root directory.')
    

def tokenize(sentence, tokenizer):
    """"""
    return [tok.text for tok in tokenizer(sentence)]


def create_datasets(args):
    """"""
    # src_lang, tgt_lang = spacy.load('en_core_web_sm'), \
    #     spacy.load('de_core_news_sm')

    english = Field(
        # tokenize=lambda s: tokenize(s, src_lang.tokenizer),
        batch_first=True)

    german = Field(
        # tokenize=lambda s: tokenize(s, tgt_lang.tokenizer), 
        init_token = "<sos>", 
        eos_token = "<eos>",
        batch_first=True)

    train, val, test = Multi30k.splits(
        exts=('.en', '.de'),
        fields=(english, german),
        root=args.data_dir)

    iterators = BucketIterator.splits(
        (train, val, test), batch_sizes=[args.batch_size] * 3,
        sort_key=lambda x: len(x.src), device=0)

    english.build_vocab()
    german.build_vocab()
    
    return iterators
