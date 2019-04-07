"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import torch


def setup_model_args(parser):
    """"""
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='seq2seq',
        help='Name of the model.'
    )


def create_model(opt):
    pass
