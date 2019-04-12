"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import torch


def setup_beam_args(parser):
    parser.add_argument(
        '--beam_width',
        type=int,
        default=3,
        help='Number of beam segments.')


def decode_beam_search(model, inputs, max_len=50):
    # TODO implement beam search
    return inputs
