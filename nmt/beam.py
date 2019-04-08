"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import torch


def setup_beam_args(parser):
    parser.add_argument('')


@torch.no_grad()
def beam_search_decode(model, batch):
    return batch
