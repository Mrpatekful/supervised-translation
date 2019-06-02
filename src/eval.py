"""

@author:    Patrik Purgai
@copyright: Copyright 2019, nmt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=no-name-in-module

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from os.path import join

from beam import setup_beam_args, beam_search

from data import (
    PAD, END,
    ids2text,
    text2ids,
    get_special_indices)

from model import create_model, setup_model_args


def setup_eval_args():
    """
    Sets up the arguments for evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Path of the model file.')
    parser.add_argument(
        '--cuda',
        type=bool,
        default=False,
        help='Device for evaluation.')

    setup_beam_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


@torch.no_grad()
def translate(text, model, fields, vocabs, indices, beam_size,
              device):
    """
    Translates the given text with beam search.
    """
    SRC, TRG = fields
    ids = text2ids(text, SRC)
    _, preds = beam_search(
        model=model,
        inputs=ids,
        indices=indices,
        beam_size=beam_size,
        device=device)
    output = ids2text([preds.squeeze()], TRG)[0]

    return ' '.join(w for w in output if w not in (PAD, END))


def main():
    args = setup_eval_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    state_dict = torch.load(join(args.model_dir, 'model.pt'),
                            map_location=device)
    fields = torch.load(join(args.model_dir, 'fields.pt'),
                        map_location=device)

    SRC, TRG = fields['src'], fields['trg']

    fields = SRC, TRG
    vocabs = SRC.vocab, TRG.vocab
    indices = get_special_indices(fields)

    model = create_model(args, fields, device)
    model.load_state_dict(state_dict['model'])
    model.eval()

    print('Type a sentence. CTRL + C to escape.')

    while True:
        try:
            print()
            text = input()
            output = translate(
                text=text, model=model, fields=fields,
                vocabs=vocabs, indices=indices,
                beam_size=args.beam_size, device=device)
            print(output)
            print()

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
