"""

@author:    Patrik Purgai
@copyright: Copyright 2019, supervised-translation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=no-name-in-module

import argparse
import torch

from os.path import join

from beam import (
    beam_search,
    setup_beam_args)

from data import (
    PAD, END,
    ids2text,
    text2ids,
    get_special_indices)

from model import (
    create_model,
    setup_model_args)


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


def main():
    args = setup_eval_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    state_dict = torch.load(
        join(args.model_dir, 'model.pt'),
        map_location=device)
    fields = torch.load(
        join(args.model_dir, 'fields.pt'),
        map_location=device)

    SRC, TRG = fields['src'], fields['trg']

    indices = get_special_indices((SRC, TRG))

    model = create_model(args, (SRC, TRG), device)
    model.load_state_dict(state_dict['model'])
    model.eval()

    @torch.no_grad()
    def translate(text):
        """
        Translates the given text with beam search.
        """
        ids = text2ids(text, SRC)
        _, preds = beam_search(
            model=model,
            inputs=ids,
            indices=indices,
            beam_size=args.beam_size,
            device=device)
        output = ids2text([preds.squeeze()], TRG)[0]

        return ' '.join(w for w in output
                        if w not in (PAD, END))

    print('Type a sentence to translate. ' + \
          'CTRL + C to escape.')
          
    while True:
        try:
            print()
            text = input()
            output = translate(text)
            print(output)
            print()

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
