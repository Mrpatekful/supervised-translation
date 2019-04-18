"""

@author:    Patrik Purgai
@copyright: Copyright 2019, nmt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=not-callable
# pylint: disable=no-member

import torch

from torch.nn.functional import log_softmax


NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def setup_beam_args(parser):
    """"""
    parser.add_argument(
        '--beam_size',
        type=int,
        default=3,
        help='Number of beam segments.')


def neginf(dtype):
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def create_beams(*args, batch_size, **kwargs):
    """"""
    return [Beam(*args, **kwargs) for _ in range(batch_size)]


def beam_search(model, inputs, indices, beam_size, device, max_len=50):
    """"""
    batch_size = inputs.size(0)
    _, trg_pad_index, start_index, end_index = indices 

    encoder_outputs, hidden_state = model.encoder(inputs)

    beams = create_beams(
        beam_size=beam_size, pad_index=trg_pad_index, 
        start_index=start_index, end_index=end_index,
        min_length=3, device=device, batch_size=batch_size)
    
    decoder_input = torch.tensor(
        start_index).to(device)
    decoder_input = decoder_input.expand(
        batch_size * beam_size, 1)

    indices = torch.arange(
        batch_size).to(device)
    indices = indices.unsqueeze(1).repeat(
        1, beam_size).view(-1)
    
    encoder_outputs = encoder_outputs.index_select(0, indices)
    hidden_state = tuple(
        s.index_select(1, indices) for s in hidden_state)

    for _ in range(max_len):
        if all(beam.finished for beam in beams):
            break

        logits, hidden_state = model.decoder(
            decoder_input, encoder_outputs, hidden_state)

        logits = logits[:, -1:, :]
        scores = log_softmax(logits, dim=2)

        scores = scores.view(batch_size, beam_size, -1)

        for index, beam in enumerate(beams):
            if not beam.finished:
                beam.step(scores[index])

    return inputs


class Beam:

    def __init__(self, beam_size, pad_index, start_index,
                 end_index, min_length, device):
        """"""
        self.beam_size = beam_size
        self.pad_index = pad_index
        self.start_index = start_index
        self.end_index = end_index
        self.min_length = min_length
        self.device = device

        self.scores = torch.zeros(beam_size).to(device)
        self.history = []
        self.outputs = [torch.tensor(beam_size).long()
                        .fill_(start_index).to(device)]

        self.finished = False

    def step(self, scores):
        """"""
        vocab_size = scores.size(1)
        print(scores.size())
        current_length = len(self.all_scores) - 1

        if current_length < self.min_length:
            for hyp_id in range(scores.size(0)):
                scores[hyp_id][self.end_index] = neginf(scores.dtype)

        if len(self.bookkeep) == 0:
            beam_scores = scores[0]

        else:
            beam_scores = (
                scores + self.scores.unsqueeze(1).expand_as(scores))
            for index in range(self.outputs[-1].size(0)):
                if self.outputs[-1][index] == self.end_index:
                    beam_scores[index] = neginf(scores.dtype)

        flatten_beam_scores = beam_scores.view(-1)
        best_scores, best_idxs = torch.topk(
            flatten_beam_scores, self.beam_size, dim=-1)

        self.scoress = best_scores
        self.all_scores.append(self.scores)
        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs / vocab_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % vocab_size

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [self.partial_hyps[hyp_ids[i]] +
                             [tok_ids[i].item()] for i in range(self.beam_size)]

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                eostail = self.HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.scores[hypid],
                    tokenid=self.eos)

                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1
