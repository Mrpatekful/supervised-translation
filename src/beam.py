"""
@author:    Patrik Purgai
@copyright: Copyright 2019, supervised-translation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import math

from numpy import inf
from torch.nn.functional import log_softmax
from collections import namedtuple


Result = namedtuple('Result', ['hyp_id', 'time_step', 'score'])


def setup_beam_args(parser):
    """
    Sets up the parameters for the beam search.
    """
    parser.add_argument(
        '--beam_size',
        type=int,
        default=10,
        help='Number of beam segments.')


def create_beams(*args, batch_size, **kwargs):
    """
    Creates a beam for each sample in a batch.
    """
    return [Beam(*args, **kwargs) for _ in range(batch_size)]


def select_hidden_states(hidden_states, indices):
    """
    Prepares the hidden states for the next step.
    """
    hidden_states = tuple(
        hs.index_select(1, indices) for hs in hidden_states)
    return hidden_states


def beam_search(model, inputs, indices, beam_size, device, 
                max_len=50):
    """
    Applies beam search decoding on the provided inputs.
    Implementation is based on `facebookresearch ParlAI`
    and `PyTorch-OpenNMT`.
    """
    batch_size = inputs.size(0)
    start_index, *_ = indices

    encoder_outputs, hidden_state = model.encoder(inputs)

    # a beam is created for each element of the batch
    beams = create_beams(
        beam_size=beam_size, indices=indices,
        max_len=max_len, device=device,
        batch_size=batch_size)

    # the decoder has beam_size * batch_size inputs
    decoder_input = torch.tensor(start_index).to(device)
    decoder_input = decoder_input.expand(
        batch_size * beam_size, 1)

    indices = torch.arange(batch_size).to(device)
    indices = indices.unsqueeze(1).repeat(
        1, beam_size).view(-1)

    # each encoder output is copied beam_size times,
    # making `encoder_outputs` of size
    # [batch_size * beam_size, seq_len, hidden_size]
    encoder_outputs = encoder_outputs.index_select(0, indices)
    hidden_state = select_hidden_states(hidden_state, indices)

    for _ in range(max_len):
        if all(beam.finished for beam in beams):
            break

        logits, hidden_state = model.decoder(
            inputs=decoder_input[:, -1:],
            encoder_outputs=encoder_outputs,
            prev_hiddens=hidden_state)

        logits = logits[:, -1:, :]
        scores = log_softmax(logits, dim=-1)
        scores = scores.view(batch_size, beam_size, -1)

        # each beam receives the corresponding score
        # output, to calculate the best candidates
        for idx, beam in enumerate(beams):
            if not beam.finished:
                beam.step(scores[idx])

        # prepares the indices, which select the hidden states
        # of the best scoring outputs
        indices = torch.cat([
            beam_size * idx + beam.hyp_ids[-1]
            for idx, beam in enumerate(beams)
        ])

        hidden_state = select_hidden_states(hidden_state, indices)
        decoder_input = torch.index_select(decoder_input, 0, indices)

        prev_output = torch.cat([b.token_ids[-1] for b in beams])
        prev_output = prev_output.unsqueeze(-1)
        decoder_input = torch.cat([decoder_input, prev_output], dim=-1)

    # merging the best result from the beams into
    # a single batch of outputs
    top_scores, top_preds = list(
        zip(*[b.get_result(decoder_input.size(-1)) for b in beams]))

    top_preds = torch.cat(top_preds).view(batch_size, -1)
    top_scores = torch.cat(top_scores).view(
        batch_size, top_preds.size(-1), -1)

    return top_scores, top_preds


class Beam:

    def __init__(self, beam_size, indices, max_len, device):
        """
        A beam that contains `beam_size` decoding candidates.
        Each beam operates on a single input sequence from the batch.
        """
        self.beam_size = beam_size
        self.max_len = max_len
        self.finished = False

        self.start_idx, self.end_idx, self.pad_idx, *_ = indices

        # scores of each candidate of the beam
        self.scores = torch.zeros(beam_size).to(device)
        # top score values of each hyp for each time step
        self.top_scores = [torch.zeros(beam_size).to(device)]
        self.all_scores = [None]
        # ids of the previous candidates
        self.hyp_ids = []
        self.results = []

        # output token ids of the hyps at each step
        self.token_ids = [
            torch.tensor(self.start_idx)
            .expand(beam_size).to(device)]

    def step(self, scores):
        """
        Advances the beam a step forward by selecting the
        best `beam_size` number of candidates for the provided
        scores.
        """
        if self.finished:
            return

        # `scores` is the softmax output of the decoder
        vocab_size = scores.size(-1)
        self.all_scores.append(scores)

        if len(self.hyp_ids) == 0:
            # the scores for the first step is simply the
            # first candidate
            beam_scores = scores[0]

        else:
            prev_scores = self.scores.unsqueeze(1)
            prev_scores = prev_scores.expand_as(scores)
            beam_scores = scores + prev_scores

            for idx in range(self.token_ids[-1].size(0)):
                if self.token_ids[-1][idx] == self.end_idx:
                    beam_scores[idx] = -inf

        # flatten beam scores is vocab_size * beam_size
        flatten_beam_scores = beam_scores.view(-1)
        top_scores, top_idxs = torch.topk(
            flatten_beam_scores, self.beam_size, dim=-1)

        self.scores = top_scores
        self.top_scores.append(self.scores)

        # selecting the id of the best hyp at the current step
        self.hyp_ids.append(top_idxs / vocab_size)
        self.token_ids.append(top_idxs % vocab_size)

        time_step = len(self.token_ids)
        for hyp_idx in range(self.beam_size):
            if self.token_ids[-1][hyp_idx] == self.end_idx or \
                    time_step == self.max_len - 1:
                self.results.append(
                    Result(hyp_id=hyp_idx, time_step=time_step,
                           score=self.scores[hyp_idx] / (time_step + 1)))

                if len(self.results) == self.beam_size:
                    self.finished = True
                    break

    def get_result(self, size):
        """
        Returns the tokens and scores of the best hypotesis.
        """
        best_hyp = sorted(self.results, key=lambda x: x.score)[0]
        token_ids, scores = [], []
        hyp_id = best_hyp.hyp_id

        for ts in range(best_hyp.time_step - 1, 0, -1):
            token_ids.insert(0, self.token_ids[ts][hyp_id])
            scores.insert(0, self.all_scores[ts][hyp_id].unsqueeze(0))
            hyp_id = int(self.hyp_ids[ts - 1][hyp_id])

        # creating a tensor of size `size` at first dimension, so
        # it can be concatenated with the results of other beams
        padded_preds = torch.tensor(self.pad_idx).expand(size)

        # apparently cloning is required after expand when
        # performing in-place operations
        padded_preds = padded_preds.clone()
        padded_preds[:len(token_ids)] = torch.tensor(token_ids)

        padded_scores = torch.zeros([1, size, scores[0].size(-1)])
        padded_scores[0, :len(scores), :] = torch.cat(scores, 0)

        return padded_scores, padded_preds
