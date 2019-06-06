"""

@author:    Patrik Purgai
@copyright: Copyright 2019, nmt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=no-member
# pylint: disable=not-callable

import torch
import random

from numpy import inf

from torch.nn.modules import (
    Module, ModuleList)

from torch.nn.functional import (
    log_softmax, softmax, linear,
    embedding)

from torch.nn import (
    GRU, Dropout,
    Embedding, Linear,
    Softmax, Parameter)

from torchnlp.nn import (
    WeightDropGRU, LockedDropout,
    DropoutEmbedding)

from data import get_special_indices


def setup_model_args(parser):
    """
    Sets up the model arguments.
    """
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=256,
        help='Hidden size of the model.')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=128,
        help='Embedding dimension for the tokens.')


def create_model(args, fields, device):
    """
    Creates the sequence to sequence model.
    """
    SRC, TRG = fields
    indices = get_special_indices(fields)

    tensor_indices = [torch.tensor(i).to(device) for i in indices]

    model = Seq2Seq(
        source_vocab_size=len(SRC.vocab),
        target_vocab_size=len(TRG.vocab),
        indices=tensor_indices,
        **vars(args)).to(device)

    return model


class Seq2Seq(Module):
    """
    The sequence-to-sequence model.
    """

    def __init__(self, embedding_size,
                 hidden_size, indices,
                 source_vocab_size, target_vocab_size, **kwargs):
        super().__init__()

        self.start_idx, self.end_idx, \
            self.pad_idx, _, self.unk_idx = indices

        self.encoder = Encoder(
            input_size=embedding_size,
            hidden_size=hidden_size,
            pad_idx=self.pad_idx,
            vocab_size=source_vocab_size)

        self.decoder = Decoder(
            input_size=embedding_size,
            hidden_size=hidden_size,
            vocab_size=target_vocab_size,
            num_softmax=15)

    def forward(self, inputs, targets=None, max_len=50):
        """
        Runs the inputs through the encoder-decoder model.
        """
        # inputs are expexted in sequence-first format
        batch_size = inputs.size(1)
        max_len = targets.size(0) if targets is not None else max_len

        attn_mask = inputs.eq(self.pad_idx).t()

        # randomly masking input with unk during training
        # which makes the model more robust to unks during testing
        # NOTE unk dropout probability is hardcoded to be 0.1
        if self.training:
            unk_mask = inputs.new_empty(inputs.size()).bernoulli_(0.1)
            unk_mask = unk_mask.byte() & inputs.ne(self.pad_idx)
            inputs.masked_fill_(mask=unk_mask, value=self.unk_idx)

        # the number of layers in the decoder must be equal
        # to the number of layers in the encoder because of
        # the initial hidden states from the encoder
        encoder_outputs, hidden_states = self.encoder(inputs)

        scores = []
        preds = self.start_idx.detach().expand(1, batch_size)

        # computing weight dropout and regular dropout for
        # each timestep
        for layer in self.decoder.rnn:
            layer.compute_mask()

        for layer in self.decoder.dropout:
            layer.compute_mask(batch_size)

        for idx in range(max_len):
            # if targets are provided and training then apply
            # teacher forcing 50% of the time
            if targets is not None and random.random() > 0.5 and \
                    self.training:
                prev_output = targets[idx].unsqueeze(0)
            else:
                prev_output = preds[-1:]

            step_scores, hidden_states = self.decoder(
                inputs=prev_output,
                encoder_outputs=encoder_outputs,
                prev_hiddens=hidden_states,
                attn_mask=attn_mask)

            _, step_preds = step_scores.max(dim=-1)

            preds = torch.cat([preds, step_preds])
            scores.append(step_scores)

        scores = torch.cat(scores)
        preds = preds.narrow(0, 1, preds.size(0) - 1)

        return scores, preds


class Encoder(Module):
    """
    Encoder module for the seq2seq model.
    """

    def __init__(self, input_size, hidden_size, pad_idx,
                 vocab_size):
        super().__init__()

        self.embedding = DropoutEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size,
            padding_idx=pad_idx)

        self.dropout = ModuleList([
            LockedDropout(p=0.3, in_features=input_size),
            LockedDropout(p=0.3, in_features=hidden_size)
        ])

        self.merge = Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False)

        # creating rnn layer as module list so locked
        # dropout can be applied between each layer
        self.rnn = ModuleList([
            WeightDropGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                bidirectional=True,
                weight_dropout=0.5)] + [
            WeightDropGRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                weight_dropout=0.5)
            for _ in range(2)
        ])

    def forward(self, inputs):
        """
        Computes the embeddings and runs them through an RNN.
        """
        embedded = self.embedding(inputs)
        embedded = self.dropout[0](embedded)

        outputs, hidden_state = self.rnn[0](embedded)

        # merging the two directions of bidirectional layer
        # by summing along the first axis
        hidden_states = [hidden_state.sum(0, keepdim=True)]
        outputs = self.merge(outputs)

        for layer in self.rnn[1:]:
            outputs, hidden_state = layer(outputs)
            outputs = self.dropout[1](outputs)
            hidden_states.append(hidden_state)

        outputs_t = outputs.transpose(0, 1)

        return outputs_t, hidden_states


class Decoder(Module):
    """
    Decoder module for the seq2seq.
    """

    def __init__(self, input_size, hidden_size, vocab_size,
                 num_softmax):
        super().__init__()

        self.num_softmax = num_softmax
        self.input_size = input_size
        self.vocab_size = vocab_size

        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size)

        self.rnn = ModuleList([
            WeightDropGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                weight_dropout=0.5)] + [
            WeightDropGRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                weight_dropout=0.5)
            for _ in range(2)
        ])

        # an individual dropout must be created for
        # each layer so their mask can be shared
        # for each time step
        self.dropout = ModuleList([
            LockedDropout(p=0.3, in_features=input_size)] + [
            LockedDropout(p=0.3, in_features=hidden_size) 
            for _ in range(len(self.rnn))] + [
            LockedDropout(p=0.3, 
                in_features=num_softmax * input_size),
        ])

        self.attn = Attention(
            hidden_size=hidden_size)

        # weight matrices for mixture of softmaxes
        self.prior = Linear(
            in_features=hidden_size,
            out_features=num_softmax,
            bias=False)

        self.latent = Linear(
            in_features=hidden_size,
            out_features=num_softmax * input_size)

        # initializing output layer like this
        # instead of torch.nn.Linear so
        # shared embedding can be implemented easily
        self.out_bias = Parameter(torch.zeros((vocab_size, )))
        self.out_weight = self.embedding.weight

    def forward(self, inputs, encoder_outputs, prev_hiddens,
                attn_mask=None):
        """
        Applies decoding with attention mechanism, mixture
        of sofmaxes and multi dropout during training.
        """
        embedded = self.embedding(inputs)
        # each dropout layer is called with `compute_mask=False`
        # so the same dropout mask will be used for each timestep
        output = self.dropout[0](
            embedded, compute_mask=False)

        hidden_states = []
        for idx, layer in enumerate(self.rnn):
            output, hidden_state = layer(
                output, prev_hiddens[idx], 
                compute_mask=False)
            output = self.dropout[idx + 1](
                output, compute_mask=False)
            hidden_states.append(hidden_state)

        output, _ = self.attn(
            decoder_output=output,
            hidden_state=hidden_state,
            encoder_outputs=encoder_outputs,
            attn_mask=attn_mask)

        latent = torch.tanh(self.latent(output))
        latent = self.dropout[len(self.rnn) + 1](
            latent, compute_mask=False)
        latent = latent.view(-1, self.input_size)

        # computing softmaxes from contexts
        context = linear(
            latent, self.out_weight, self.out_bias)
        softmaxes = softmax(context, dim=-1)
        softmaxes = softmaxes.view(
            -1, self.num_softmax, self.vocab_size)

        # computing prior for softmax weighting
        prior = self.prior(output).view(-1, self.num_softmax)
        prior = softmax(prior, dim=-1).unsqueeze(2)

        probs = (softmaxes * prior).sum(1).view(
            1, inputs.size(1), self.vocab_size)
        # adding 1e-8 for numerical stability before
        # transforming to log space
        log_probs = probs.add_(1e-8).log()

        return log_probs, hidden_states


class Attention(Module):
    """
    Luong style general attention from 
    https://arxiv.org/pdf/1508.04025.pdf.
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.project = Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False)

        self.combine = Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False)

    def forward(self, decoder_output, hidden_state, encoder_outputs,
                attn_mask=None):
        """
        Applies attention by creating the weighted context vector.
        Implementation is based on `IBM/pytorch-seq2seq`.
        """
        # converting sequence first format to batch first for bmm
        hidden_state = hidden_state.transpose(0, 1)
        hidden_state = self.project(hidden_state)

        encoder_outputs_t = encoder_outputs.transpose(1, 2)
        attn_scores = torch.bmm(hidden_state, encoder_outputs_t)

        # applying mask on padded values of the input
        # NOTE during beam search mask might not be provided
        if attn_mask is not None:
            attn_scores = attn_scores.squeeze(1)
            attn_scores.masked_fill_(attn_mask, -inf)
            attn_scores = attn_scores.unsqueeze(1)

        attn_weights = softmax(attn_scores, dim=-1)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        # converting back to sequence first format
        attn_applied = attn_applied.transpose(0, 1)

        stacked = torch.cat([decoder_output, attn_applied], dim=-1)
        outputs = self.combine(stacked)

        return outputs, attn_weights
