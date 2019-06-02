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
    log_softmax, softmax, linear)

from torch.nn import (
    GRU, Dropout,
    Embedding, Linear,
    Softmax, Parameter)

from torchnlp.nn import (
    WeightDropGRU, LockedDropout)

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


def embedded_dropout(embed, words, dropout, pad_idx):

    mask = embed.weight.new((embed.weight.size(0), 1)).bernoulli_(
        1 - dropout).expand_as(embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight

    X = torch.nn.functional.embedding(
        words, masked_embed_weight,
        pad_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse)

    return X


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
            vocab_size=target_vocab_size)

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
        # NOTE dropout probability is hardcoded to be 0.1
        if self.training:
            unk_mask = inputs.new(inputs.size()).float().uniform_(0, 1)
            unk_mask = (unk_mask < 0.1) & inputs.ne(self.pad_idx)
            inputs.masked_fill_(mask=unk_mask, value=self.unk_idx)

        # the number of layers in the decoder must be equal
        # to the number of layers in the encoder because of
        # the initial hidden states from the encoder
        encoder_outputs, hidden_states = self.encoder(inputs)

        scores = []
        preds = self.start_idx.detach().expand(1, batch_size)

        for idx in range(max_len):
            # if targets are provided and training then apply
            # teacher forcing 50% of the time
            if targets is not None and random.random() > 0.5 and \
                    self.training:
                step_input = targets[idx].unsqueeze(0)
            else:
                step_input = preds[-1:]

            logits, hidden_states = self.decoder(
                inputs=step_input,
                encoder_outputs=encoder_outputs,
                prev_hiddens=hidden_states,
                attn_mask=attn_mask)

            logits = logits[-1:]
            step_scores = log_softmax(logits, dim=-1)
            _, step_preds = step_scores.max(dim=-1)

            preds = torch.cat([preds, step_preds])
            scores.append(step_scores)

        scores = torch.cat(scores)
        preds = preds.narrow(0, 1, preds.size(1) - 1)

        return scores, preds


class Encoder(Module):
    """
    Encoder module for the seq2seq model.
    """

    def __init__(self, input_size, hidden_size, pad_idx,
                 vocab_size):
        super().__init__()

        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size,
            padding_idx=pad_idx,
            scale_grad_by_freq=True)

        self.dropout = LockedDropout(p=0.3)

        self.merge = Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False)

        # creating rnn layer as module list so drop connect
        # can be applied between each layer
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
        embedded = self.dropout(embedded)

        outputs, hidden_state = self.rnn[0](embedded)

        # merging the two directions of bidirectional layer
        # by summing along the first axis
        hidden_states = [hidden_state.sum(0).unsqueeze(0)]
        outputs = self.merge(outputs)

        for layer in self.rnn[1:]:
            outputs, hidden_state = layer(outputs)
            outputs = self.dropout(outputs)
            hidden_states.append(hidden_state)

        outputs_t = outputs.transpose(0, 1)

        return outputs_t, hidden_states


class Decoder(Module):
    """
    Decoder module for the seq2seq.
    """

    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()

        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size,
            scale_grad_by_freq=True)

        # TODO implement locked dropout for decoder

        self.dropout = Dropout(p=0.3)

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

        self.attn = Attention(
            hidden_size=hidden_size)

        self.project = Linear(
            in_features=hidden_size,
            out_features=input_size)

        # initializing output layer like this
        # instead of torch.nn.Linear so
        # shared embedding can be implemented easily
        self.output_bias = Parameter(torch.zeros((vocab_size, )))
        self.output_weight = self.embedding.weight

    def forward(self, inputs, encoder_outputs, prev_hiddens,
                attn_mask=None):
        """
        Applies decoding with attention mechanism.
        """
        embedded = self.embedding(inputs)
        output = self.dropout(embedded)

        hidden_states = []
        for idx, layer in enumerate(self.rnn):
            output, hidden_state = layer(
                output, prev_hiddens[idx])
            output = self.dropout(output)
            hidden_states.append(hidden_state)

        output, _ = self.attn(
            decoder_output=output,
            hidden_state=hidden_state,
            encoder_outputs=encoder_outputs,
            attn_mask=attn_mask)

        output = self.project(output)

        logits = linear(
            output, self.output_weight, self.output_bias)

        return logits, hidden_states


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
        # selecting the last hidden state of the previous outputs
        hidden_state = hidden_state[-1].unsqueeze(1)
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

        outputs = self.combine(torch.cat(
            (decoder_output.squeeze(1), attn_applied.squeeze(1)),
            dim=1).unsqueeze(1))

        return outputs, attn_weights
