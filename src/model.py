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

from torch.nn.modules import Module
from torch.nn.functional import (
    log_softmax, softmax)

from torch.nn import (
    LSTM, Dropout,
    Embedding, Linear,
    Softmax)

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


def neginf():
    """
    Represents the negative infinity for the dtype.
    """
    return -1e20


def create_model(args, fields, device):
    """
    Creates the sequence to sequence model.
    """
    src, trg = fields
    indices = get_special_indices(fields)

    tensor_indices = [torch.tensor(i).to(device) for i in indices]

    model = Seq2Seq(
        source_vocab_size=len(src.vocab), 
        target_vocab_size=len(trg.vocab),
        indices=tensor_indices,
        **vars(args)).to(device)

    return model


def repeat_hidden(hidden_states, times):
    """
    Repeats the tuple of hidden states along the first axis.
    """
    hidden_states = tuple(hs.repeat([times, 1, 1]) 
        for hs in hidden_states)
    return hidden_states


class Seq2Seq(Module):
    """
    The sequence-to-sequence model.
    """

    def __init__(self, embedding_size, 
                 hidden_size, indices,
                 source_vocab_size, target_vocab_size, **kwargs):
        super().__init__()

        self.encoder = Encoder(
            input_size=embedding_size,
            hidden_size=hidden_size,
            vocab_size=source_vocab_size)

        self.decoder = Decoder(
            input_size=embedding_size,
            hidden_size=hidden_size, 
            vocab_size=target_vocab_size)

        self.start_idx, self.end_idx, \
            self.pad_idx, _, self.unk_idx = indices

    def forward(self, inputs, targets=None, max_len=50):
        """
        Runs the inputs through the encoder-decoder model.
        """
        batch_size = inputs.size(0)
        max_len = targets.size(1) if \
            targets is not None else max_len

        attn_mask = inputs.eq(self.pad_idx)

        # randomly masking input with unk during training
        # which makes the model more robust to unks during testing
        if self.training:
            unk_mask = inputs.new(
                inputs.size()).float().uniform_(0, 1) < 0.1
            inputs.masked_fill_(
                mask=unk_mask & inputs.ne(self.pad_idx), 
                value=self.unk_idx)

        encoder_outputs, encoder_hidden = self.encoder(inputs)
        
        hidden_state = repeat_hidden(
            encoder_hidden, self.decoder.rnn_layer.num_layers)

        scores = []
        preds = self.start_idx.detach().expand(batch_size, 1)
        
        for idx in range(max_len):
            # if targets are provided and training then
            # apply teacher forcing 50% of the time
            if targets is not None and random.random() > 0.5 and \
                    self.training:
                step_input = targets[:, idx].unsqueeze(1)
            else:
                step_input = preds[:, -1:]

            logits, hidden_state = self.decoder(
                inputs=step_input, 
                encoder_outputs=encoder_outputs,
                hidden_state=hidden_state,
                attn_mask=attn_mask)

            logits = logits[:, -1:, :]
            step_scores = log_softmax(logits, dim=-1)
            _, step_preds = step_scores.max(dim=-1)

            preds = torch.cat([preds, step_preds], 1)
            scores.append(step_scores)

        scores = torch.cat(scores, 1)
        preds = preds.narrow(1, 1, preds.size(1) - 1)
        preds = preds.contiguous()

        return scores, preds


class DropConnect(Module):
    
    def __init__(self):
        pass


class Encoder(Module):
    """
    Encoder module for the seq2seq model.
    """

    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()

        self.emb_layer = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size)

        self.dropout = Dropout(p=0.1)

        self.rnn_layer = LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
            num_layers=2)

    def forward(self, inputs):
        """
        Computes the embeddings and runs them through an LSTM.
        """
        embedded = self.emb_layer(inputs)
        embedded = self.dropout(embedded)
        encoder_outputs, hidden_states = self.rnn_layer(
            embedded)

        hidden_states = (
            hidden_states[0].sum(0).unsqueeze(0),
            hidden_states[1].sum(0).unsqueeze(0))

        return encoder_outputs, hidden_states


class Decoder(Module):
    """
    Decoder module for the seq2seq.
    """

    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()

        self.emb_layer = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size)

        self.dropout = Dropout(p=0.1)

        self.rnn_layer = LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.2,
            num_layers=2)

        self.attn_layer = Attention(hidden_size=hidden_size)

        # the final token in the vocab is the init token
        # which wont be the output of the model
        self.output_layer = Linear(
            in_features=hidden_size, 
            out_features=vocab_size - 1)

    def forward(self, inputs, encoder_outputs, hidden_state, 
                attn_mask=None):
        """
        Applies decoding with attention mechanism.
        """
        embedded = self.emb_layer(inputs)
        embedded = self.dropout(embedded)

        output, hidden_state = self.rnn_layer(
            embedded, hidden_state)
        
        output, _ = self.attn_layer(
            decoder_output=output, 
            last_hidden=hidden_state, 
            encoder_outputs=encoder_outputs,
            attn_mask=attn_mask)

        logits = self.output_layer(output)

        return logits, hidden_state


class Attention(Module):
    """
    Luong style general attention from:
    https://arxiv.org/pdf/1508.04025.pdf
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.attn_layer = Linear(
            in_features=hidden_size, 
            out_features=hidden_size * 2, 
            bias=False)

        self.merge_layer = Linear(
            in_features=hidden_size * 3,
            out_features=hidden_size,
            bias=False)

    def forward(self, decoder_output, last_hidden, encoder_outputs, 
                attn_mask=None):
        """
        Applies attention by creating the weighted context vector.
        Implementation is based on `facebookresearch ParlAI`.
        """
        # last_hidden is a tuple because of lstm, selecting the
        # the first tensor which is the real hidden state
        last_hidden = last_hidden[0][-1].unsqueeze(1)
        last_hidden = self.attn_layer(last_hidden)

        encoder_outputs_t = encoder_outputs.transpose(1, 2)
        attn_scores = torch.bmm(
            last_hidden, encoder_outputs_t).squeeze(1)

        # during beam search mask might not be provided
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, neginf())

        attn_weights = softmax(attn_scores.unsqueeze(1), dim=-1)
        attention_applied = torch.bmm(attn_weights, encoder_outputs)

        output = self.merge_layer(torch.cat(
            (decoder_output.squeeze(1), 
            attention_applied.squeeze(1)), dim=1).unsqueeze(1))

        return output, attn_weights
