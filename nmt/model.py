"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import torch
import random

from torch.nn.modules import Module

from torch.nn.functional import ( # pylint: disable=no-name-in-module, syntax-error
    log_softmax)

from torch.nn import (
    CrossEntropyLoss, LSTM, 
    Embedding, Linear,
    Softmax, Dropout)


def setup_model_args(parser):
    """"""
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=128,
        help='Hidden size of the model.')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=128,
        help='Embedding dimension for the tokens.')
    parser.add_argument(
        '--batch_first',
        type=bool,
        default=True,
        help='Set the batch dimension as first.')


def create_model(args, vocab_sizes, indices, device):
    """"""
    source_vocab_size, target_vocab_size = vocab_sizes

    start_index, end_index = indices
    start_index = torch.tensor(start_index) # pylint: disable=not-callable
    start_index.to(device)

    end_index = torch.tensor(end_index) # pylint: disable=not-callable
    end_index.to(device)

    model = Seq2Seq(
        source_vocab_size=source_vocab_size, 
        target_vocab_size=target_vocab_size,
        indices=(start_index, end_index),
        **vars(args))

    model.to(device)

    return model


def create_criterion(args, pad_index):
    """"""
    return CrossEntropyLoss(ignore_index=pad_index, reduction='sum')


class Seq2Seq(Module):
    """"""

    def __init__(self, embedding_size, 
                 hidden_size, batch_first, 
                 source_vocab_size, target_vocab_size, 
                 indices, **kwargs):
        super().__init__()

        self.encoder = Encoder(
            input_size=embedding_size,
            hidden_size=hidden_size,
            vocab_size=source_vocab_size,
            batch_first=batch_first)

        self.decoder = Decoder(
            input_size=embedding_size,
            hidden_size=hidden_size, 
            vocab_size=target_vocab_size,
            batch_first=batch_first)

        start_index, end_index = indices

        self.START_INDEX = start_index
        self.END_INDEX = end_index

    def forward(self, inputs, targets=None, max_len=50):
        """"""
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        if targets is None:
            scores, preds = self.decode_greedy(
                encoder_outputs, encoder_hidden, max_len)
        else:
            scores, preds = self.decode_forced(
                targets, encoder_outputs, encoder_hidden)

        return scores, preds

    def decode_greedy(self, encoder_outputs, encoder_hidden, max_len):
        """"""
        batch_size = encoder_outputs.size(0)
        preds = self.START_INDEX.detach().expand(batch_size, 1)
        scores = []

        hidden_state = encoder_hidden

        for _ in range(max_len):
            step_output, hidden_state = self.decoder(
                inputs=preds[:, -1:], 
                encoder_outputs=encoder_outputs,
                hidden_state=hidden_state)

            step_output = step_output[:, -1:, :]
            step_scores = log_softmax(step_output, dim=2)

            _, step_preds = step_scores.max(dim=2)

            preds = torch.cat( # pylint: disable=no-member
                [preds, step_preds], dim=1)

            scores.append(step_scores)
            all_finished = (
                (preds == self.END_INDEX)
                .sum(dim=1) > 0).sum().item() == batch_size
            if all_finished and not self.training:
                break
            
        scores = torch.cat(scores, 1) # pylint: disable=no-member
        preds = preds.narrow(1, 1, preds.size(1) - 1)
        preds = preds.contiguous()

        return scores, preds

    def decode_forced(self, targets, encoder_outputs, 
                      encoder_hidden):
        """"""
        logits, _ = self.decoder(
            targets, encoder_outputs, encoder_hidden)
        scores = log_softmax(logits, dim=2)
        _, preds = logits.max(dim=2)

        return scores, preds
    

class Encoder(Module):
    """"""

    def __init__(self, input_size, hidden_size, 
                 batch_first, vocab_size):
        super().__init__()

        self.embedding_layer = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size)

        self.recurrent_layer = LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=True,
            num_layers=1)

    def forward(self, inputs):
        """"""
        embedded_inputs = self.embedding_layer(inputs)
        encoder_outputs, hidden_states = self.recurrent_layer(
            embedded_inputs)

        hidden_states = (
            hidden_states[0].sum(0).unsqueeze(0),
            hidden_states[1].sum(0).unsqueeze(0))

        return encoder_outputs, hidden_states


class Decoder(Module):
    """"""

    def __init__(self, input_size, hidden_size, 
                 vocab_size, batch_first):
        super().__init__()

        self.embedding_layer = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size)

        self.recurrent_layer = LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            batch_first=batch_first,
            num_layers=1)

        self.output_layer = Linear(
            in_features=hidden_size, 
            out_features=vocab_size)

        self.attention_layer = Attention(
            hidden_size=hidden_size)
        
    def forward(self, inputs, encoder_outputs, hidden_state):
        """"""       
        outputs = []
        sequence_length = inputs.size(1)
        embedded_inputs = self.embedding_layer(inputs)

        for step in range(sequence_length):
            step_output, hidden_state = self.recurrent_layer(
                embedded_inputs[:, step, :].unsqueeze(1), 
                hidden_state)
            step_output, _ = self.attention_layer(
                decoder_output=step_output, 
                last_hidden=hidden_state, 
                encoder_outputs=encoder_outputs)

            outputs.append(step_output)

        outputs = torch.cat( # pylint: disable=no-member
            outputs, dim=1).to(inputs.device)

        outputs = self.output_layer(outputs)

        return outputs, hidden_state


class Attention(Module):
    """"""

    def __init__(self, hidden_size):
        super().__init__()

        self.attention_layer = Linear(
            in_features=hidden_size, 
            out_features=hidden_size * 2, 
            bias=False)

        self.combine_layer = Linear(
            in_features=hidden_size * 3,
            out_features=hidden_size)        

    def forward(self, decoder_output, last_hidden, encoder_outputs):
        """"""
        # Transforming hidden state to the correct dimension
        last_hidden = last_hidden[0][-1].unsqueeze(1)
        last_hidden = self.attention_layer(last_hidden)
        encoder_outputs_transpose = encoder_outputs.transpose(1, 2)

        attention_weights = torch.bmm( # pylint: disable=no-member
            last_hidden, 
            encoder_outputs_transpose).squeeze(1)
        attention_applied = torch.bmm( # pylint: disable=no-member
            attention_weights.unsqueeze(1), 
            encoder_outputs)

        merged = torch.cat( # pylint: disable=no-member
            (decoder_output.squeeze(1), 
            attention_applied.squeeze(1)), dim=1)
        output = torch.tanh( # pylint: disable=no-member
            self.combine_layer(merged).unsqueeze(1))

        return output, attention_weights
