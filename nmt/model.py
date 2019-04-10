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
    log_softmax, tanh)

from torch.nn import (
    CrossEntropyLoss, LSTM, 
    Embedding, Linear,
    Softmax, Dropout)


def setup_model_args(parser):
    """"""
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Path of the model checkpoints.')
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=128,
        help='Hidden size of the model.')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=300,
        help='Embedding dimension for the tokens.')
    parser.add_argument(
        '--batch_first',
        type=bool,
        default=True,
        help='Set the batch dimension as first.')


def create_model(args, vocab_sizes):
    """"""
    source_vocab_size, target_vocab_size = vocab_sizes

    return Seq2Seq(
        source_vocab_size=source_vocab_size, 
        target_vocab_size=target_vocab_size,
        **vars(args))


def create_criterion(args):
    """"""
    return CrossEntropyLoss(size_average=False)


class Seq2Seq(Module):
    """"""

    def __init__(self, embedding_size, 
                 hidden_size, batch_first, 
                 source_vocab_size, target_vocab_size, 
                 **kwargs):
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

    def forward(self, inputs, targets=None):
        """"""
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        if targets is None:
            scores, preds = self.decode_greedy(
                encoder_outputs, encoder_hidden)
        
        else:
            scores, preds = self.decode_forced(
                targets, encoder_outputs, encoder_hidden)

        return scores, preds

    def decode_greedy(self, encoder_outputs, max_len=40):
        """"""
        for _ in range(max_len):
            pass

        return 0

    def decode_forced(self, targets, encoder_outputs, 
                      encoder_hidden):
        """"""
        latent, _ = self.decoder(
            targets, encoder_outputs, encoder_hidden)
        logits = log_softmax(latent)
        _, preds = logits.max(dim=2)  

        return logits, preds
    

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
        
    def forward(self, inputs, encoder_outputs, encoder_hidden,
                previous_hidden_state=None):
        """"""
        seq_len = inputs.size(1)

        if previous_hidden_state is None:
            hidden_state = encoder_hidden

        else:
            hidden_state = previous_hidden_state

        output = []
        embedded_inputs = self.embedding_layer(inputs)

        for step in range(seq_len):
            step_output, hidden_state = self.recurrent_layer(
                embedded_inputs[:, step, :].unsqueeze(1), 
                hidden_state)
            step_output = self.attention_layer(
                inputs=step_output, 
                hidden_state=hidden_state, 
                encoder_outputs=encoder_outputs)

            output.append(step_output)

        output = torch.cat( # pylint: disable=no-member
            output, dim=1).to(inputs.device)

        return output, hidden_state


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

    def forward(self, inputs, hidden_state, encoder_outputs):
        """"""
        # Transforming hidden state to the correct dimension
        hidden_state = hidden_state[0][-1].unsqueeze(1)
        hidden_state = self.attention_layer(hidden_state)

        encoder_outputs_transpose = encoder_outputs.transpose(1, 2)
        attention_weights = torch.bmm( # pylint: disable=no-member
            hidden_state, 
            encoder_outputs_transpose).squeeze(1)

        attention_applied = torch.bmm( # pylint: disable=no-member
            attention_weights.unsqueeze(1), 
            encoder_outputs)

        merged = torch.cat( # pylint: disable=no-member
            (inputs.squeeze(1), 
            attention_applied.squeeze(1)), 1)

        output = tanh(self.combine_layer(merged).unsqueeze(1))

        return output
