"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import torch

from torch.nn.modules import Module

from torch.nn import (
    CrossEntropyLoss, 
    LSTM, 
    Embedding,
    Linear)

from torch.optim import Adam


def setup_model_args(parser):
    """"""
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=128,
        help='Hidden size of the model.')
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=300,
        help='Embedding dimension for the tokens.')


def greedy_decode(model, criterion, batch):
    """"""
    return batch


def create_model(args, output_size):
    """"""
    return Seq2Seq(**vars(args), output_size=output_size)


def create_criterion(args):
    """"""
    return CrossEntropyLoss()


def create_optimizer(args, parameters):
    """"""
    return Adam(parameters)
    

class Seq2Seq(Module):
    """"""

    def __init__(self, embedding_size, 
                 hidden_size, output_size, **kwargs):
        super().__init__()

        self.embedding_layer = Embedding(
            num_embeddings=output_size,
            embedding_dim=embedding_size)

        self.encoder = EncoderRNN(hidden_size=hidden_size)

        self.decoder = DecoderRNN(
            hidden_size=hidden_size, 
            output_size=output_size)

    def forward(self, inputs):
        """"""
        embedded = self.embedding_layer(inputs)
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs)

        return encoder_outputs, decoder_outputs


class EncoderRNN(Module):
    """"""

    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.rnn = LSTM(hidden_size=hidden_size)

    def forward(self, inputs):
        """"""
        outputs = self.rnn(inputs) 

        return outputs


class DecoderRNN(Module):
    """"""

    def __init__(self, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = LSTM(hidden_size=hidden_size)
        self.output_layer = Linear(
            in_features=hidden_size, 
            out_features=output_size)

    def forward(self, inputs):
        """"""
        rnn_outputs = self.rnn(inputs) 
        outputs = self.output_layer(rnn_outputs)

        return outputs
