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
    Linear, Softmax, Parameter, 
    GRU, Dropout, Embedding)

from torchnlp.nn import LockedDropout

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

    tensor_indices = [
        torch.tensor(i).to(device) for i in indices]

    model = Seq2Seq(
        source_vocab_size=len(SRC.vocab),
        target_vocab_size=len(TRG.vocab),
        indices=tensor_indices,
        **vars(args)).to(device)

    return model


def embeddeding_dropout(embed, inputs, training, mask=None, p=0.1):
    """
    Applies dropout to the embedding layer based on
    https://arxiv.org/pdf/1512.05287.pdf. The code is
    based on salesforce/awd-lstm-lm.
    """
    if not training:
        masked_embed_weight = embed.weight
    if mask is not None:
        # masks might be provided, which is useful for shared
        # dropout masks over the whole sequence of inputs
        masked_embed_weight = mask * embed.weight
    elif p:
        mask = embed.weight.new_empty((embed.weight.size(0), 1))
        mask.bernoulli_(1 - p).expand_as(embed.weight) / (1 - p)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    return embedding(
        inputs, masked_embed_weight, embed.padding_idx, 
        embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse)


class Seq2Seq(Module):
    """
    The sequence-to-sequence model.
    """

    def __init__(self, embedding_size, hidden_size, indices,
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
        batch_size = inputs.size(0)
        max_len = targets.size(1) if targets is not None else max_len

        attn_mask = inputs.eq(self.pad_idx)

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
        preds = self.start_idx.detach().expand(batch_size, 1)

        # precomputing embedding dropout mask for the decoder
        # NOTE embedding dropout is hardcoded 0.1
        embed = self.decoder.embedding
        embed_mask = embed.weight.new_empty((embed.weight.size(0), 1))
        embed_mask.bernoulli_(0.9).expand_as(embed.weight) / 0.9

        for idx in range(max_len):
            # if targets are provided and training then apply
            # teacher forcing 50% of the time
            if targets is not None and self.training and \
                    random.random() > 0.5:
                prev_output = targets[:, idx].unsqueeze(1)
            else:
                prev_output = preds[:, -1:]

            step_scores, hidden_states = self.decoder(
                inputs=prev_output,
                encoder_outputs=encoder_outputs,
                prev_hiddens=hidden_states,
                attn_mask=attn_mask,
                embed_mask=embed_mask)

            _, step_preds = step_scores.max(dim=-1)

            preds = torch.cat([preds, step_preds], dim=-1)
            scores.append(step_scores)

        scores = torch.cat(scores, dim=1)
        preds = preds.narrow(1, 1, preds.size(1) - 1)

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
            padding_idx=pad_idx)

        self.dropout = Dropout(p=0.3)

        self.merge = Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False)

        # creating rnn layer as module list so locked
        # dropout can be applied between each layer
        self.rnn = ModuleList([
            GRU(input_size=input_size,
                hidden_size=hidden_size,
                bidirectional=True,
                batch_first=True)] + [
            GRU(input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True)
            for _ in range(2)
        ])

    def forward(self, inputs):
        """
        Computes the embeddings and runs them through an RNN.
        """
        embedded = embeddeding_dropout(
            embed=self.embedding, inputs=inputs, 
            training=self.training)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        outputs, hidden_state = self.rnn[0](embedded)

        # merging the two directions of bidirectional layer
        # by summing along the first axis
        hidden_states = [hidden_state.sum(0, keepdim=True)]
        outputs = self.merge(outputs)

        for layer in self.rnn[1:]:
            outputs, hidden_state = layer(outputs)
            outputs = self.dropout(outputs)
            hidden_states.append(hidden_state)

        return outputs, hidden_states


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

        self.dropout = Dropout(p=0.3)

        self.rnn = ModuleList([
            GRU(input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True)] + [
            GRU(input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True)
            for _ in range(2)
        ])

        self.attn = Attention(hidden_size=hidden_size)

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
                attn_mask=None, embed_mask=None):
        """
        Applies decoding with attention mechanism, mixture
        of sofmaxes and multi dropout during training.
        MoS implementation is taken from 
        """
        embedded = embeddeding_dropout(
            embed=self.embedding, inputs=inputs, 
            training=self.training, mask=embed_mask)
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

        latent = torch.tanh(self.latent(output))
        latent = self.dropout(latent)
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
            inputs.size(0), 1, self.vocab_size)
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

    def forward(self, decoder_output, hidden_state, 
                encoder_outputs, attn_mask=None):
        """
        Applies attention by creating the weighted 
        context vector. Implementation is based on 
        `IBM/pytorch-seq2seq`.
        """
        hidden_state = self.project(hidden_state)
        hidden_state = hidden_state.transpose(0, 1)
        
        encoder_outputs_t = encoder_outputs.transpose(1, 2)
        attn_scores = torch.bmm(
            hidden_state, encoder_outputs_t)

        # applying mask on padded values of the input
        # NOTE during beam search mask might not be provided
        if attn_mask is not None:
            attn_scores = attn_scores.squeeze(1)
            attn_scores.masked_fill_(attn_mask, -inf)
            attn_scores = attn_scores.unsqueeze(1)

        attn_weights = softmax(attn_scores, dim=-1)
        attn_applied = torch.bmm(
            attn_weights, encoder_outputs)

        stacked = torch.cat(
            [decoder_output, attn_applied], dim=-1)
        outputs = self.combine(stacked)

        return outputs, attn_weights
