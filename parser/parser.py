# -*- coding: utf-8 -*-

from parser.modules import CHAR_LSTM, LSTM, MLP, Biaffine
from parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class BiaffineParser(nn.Module):

    def __init__(self, vocab, params):
        super(BiaffineParser, self).__init__()

        self.vocab = vocab
        self.params = params
        # the embedding layer
        self.embed = nn.Embedding(vocab.n_train_words, params['n_embed'])
        self.pretrained = nn.Embedding.from_pretrained(vocab.embeddings)
        # the char-lstm layer
        self.char_lstm = CHAR_LSTM(n_char=vocab.n_chars,
                                   n_embed=params['n_char_embed'],
                                   n_out=params['n_char_out'])
        self.embed_drop = IndependentDropout(p=params['dropout'])

        # the word-lstm layer
        self.lstm = LSTM(input_size=params['n_embed']+params['n_char_out'],
                         hidden_size=params['n_lstm_hidden'],
                         num_layers=params['n_lstm_layers'],
                         dropout=params['dropout'],
                         bidirectional=True)
        self.lstm_drop = SharedDropout(p=params['dropout'])

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_arc'],
                             dropout=params['dropout'])
        self.mlp_arc_d = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_arc'],
                             dropout=params['dropout'])
        self.mlp_lab_h = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_lab'],
                             dropout=params['dropout'])
        self.mlp_lab_d = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_lab'],
                             dropout=params['dropout'])

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=params['n_mlp_arc'],
                                 bias_x=True,
                                 bias_y=False)
        self.lab_attn = Biaffine(n_in=params['n_mlp_lab'],
                                 n_out=vocab.n_labels,
                                 bias_x=True,
                                 bias_y=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.embed.weight)

    def forward(self, words, chars):
        # get the mask and lengths of given batch
        mask = words.gt(0)
        lens = mask.sum(dim=1)
        # get outputs from embedding layers
        word_embed = self.pretrained(words)
        word_embed += self.embed(
            words.masked_fill_(words.ge(self.vocab.n_train_words),
                               self.vocab.unk_index)
        )
        char_embed = self.char_lstm(chars[mask])
        char_embed = pad_sequence(torch.split(char_embed, lens.tolist()), True)
        word_embed, char_embed = self.embed_drop(word_embed, char_embed)
        # concatenate the word and char representations
        x = torch.cat((word_embed, char_embed), dim=-1)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_drop(x)[inverse_indices]

        # apply MLPs to the LSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        lab_h = self.mlp_lab_h(x)
        lab_d = self.mlp_lab_d(x)

        # get arc and label scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_labels]
        s_lab = self.lab_attn(lab_d, lab_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_((1 - mask).unsqueeze(1), float('-inf'))

        return s_arc, s_lab

    @classmethod
    def load(cls, fname):
        state = torch.load(fname)
        parser = cls(state['vocab'], state['params'])
        parser.load_state_dict(state['state_dict'])
        if torch.cuda.is_available():
            parser = parser.cuda()

        return parser

    def save(self, fname):
        state = {
            'vocab': self.vocab,
            'params': self.params,
            'state_dict': self.state_dict(),
        }
        torch.save(state, fname)
