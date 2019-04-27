# -*- coding: utf-8 -*-

from parser.modules import LSTM, MLP, Biaffine
from parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiaffineParser(nn.Module):

    def __init__(self, params, embeddings):
        super(BiaffineParser, self).__init__()

        self.params = params
        # the embedding layer
        self.pretrained = nn.Embedding.from_pretrained(embeddings)
        self.embed = nn.Embedding(num_embeddings=params['n_words'],
                                  embedding_dim=params['n_embed'])
        self.tag_embed = nn.Embedding(num_embeddings=params['n_tags'],
                                      embedding_dim=params['n_tag_embed'])
        self.embed_dropout = IndependentDropout(p=params['embed_dropout'])

        # the word-lstm layer
        self.lstm = LSTM(input_size=params['n_embed']+params['n_tag_embed'],
                         hidden_size=params['n_lstm_hidden'],
                         num_layers=params['n_lstm_layers'],
                         dropout=params['lstm_dropout'],
                         bidirectional=True)
        self.lstm_dropout = SharedDropout(p=params['lstm_dropout'])

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_arc'],
                             dropout=params['mlp_dropout'])
        self.mlp_arc_d = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_arc'],
                             dropout=params['mlp_dropout'])
        self.mlp_rel_h = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_rel'],
                             dropout=params['mlp_dropout'])
        self.mlp_rel_d = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_rel'],
                             dropout=params['mlp_dropout'])

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=params['n_mlp_arc'],
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=params['n_mlp_rel'],
                                 n_out=params['n_rels'],
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = params['pad_index']
        self.unk_index = params['unk_index']

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.embed.weight)

    def forward(self, words, tags):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # get outputs from embedding layers
        embed = self.pretrained(words)
        embed += self.embed(
            words.masked_fill_(words.ge(self.embed.num_embeddings),
                               self.unk_index)
        )
        tag_embed = self.tag_embed(tags)
        embed, tag_embed = self.embed_dropout(embed, tag_embed)
        # concatenate the word and tag representations
        x = torch.cat((embed, tag_embed), dim=-1)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_dropout(x)[inverse_indices]

        # apply MLPs to the LSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_((1 - mask).unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    @classmethod
    def load(cls, fname):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(fname, map_location=device)
        network = cls(state['params'], state['embeddings'])
        network.load_state_dict(state['state_dict'])
        network.to(device)

        return network

    def save(self, fname):
        state = {
            'params': self.params,
            'embeddings': self.pretrained.weight,
            'state_dict': self.state_dict(),
        }
        torch.save(state, fname)
