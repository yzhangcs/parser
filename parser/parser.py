# -*- coding: utf-8 -*-

from parser.modules import MLP, Biaffine, CharLSTM, ParserLSTM
from parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class BiaffineParser(nn.Module):
    def __init__(self, vocab,
                 n_embed, n_char_embed, n_char_out,
                 n_lstm_hidden, n_lstm_layers, n_mlp_arc, n_mlp_lab, dropout):
        super(BiaffineParser, self).__init__()

        self.vocab = vocab
        # the embedding layer
        self.embed = nn.Embedding(vocab.n_train_words, n_embed)
        self.pretrained = nn.Embedding.from_pretrained(vocab.embeddings)
        # the char-lstm layer
        self.char_lstm = CharLSTM(n_char=vocab.n_chars,
                                  n_embed=n_char_embed,
                                  n_out=n_char_out)
        self.embed_drop = IndependentDropout(p=dropout)

        # the word-lstm layer
        self.lstm = ParserLSTM(input_size=n_embed+n_char_out,
                               hidden_size=n_lstm_hidden,
                               num_layers=n_lstm_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=True)
        self.lstm_drop = SharedDropout(p=dropout)

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=n_lstm_hidden*2,
                             n_hidden=n_mlp_arc,
                             dropout=dropout)
        self.mlp_arc_d = MLP(n_in=n_lstm_hidden*2,
                             n_hidden=n_mlp_arc,
                             dropout=dropout)
        self.mlp_lab_h = MLP(n_in=n_lstm_hidden*2,
                             n_hidden=n_mlp_lab,
                             dropout=dropout)
        self.mlp_lab_d = MLP(n_in=n_lstm_hidden*2,
                             n_hidden=n_mlp_lab,
                             dropout=dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.lab_attn = Biaffine(n_in=n_mlp_lab,
                                 n_out=vocab.n_labels,
                                 bias_x=True,
                                 bias_y=True)

        self.reset_parameters()

    def reset_parameters(self):
        bias = (3. / self.embed.weight.size(1)) ** 0.5
        nn.init.uniform_(self.embed.weight, -bias, bias)

    def forward(self, x, char_x):
        # get the mask and lengths of given batch
        mask = x.gt(0)
        lens = mask.sum(dim=1)
        # get outputs from embedding layers
        embed_x = self.pretrained(x)
        embed_x += self.embed(x.masked_fill_(x >= self.vocab.n_train_words,
                                             self.vocab.unk_index))

        char_x = self.char_lstm(char_x[mask])
        char_x = pad_sequence(torch.split(char_x, lens.tolist()), True)
        embed_x, char_x = self.embed_drop(embed_x, char_x)
        # concatenate the word and char representations
        x = torch.cat((embed_x, char_x), dim=-1)

        x = self.lstm(x, mask)
        x = self.lstm_drop(x)

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

    def get_loss(self, s_arc, s_lab, heads, labels, mask):
        s_arc = s_arc[mask]
        s_lab = s_lab[mask]
        heads = heads[mask]
        labels = labels[mask]
        s_lab = s_lab[torch.arange(len(s_arc)), heads]

        arc_loss = F.cross_entropy(s_arc, heads)
        lab_loss = F.cross_entropy(s_lab, labels)
        loss = arc_loss + lab_loss

        return loss
