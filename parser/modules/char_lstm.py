# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CHAR_LSTM(nn.Module):

    def __init__(self, n_chars, n_embed, n_out, pad_index=0):
        super(CHAR_LSTM, self).__init__()

        self.n_chars = n_chars
        self.n_embed = n_embed
        self.n_out = n_out
        self.pad_index = pad_index

        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=n_chars,
                                  embedding_dim=n_embed)
        # the lstm layer
        self.lstm = nn.LSTM(input_size=n_embed,
                            hidden_size=n_out//2,
                            batch_first=True,
                            bidirectional=True)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.n_chars}, {self.n_embed}, "
        s += f"n_out={self.n_out}, "
        s += f"pad_index={self.pad_index}"
        s += ')'

        return s

    def forward(self, x):
        mask = x.ne(self.pad_index)
        lens = mask.sum(dim=1)

        x = pack_padded_sequence(self.embed(x), lens, True, False)
        x, (hidden, _) = self.lstm(x)
        hidden = torch.cat(torch.unbind(hidden), dim=-1)

        return hidden
