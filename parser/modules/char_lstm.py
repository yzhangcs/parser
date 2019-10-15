# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CHAR_LSTM(nn.Module):

    def __init__(self, n_chars, n_embed, n_out):
        super(CHAR_LSTM, self).__init__()

        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=n_chars,
                                  embedding_dim=n_embed)
        # the lstm layer
        self.lstm = nn.LSTM(input_size=n_embed,
                            hidden_size=n_out//2,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        mask = x.gt(0)
        lens = mask.sum(dim=1)

        x = pack_padded_sequence(self.embed(x), lens, True, False)
        x, (hidden, _) = self.lstm(x)
        hidden = torch.cat(torch.unbind(hidden), dim=-1)

        return hidden
