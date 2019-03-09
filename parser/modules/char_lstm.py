# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CharLSTM(nn.Module):

    def __init__(self, n_char, n_embed, n_out):
        super(CharLSTM, self).__init__()

        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=n_char,
                                  embedding_dim=n_embed)
        # the lstm layer
        self.lstm = nn.LSTM(input_size=n_embed,
                            hidden_size=n_out // 2,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        mask = x.gt(0)
        lens, indices = torch.sort(mask.sum(dim=1), descending=True)
        _, inverse_indices = indices.sort()
        max_len = lens[0]
        x = x[indices, :max_len]
        x = self.embed(x)

        x = pack_padded_sequence(x, lens, True)
        x, (hidden, _) = self.lstm(x)
        reprs = torch.cat(torch.unbind(hidden), dim=1)
        reprs = reprs[inverse_indices]

        return reprs
