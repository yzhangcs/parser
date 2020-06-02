# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ScalarMix(nn.Module):

    def __init__(self, n_layers, dropout=0):
        super(ScalarMix, self).__init__()

        self.n_layers = n_layers
        self.dropout = dropout

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.dropout = nn.Dropout(dropout)

    def extra_repr(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return s

    def forward(self, tensors):
        normed_weights = self.dropout(self.weights.softmax(-1))
        weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))

        return self.gamma * weighted_sum
