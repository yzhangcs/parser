# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ScalarMix(nn.Module):
    '''
    Computes a parameterised scalar mixture of N tensors,
    ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

    If ``dropout > 0``, then for each scalar weight, adjust its
    softmax weight mass to 0 with the dropout probability (i.e.,
    setting the unnormalized weight to -inf). This effectively should
    redistribute dropped probability mass to all other weights.
    '''

    def __init__(self, n_layers, dropout=0):
        super().__init__()

        self.n_layers = n_layers

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
