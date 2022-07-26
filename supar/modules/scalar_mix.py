# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ScalarMix(nn.Module):
    r"""
    Computes a parameterized scalar mixture of :math:`N` tensors, :math:`mixture = \gamma * \sum_{k}(s_k * tensor_k)`
    where :math:`s = \mathrm{softmax}(w)`, with :math:`w` and :math:`\gamma` scalar parameters.

    Args:
        n_layers (int):
            The number of layers to be mixed, i.e., :math:`N`.
        dropout (float):
            The dropout ratio of the layer weights.
            If dropout > 0, then for each scalar weight, adjusts its softmax weight mass to 0
            with the dropout probability (i.e., setting the unnormalized weight to -inf).
            This effectively redistributes the dropped probability mass to all other weights.
            Default: 0.
    """

    def __init__(self, n_layers: int, dropout: float = .0) -> ScalarMix:
        super().__init__()

        self.n_layers = n_layers

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            tensors (List[~torch.Tensor]):
                :math:`N` tensors to be mixed.

        Returns:
            The mixture of :math:`N` tensors.
        """

        normed_weights = self.dropout(self.weights.softmax(-1))
        weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))

        return self.gamma * weighted_sum
