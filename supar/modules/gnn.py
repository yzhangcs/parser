# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class GraphConvolutionalNetwork(nn.Module):
    r"""
    Multiple GCN layers with layer normalization and residual connections, each executing the operator
    from the `"Semi-supervised Classification with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency matrix with inserted self-loops
    and :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        n_model (int):
            The size of node feature vectors.
        n_layers (int):
            The number of GCN layers. Default: 1.
        selfloop (bool):
            If ``True``, adds self-loops to adjacent matrices. Default: ``True``.
        norm (bool):
            If ``True``, adds a LayerNorm layer after each GCN layer. Default: ``True``.
    """

    def __init__(
        self,
        n_model: int,
        n_layers: int = 1,
        selfloop: bool = True,
        norm: bool = True
    ) -> GraphConvolutionalNetwork:
        super().__init__()

        self.n_model = n_model
        self.n_layers = n_layers
        self.selfloop = selfloop
        self.norm = norm

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                GraphConv(n_model),
                nn.LayerNorm([n_model]) if norm else nn.Identity()
            )
            for _ in range(n_layers)
        ])

    def __repr__(self):
        s = f"n_model={self.n_model}, n_layers={self.n_layers}"
        if self.selfloop:
            s += f", selfloop={self.selfloop}"
        if self.norm:
            s += f", norm={self.norm}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        r"""
        Args:
            x (~torch.Tensor):
                Node feature tensors of shape ``[batch_size, seq_len, n_model]``.
            adj (~torch.Tensor):
                Adjacent matrix of shape ``[batch_size, seq_len, seq_len]``.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            ~torch.Tensor:
                Node feature tensors of shape ``[batch_size, seq_len, n_model]``.
        """

        if self.selfloop:
            adj.diagonal(0, 1, 2).fill_(1.)
        adj = adj.masked_fill(~mask.unsqueeze(1), 0)
        for conv, norm in self.conv_layers:
            x = x + norm(conv(x, adj)).relu()
        return x


class GraphConv(nn.Module):

    def __init__(self, n_model: int, bias: bool = True) -> GraphConv:
        super().__init__()

        self.n_model = n_model

        self.linear = nn.Linear(n_model, n_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_model)) if bias else None

    def __repr__(self):
        s = f"n_model={self.n_model}"
        if self.bias is not None:
            s += ", bias=True"
        return f"{self.__class__.__name__}({s})"

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (~torch.Tensor):
                Node feature tensors of shape ``[batch_size, seq_len, n_model]``.
            adj (~torch.Tensor):
                Adjacent matrix of shape ``[batch_size, seq_len, seq_len]``.

        Returns:
            ~torch.Tensor:
                Node feature tensors of shape ``[batch_size, seq_len, n_model]``.
        """

        x = self.linear(x)
        d = adj.sum(-1)
        x = torch.matmul(adj * (d.unsqueeze(-1) * d.unsqueeze(2) + torch.finfo(adj.dtype).eps).pow(-0.5), x)
        if self.bias is not None:
            x = x + self.bias
        return x
