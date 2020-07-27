# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Triaffine(nn.Module):
    """
    Triaffine layer for second-order scoring.

    This function has a tensor of weights `W` and bias terms if needed.
    The score `s(x, y, z)` of the vector triple `(x, y, z)` is computed as `x^T z^T W y`.
    Usually, `x` and `y` can be concatenated with bias terms.

    References:
        - Yu Zhang, Zhenghua Li and Min Zhang (ACL'20)
          Efficient Second-Order TreeCRF for Neural Dependency Parsing
          https://www.aclweb.org/anthology/2020.acl-main.302/
        - Xinyu Wang, Jingxian Huang, and Kewei Tu (ACL'19)
          Second-Order Semantic Dependency Parsing with End-to-End Neural Networks
          https://www.aclweb.org/anthology/P19-1454/

    Args:
        n_in (int):
            The dimension of the input feature.
        bias_x (bool):
            If True, add a bias term for tensor x. Default: False.
        bias_y (bool):
            If True, add a bias term for tensor y. Default: False.
    """

    def __init__(self, n_in, bias_x=False, bias_y=False):
        super().__init__()

        self.n_in = n_in
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_in + bias_x,
                                                n_in,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y, z):
        """
        Args:
            x (Tensor): [batch_size, seq_len, n_in]
            y (Tensor): [batch_size, seq_len, n_in]
            z (Tensor): [batch_size, seq_len, n_in]

        Returns:
            s (Tensor): [batch_size, seq_len, seq_len, seq_len]
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        w = torch.einsum('bzk,ikj->bzij', z, self.weight)
        # [batch_size, seq_len, seq_len, seq_len]
        s = torch.einsum('bxi,bzij,byj->bzxy', x, w, y)

        return s
