# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Biaffine(nn.Module):
    """
    Biaffine layer for first-order scoring.

    This function has a tensor of weights `W` and bias terms if needed.
    The score `s(x, y)` of the vector pair `(x, y)` is computed as `x^T W y`,
    in which `x` and `y` can be concatenated with bias terms.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    Args:
        n_in (int):
            The dimension of the input feature.
        n_out (int):
            The number of output channels.
        bias_x (bool):
            If ``True``, add a bias term for tensor x. Default: ``False``.
        bias_y (bool):
            If ``True``, add a bias term for tensor y. Default: ``False``.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/pdf?id=Hk95PK9le
    """

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            s (torch.Tensor): ``[batch_size, n_out, seq_len, seq_len]``.
                If n_out is 1, the dimension of n_out will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s
