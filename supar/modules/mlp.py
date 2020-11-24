# -*- coding: utf-8 -*-

import torch.nn as nn
from supar.modules.dropout import SharedDropout


class MLP(nn.Module):
    r"""
    Applies a linear transformation together with a non-linear activation to the incoming tensor:
    :math:`y = \mathrm{Activation}(x A^T + b)`

    Args:
        n_in (~torch.Tensor):
            The size of each input feature.
        n_out (~torch.Tensor):
            The size of each output feature.
        dropout (float):
            If non-zero, introduce a :class:`SharedDropout` layer on the output with this dropout ratio. Default: 0.
        activation (bool):
            Whether to use activations. Default: True.
    """

    def __init__(self, n_in, n_out, dropout=0, activation=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor):
                The size of each input feature is `n_in`.

        Returns:
            A tensor with the size of each output feature `n_out`.
        """

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
