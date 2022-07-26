# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class TokenDropout(nn.Module):
    r"""
    :class:`TokenDropout` seeks to randomly zero the vectors of some tokens with the probability of `p`.

    Args:
        p (float):
            The probability of an element to be zeroed. Default: 0.5.

    Examples:
        >>> batch_size, seq_len, hidden_size = 1, 3, 5
        >>> x = torch.ones(batch_size, seq_len, hidden_size)
        >>> nn.Dropout()(x)
        tensor([[[0., 2., 2., 0., 0.],
                 [2., 2., 0., 2., 2.],
                 [2., 2., 2., 2., 0.]]])
        >>> TokenDropout()(x)
        tensor([[[2., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 0.],
                 [2., 2., 2., 2., 2.]]])
    """

    def __init__(self, p: float = 0.5) -> TokenDropout:
        super().__init__()

        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (~torch.Tensor):
                A tensor of any shape.
        Returns:
            A tensor with the same shape as `x`.
        """

        if not self.training:
            return x
        return x * (x.new_empty(x.shape[:2]).bernoulli_(1 - self.p) / (1 - self.p)).unsqueeze(-1)


class SharedDropout(nn.Module):
    r"""
    :class:`SharedDropout` differs from the vanilla dropout strategy in that the dropout mask is shared across one dimension.

    Args:
        p (float):
            The probability of an element to be zeroed. Default: 0.5.
        batch_first (bool):
            If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
            Default: ``True``.

    Examples:
        >>> batch_size, seq_len, hidden_size = 1, 3, 5
        >>> x = torch.ones(batch_size, seq_len, hidden_size)
        >>> nn.Dropout()(x)
        tensor([[[0., 2., 2., 0., 0.],
                 [2., 2., 0., 2., 2.],
                 [2., 2., 2., 2., 0.]]])
        >>> SharedDropout()(x)
        tensor([[[2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.]]])
    """

    def __init__(self, p: float = 0.5, batch_first: bool = True) -> SharedDropout:
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def __repr__(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (~torch.Tensor):
                A tensor of any shape.
        Returns:
            A tensor with the same shape as `x`.
        """

        if not self.training:
            return x
        return x * self.get_mask(x[:, 0], self.p).unsqueeze(1) if self.batch_first else self.get_mask(x[0], self.p)

    @staticmethod
    def get_mask(x: torch.Tensor, p: float) -> torch.FloatTensor:
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)


class IndependentDropout(nn.Module):
    r"""
    For :math:`N` tensors, they use different dropout masks respectively.
    When :math:`N-M` of them are dropped, the remaining :math:`M` ones are scaled by a factor of :math:`N/M` to compensate,
    and when all of them are dropped together, zeros are returned.

    Args:
        p (float):
            The probability of an element to be zeroed. Default: 0.5.

    Examples:
        >>> batch_size, seq_len, hidden_size = 1, 3, 5
        >>> x, y = torch.ones(batch_size, seq_len, hidden_size), torch.ones(batch_size, seq_len, hidden_size)
        >>> x, y = IndependentDropout()(x, y)
        >>> x
        tensor([[[1., 1., 1., 1., 1.],
                 [0., 0., 0., 0., 0.],
                 [2., 2., 2., 2., 2.]]])
        >>> y
        tensor([[[1., 1., 1., 1., 1.],
                 [2., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 0.]]])
    """

    def __init__(self, p: float = 0.5) -> IndependentDropout:
        super().__init__()

        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, *items: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""
        Args:
            items (List[~torch.Tensor]):
                A list of tensors that have the same shape except the last dimension.
        Returns:
            A tensors are of the same shape as `items`.
        """

        if not self.training:
            return items
        masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p) for x in items]
        total = sum(masks)
        scale = len(items) / total.max(torch.ones_like(total))
        masks = [mask * scale for mask in masks]
        return [item * mask.unsqueeze(-1) for item, mask in zip(items, masks)]
