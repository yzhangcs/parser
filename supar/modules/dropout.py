# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SharedDropout(nn.Module):
    """
    SharedDropout differs from the vanilla dropout strategy in that
    the dropout mask is shared across one dimension.

    Args:
        p (float, default:0.5):
            The probability of an element to be zeroed.
        batch_first (bool, default: True):
            If True, then the input and output tensors are provided as [batch_size, seq_len, *].

    Examples::
        >>> x = torch.randn(1, 3, 5)
        >>> x
        tensor([[[-0.2442,  0.5137, -0.4336, -1.4136, -0.5616],
                 [-0.7219, -2.8193, -1.0374, -0.2053,  1.6317],
                 [ 1.0911,  0.2259,  0.9608,  0.0606,  1.3651]]])
        >>> nn.Dropout()(x)
        tensor([[[-0.0000,  1.0274, -0.8673, -2.8272, -0.0000],
                 [-1.4439, -5.6386, -2.0748, -0.0000,  0.0000],
                 [ 2.1822,  0.0000,  1.9215,  0.0000,  2.7303]]])
        >>> SharedDropout()(x)
        tensor([[[-0.0000,  0.0000, -0.8673, -2.8272, -1.1233],
                 [-0.0000, -0.0000, -2.0748, -0.4106,  3.2633],
                 [ 0.0000,  0.0000,  1.9215,  0.1212,  2.7303]]])
    """

    def __init__(self, p=0.5, batch_first=True):
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def forward(self, x):
        """
        Args:
            x (Tensor):
                x can be of any shape.
        Returns:
            The returned tensor is of the same shape as x.
        """

        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
            else:
                mask = self.get_mask(x[0], self.p)
            x = x * mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask


class IndependentDropout(nn.Module):
    """
    For N tensors, they use different dropout masks respectively.
    When N-M of them are dropped, the remaining M ones are scaled by a factor of N/M to compensate,
    and when all of them are dropped together, zeros are returned.

    Args:
        p (float, default:0.5):
            The probability of an element to be zeroed.

    Examples::
        >>> x, y = torch.ones(1, 3, 5), torch.ones(1, 3, 5)
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

    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        """
        Args:
            items (List[Tensor]):
                A list of tensors that have the same shape except for the last dimension.
        Returns:
            The returned tensors are of the same shape as inputs.
        """

        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p) for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(-1) for item, mask in zip(items, masks)]

        return items
