# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SharedDropout(nn.Module):
    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        info = f"p={self.p}"
        if self.batch_first:
            info += f", batch_first={self.batch_first}"

        return info

    def forward(self, x):
        if self.training:
            if self.batch_first:
                batch_size, seq_len = x.shape[:2]
            else:
                seq_len, batch_size = x.shape[:2]
            mask = x.new_full((batch_size, *x.shape[2:]), 1 - self.p)
            mask = torch.bernoulli(mask) / (1 - self.p)
            x = x * mask.unsqueeze(1) if self.batch_first else x * mask

        return x


class IndependentDropout(nn.Module):
    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, x, y):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            y_mask = torch.bernoulli(y.new_full(y.shape[:2], 1 - self.p))
            scale = 3.0 / (2.0 * x_mask + y_mask + 1e-12)
            x_mask *= scale
            y_mask *= scale
            x = x * x_mask.unsqueeze(dim=2)
            y = y * y_mask.unsqueeze(dim=2)

        return x, y
