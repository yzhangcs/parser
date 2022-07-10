# -*- coding: utf-8 -*-

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class InverseSquareRootLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ) -> InverseSquareRootLR:
        self.warmup_steps = warmup_steps
        self.factor = warmup_steps ** 0.5
        super(InverseSquareRootLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = max(self.last_epoch, 1)
        scale = min(epoch ** -0.5, epoch * self.warmup_steps ** -1.5) * self.factor
        return [scale * lr for lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    r"""
    Set the learning rate for each parameter group using a polynomial defined as: `lr = base_lr * (1 - t / T) ^ (power)`,
    where `t` is the current epoch and `T` is the maximum number of epochs.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        steps: int = 100000,
        power: float = 1.,
        last_epoch: int = -1
    ) -> PolynomialLR:
        self.warmup_steps = warmup_steps
        self.steps = steps
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = max(self.last_epoch, 1)
        if epoch <= self.warmup_steps:
            return [epoch / self.warmup_steps * lr for lr in self.base_lrs]
        t, T = (epoch - self.warmup_steps), (self.steps - self.warmup_steps)
        return [lr * (1 - t / T) ** self.power for lr in self.base_lrs]


def LinearLR(optimizer: Optimizer, warmup_steps: int = 0, steps: int = 100000, last_epoch: int = -1) -> PolynomialLR:
    return PolynomialLR(optimizer, warmup_steps, steps, 1, last_epoch)
