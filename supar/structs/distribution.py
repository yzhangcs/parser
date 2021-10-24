# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
from supar.structs.semiring import (CrossEntropySemiring, EntropySemiring,
                                    KLDivergenceSemiring, KMaxSemiring,
                                    LogSemiring, MaxSemiring)
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property


class StructuredDistribution(Distribution):

    def __init__(self, scores, **kwargs):
        self.scores = scores.requires_grad_() if isinstance(scores, torch.Tensor) else [s.requires_grad_() for s in scores]
        self.kwargs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @lazy_property
    def log_partition(self):
        return self.forward(LogSemiring)

    @lazy_property
    def marginals(self):
        return self.backward(self.log_partition.sum())

    @lazy_property
    def max(self):
        return self.forward(MaxSemiring)

    @lazy_property
    def argmax(self):
        raise NotImplementedError

    @lazy_property
    def mode(self):
        return self.argmax

    def kmax(self, k):
        return self.forward(KMaxSemiring(k))

    def topk(self, k):
        raise NotImplementedError

    @lazy_property
    def entropy(self):
        return self.forward(EntropySemiring)

    def cross_entropy(self, other):
        return (self + other).forward(CrossEntropySemiring)

    def kl(self, other):
        return (self + other).forward(KLDivergenceSemiring)

    def log_prob(self, value, **kwargs):
        return self.score(value, **kwargs) - self.log_partition

    def score(self, value):
        raise NotImplementedError

    @torch.enable_grad()
    def forward(self, semiring):
        raise NotImplementedError

    def backward(self, log_partition):
        return autograd.grad(log_partition,
                             self.scores if isinstance(self.scores, torch.Tensor) else self.scores[0],
                             retain_graph=True)[0]
