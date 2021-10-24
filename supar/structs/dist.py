# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
from supar.structs.semiring import (CrossEntropySemiring, EntropySemiring,
                                    KLDivergenceSemiring, KMaxSemiring,
                                    LogSemiring, MaxSemiring)
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property


class StructuredDistribution(Distribution):
    r"""
    Base class for structured distribution :math:`p(y)` :cite:`eisner-2016-inside,goodman-1999-semiring,li-eisner-2009-first`.

    Args:
        scores (torch.Tensor):
            Log potentials, also for high-order cases.

    """

    def __init__(self, scores, **kwargs):
        self.scores = scores.requires_grad_() if isinstance(scores, torch.Tensor) else [s.requires_grad_() for s in scores]
        self.kwargs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @lazy_property
    def log_partition(self):
        r"""
        Compute the log partition function.
        """
        return self.forward(LogSemiring)

    @lazy_property
    def marginals(self):
        r"""
        Compute marginals of the distribution :math:`p(y)`.
        """
        return self.backward(self.log_partition.sum())

    @lazy_property
    def max(self):
        r"""
        Compute the max score of distribution :math:`p(y)`.
        """
        return self.forward(MaxSemiring)

    @lazy_property
    def argmax(self):
        r"""
        Compute :math:`\arg\max_y p(y)` of distribution :math:`p(y)`.
        """
        raise NotImplementedError

    @lazy_property
    def mode(self):
        return self.argmax

    def kmax(self, k):
        r"""
        Compute the k-max of distribution :math:`p(y)`.
        """
        return self.forward(KMaxSemiring(k))

    def topk(self, k):
        r"""
        Compute the k-argmax of distribution :math:`p(y)`.
        """
        raise NotImplementedError

    @lazy_property
    def entropy(self):
        r"""
        Compute entropy :math:`H[p]` of distribution :math:`p(y)`.
        """
        return self.forward(EntropySemiring)

    def cross_entropy(self, other):
        r"""
        Compute cross-entropy :math:`H[p,q]` of self and another distribution.

        Args:
            other (~supar.structs.dist.StructuredDistribution): Comparison distribution.
        """
        return (self + other).forward(CrossEntropySemiring)

    def kl(self, other):
        r"""
        Compute KL-divergence :math:`KL[p \parallel q]=H[p,q]-H[p]` of self and another distribution.

        Args:
            other (~supar.structs.dist.StructuredDistribution): Comparison distribution.
        """
        return (self + other).forward(KLDivergenceSemiring)

    def log_prob(self, value, **kwargs):
        """
        Computes log probability over values :math:`p(y)`.
        """
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
