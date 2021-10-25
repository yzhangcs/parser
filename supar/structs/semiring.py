# -*- coding: utf-8 -*-

from functools import reduce

import torch
from supar.utils.common import MIN


class Semiring(object):
    r"""
    Base semiring class :cite:`goodman-1999-semiring`.

    A semiring is defined by a tuple :math:`<K, \oplus, \otimes, \mathbf{0}, \mathbf{1}>`.
    :math:`K` is a set of values;
    :math:`\oplus` is commutative, associative and has an identity element `0`;
    :math:`\otimes` is associative, has an identity element `1` and distributes over `+`.
    """

    zero = 0
    one = 1

    @classmethod
    def sum(cls, x, dim=-1):
        return x.sum(dim)

    @classmethod
    def mul(cls, x, y):
        return x * y

    @classmethod
    def dot(cls, x, y, dim=-1):
        return cls.sum(cls.mul(x, y), dim)

    @classmethod
    def prod(cls, x, dim=-1):
        return x.prod(dim)

    @classmethod
    def times(cls, *x):
        return reduce(lambda i, j: cls.mul(i, j), x)

    @classmethod
    def zero_(cls, x):
        return x.fill_(cls.zero)

    @classmethod
    def one_(cls, x):
        return x.fill_(cls.one)

    @classmethod
    def zero_mask(cls, x, mask):
        return x.masked_fill(mask, cls.zero)

    @classmethod
    def zero_mask_(cls, x, mask):
        return x.masked_fill_(mask, cls.zero)

    @classmethod
    def one_mask(cls, x, mask):
        return x.masked_fill(mask, cls.one)

    @classmethod
    def one_mask_(cls, x, mask):
        return x.masked_fill_(mask, cls.one)

    @classmethod
    def zeros_like(cls, x):
        return torch.full_like(x, cls.zero)

    @classmethod
    def ones_like(cls, x):
        return torch.full_like(x, cls.one)

    @classmethod
    def convert(cls, x):
        return x

    @classmethod
    def unconvert(cls, x):
        return x


class LogSemiring(Semiring):
    r"""
    Log-space semiring :math:`<\mathrm{logsumexp}, +, -\infty, 0>`.
    """

    zero = MIN
    one = 0

    @classmethod
    def sum(cls, x, dim=-1):
        return x.logsumexp(dim)

    @classmethod
    def mul(cls, x, y):
        return x + y

    @classmethod
    def prod(cls, x, dim=-1):
        return x.sum(dim)


class MaxSemiring(LogSemiring):
    r"""
    Max semiring :math:`<\mathrm{max}, +, -\infty, 0>`.
    """

    @classmethod
    def sum(cls, x, dim=-1):
        return x.max(dim)[0]


def KMaxSemiring(k):
    r"""
    k-max semiring :math:`<\mathrm{kmax}, +, [-\infty, -\infty, \dots], [0, -\infty, \dots]>`.
    """

    class KMaxSemiring(LogSemiring):

        @classmethod
        def convert(cls, x):
            return torch.cat((x.unsqueeze(-1), cls.zero_(x.new_empty(*x.shape, k - 1))), -1)

        @classmethod
        def sum(cls, x, dim=-1):
            return x.movedim(dim, -1).flatten(-2).topk(k, -1)[0]

        @classmethod
        def mul(cls, x, y):
            return (x.unsqueeze(-1) + y.unsqueeze(-2)).flatten(-2).topk(k, -1)[0]

        @classmethod
        def one_(cls, x):
            x[..., :1].fill_(cls.one)
            x[..., 1:].fill_(cls.zero)
            return x

    return KMaxSemiring


class EntropySemiring(LogSemiring):
    """
    Entropy expectation semiring: :math:`<\oplus, +, [-\infty, 0], [0, 0]>`,
    where :math:`\oplus` computes the log-values and the running distributional entropy :math:`H[p]`
    :cite:`li-eisner-2009-first,hwa-2000-sample,kim-etal-2019-unsupervised`.
    """

    @classmethod
    def convert(cls, x):
        return torch.stack((x, cls.ones_like(x)), -1)

    @classmethod
    def unconvert(cls, x):
        return x[..., -1]

    @classmethod
    def sum(cls, x, dim=-1):
        p = x[..., 0].logsumexp(dim)
        r = x[..., 0] - p.unsqueeze(dim)
        r = r.exp().mul((x[..., -1] - r)).sum(dim)
        return torch.stack((p, r), -1)

    @classmethod
    def mul(cls, x, y):
        return x + y

    @classmethod
    def zero_(cls, x):
        x[..., :-1].fill_(cls.zero)
        x[..., -1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x):
        return x.fill_(cls.one)


class CrossEntropySemiring(LogSemiring):
    """
    Cross Entropy expectation semiring: :math:`<\oplus, +, [-\infty, -\infty, 0], [0, 0, 0]>`,
    where :math:`\oplus` computes the log-values and the running distributional cross entropy :math:`H[p,q]`
    of the two distributions :cite:`li-eisner-2009-first`.
    """

    @classmethod
    def convert(cls, x):
        return torch.cat((x, cls.one_(torch.empty_like(x[..., :1]))), -1)

    @classmethod
    def unconvert(cls, x):
        return x[..., -1]

    @classmethod
    def sum(cls, x, dim=-1):
        p = x[..., :-1].logsumexp(dim)
        r = x[..., :-1] - p.unsqueeze(dim)
        r = r[..., 0].exp().mul((x[..., -1] - r[..., 1])).sum(dim)
        return torch.cat((p, r.unsqueeze(-1)), -1)

    @classmethod
    def mul(cls, x, y):
        return x + y

    @classmethod
    def zero_(cls, x):
        x[..., :-1].fill_(cls.zero)
        x[..., -1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x):
        return x.fill_(cls.one)


class KLDivergenceSemiring(LogSemiring):
    """
    KL divergence expectation semiring: :math:`<\oplus, +, [-\infty, -\infty, 0], [0, 0, 0]>`,
    where :math:`\oplus` computes the log-values and the running distributional KL divergence :math:`KL[p \parallel q]`
    of the two distributions :cite:`li-eisner-2009-first`.
    """
    """
    KL divergence expectation semiring: `<logsumexp, +, -inf, 0>` :cite:`li-eisner-2009-first`.
    """

    @classmethod
    def convert(cls, x):
        return torch.cat((x, cls.one_(torch.empty_like(x[..., :1]))), -1)

    @classmethod
    def unconvert(cls, x):
        return x[..., -1]

    @classmethod
    def sum(cls, x, dim=-1):
        p = x[..., :-1].logsumexp(dim)
        r = x[..., :-1] - p.unsqueeze(dim)
        r = r[..., 0].exp().mul((x[..., -1] - r[..., 1] + r[..., 0])).sum(dim)
        return torch.cat((p, r.unsqueeze(-1)), -1)

    @classmethod
    def mul(cls, x, y):
        return x + y

    @classmethod
    def zero_(cls, x):
        x[..., :-1].fill_(cls.zero)
        x[..., -1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x):
        return x.fill_(cls.one)


class SampledSemiring(LogSemiring):
    r"""
    Sampling semiring :math:`<\mathrm{logsumexp}, +, -\infty, 0>`,
    which is an exact forward-filtering, backward-sampling approach.
    """

    @classmethod
    def sum(cls, x, dim=-1):
        return SampledLogsumexp.apply(x, dim)


class SampledLogsumexp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim=-1):
        ctx.save_for_backward(x, torch.tensor(dim))
        return x.logsumexp(dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        from torch.distributions import OneHotCategorical
        x, dim = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            return grad_output.unsqueeze(dim).mul(OneHotCategorical(logits=x.movedim(dim, -1)).sample().movedim(-1, dim)), None
        return None, None
