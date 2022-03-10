# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.structs.dist import StructuredDistribution
from supar.structs.fn import mst
from supar.structs.semiring import LogSemiring
from supar.utils.fn import diagonal_stripe, expanded_stripe, stripe
from torch.distributions.utils import lazy_property


class MatrixTree(StructuredDistribution):
    r"""
    MatrixTree for calculating partitions and marginals of non-projective dependency trees in :math:`O(n^3)`
    by an adaptation of Kirchhoff's MatrixTree Theorem :cite:`koo-etal-2007-structured`.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all possible dependent-head pairs.
        lens (~torch.LongTensor): ``[batch_size]``.
            Sentence lengths for masking, regardless of root positions. Default: ``None``.
        multiroot (bool):
            If ``False``, requires the tree to contain only a single root. Default: ``True``.

    Examples:
        >>> from supar import MatrixTree
        >>> batch_size, seq_len = 2, 5
        >>> lens = torch.tensor([3, 4])
        >>> arcs = torch.tensor([[0, 2, 0, 4, 2], [0, 3, 1, 0, 3]])
        >>> s1 = MatrixTree(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s2 = MatrixTree(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s1.max
        tensor([0.7174, 3.7910], grad_fn=<SumBackward1>)
        >>> s1.argmax
        tensor([[0, 0, 1, 1, 0],
                [0, 4, 1, 0, 3]])
        >>> s1.log_partition
        tensor([2.0229, 6.0558], grad_fn=<CopyBackwards>)
        >>> s1.log_prob(arcs)
        tensor([-3.2209, -2.5756], grad_fn=<SubBackward0>)
        >>> s1.entropy
        tensor([1.9711, 3.4497], grad_fn=<SubBackward0>)
        >>> s1.kl(s2)
        tensor([1.3354, 2.6914], grad_fn=<AddBackward0>)
    """

    def __init__(self, scores, lens=None, multiroot=False):
        super().__init__(scores)

        batch_size, seq_len, *_ = scores.shape
        self.lens = scores.new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.index_fill(1, self.lens.new_tensor(0), 0)

        self.multiroot = multiroot

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    def __add__(self, other):
        return MatrixTree(torch.stack((self.scores, other.scores)), self.lens, self.multiroot)

    @lazy_property
    def max(self):
        arcs = self.argmax
        return LogSemiring.prod(LogSemiring.one_mask(self.scores.gather(-1, arcs.unsqueeze(-1)).squeeze(-1), ~self.mask), -1)

    @lazy_property
    def argmax(self):
        with torch.no_grad():
            return mst(self.scores, self.mask, self.multiroot)

    def kmax(self, k):
        # TODO: Camerini algorithm
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    @lazy_property
    def entropy(self):
        return self.log_partition - (self.marginals * self.scores).sum((-1, -2))

    def cross_entropy(self, other):
        return other.log_partition - (self.marginals * other.scores).sum((-1, -2))

    def kl(self, other):
        return other.log_partition - self.log_partition + (self.marginals * (self.scores - other.scores)).sum((-1, -2))

    def score(self, value, partial=False):
        arcs = value
        if partial:
            mask, lens = self.mask, self.lens
            mask = mask.index_fill(1, self.lens.new_tensor(0), 1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            arcs = arcs.index_fill(1, lens.new_tensor(0), -1).unsqueeze(-1)
            arcs = arcs.eq(lens.new_tensor(range(mask.shape[1]))) | arcs.lt(0)
            scores = LogSemiring.zero_mask(self.scores, ~(arcs & mask))
            return self.__class__(scores, self.mask, **self.kwargs).log_partition
        return LogSemiring.prod(LogSemiring.one_mask(self.scores.gather(-1, arcs.unsqueeze(-1)).squeeze(-1), ~self.mask), -1)

    @torch.enable_grad()
    def forward(self, semiring):
        s_arc = self.scores
        batch_size, *_ = s_arc.shape
        mask, lens = self.mask.index_fill(1, self.lens.new_tensor(0), 1), self.lens
        # double precision to prevent overflows
        s_arc = semiring.zero_mask(s_arc, ~(mask.unsqueeze(-1) & mask.unsqueeze(-2))).double()

        # A(i, j) = exp(s(i, j))
        m = s_arc.view(batch_size, -1).max(-1)[0]
        A = torch.exp(s_arc - m.view(-1, 1, 1))

        # Weighted degree matrix
        # D(i, j) = sum_j(A(i, j)), if h == m
        #           0,              otherwise
        D = torch.zeros_like(A)
        D.diagonal(0, 1, 2).copy_(A.sum(-1))
        # Laplacian matrix
        # L(i, j) = D(i, j) - A(i, j)
        L = D - A
        if not self.multiroot:
            L.diagonal(0, 1, 2).add_(-A[..., 0])
            L[..., 1] = A[..., 0]
        L = nn.init.eye_(torch.empty_like(A[0])).repeat(batch_size, 1, 1).masked_scatter_(mask.unsqueeze(-1), L[mask])
        L = L + nn.init.eye_(torch.empty_like(A[0])) * torch.finfo().tiny
        # Z = L^(0, 0), the minor of L w.r.t row 0 and column 0
        return (L[:, 1:, 1:].logdet() + m * lens).float()


class DependencyCRF(StructuredDistribution):
    r"""
    First-order TreeCRF for projective dependency trees :cite:`eisner-2000-bilexical,zhang-etal-2020-efficient`.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all possible dependent-head pairs.
        lens (~torch.LongTensor): ``[batch_size]``.
            Sentence lengths for masking, regardless of root positions. Default: ``None``.
        multiroot (bool):
            If ``False``, requires the tree to contain only a single root. Default: ``True``.

    Examples:
        >>> from supar import DependencyCRF
        >>> batch_size, seq_len = 2, 5
        >>> lens = torch.tensor([3, 4])
        >>> arcs = torch.tensor([[0, 2, 0, 4, 2], [0, 3, 1, 0, 3]])
        >>> s1 = DependencyCRF(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s2 = DependencyCRF(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s1.max
        tensor([3.6346, 1.7194], grad_fn=<IndexBackward>)
        >>> s1.argmax
        tensor([[0, 2, 3, 0, 0],
                [0, 0, 3, 1, 1]])
        >>> s1.log_partition
        tensor([4.1007, 3.3383], grad_fn=<IndexBackward>)
        >>> s1.log_prob(arcs)
        tensor([-1.3866, -5.5352], grad_fn=<SubBackward0>)
        >>> s1.entropy
        tensor([0.9979, 2.6056], grad_fn=<IndexBackward>)
        >>> s1.kl(s2)
        tensor([1.6631, 2.6558], grad_fn=<IndexBackward>)
    """

    def __init__(self, scores, lens=None, multiroot=False):
        super().__init__(scores)

        batch_size, seq_len, *_ = scores.shape
        self.lens = scores.new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.index_fill(1, self.lens.new_tensor(0), 0)

        self.multiroot = multiroot

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    def __add__(self, other):
        return DependencyCRF(torch.stack((self.scores, other.scores), -1), self.lens, self.multiroot)

    @lazy_property
    def argmax(self):
        return self.lens.new_zeros(self.mask.shape).masked_scatter_(self.mask, torch.where(self.backward(self.max.sum()))[2])

    def topk(self, k):
        preds = torch.stack([torch.where(self.backward(i))[2] for i in self.kmax(k).sum(0)], -1)
        return self.lens.new_zeros(*self.mask.shape, k).masked_scatter_(self.mask.unsqueeze(-1), preds)

    def score(self, value, partial=False):
        arcs = value
        if partial:
            mask, lens = self.mask, self.lens
            mask = mask.index_fill(1, self.lens.new_tensor(0), 1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            arcs = arcs.index_fill(1, lens.new_tensor(0), -1).unsqueeze(-1)
            arcs = arcs.eq(lens.new_tensor(range(mask.shape[1]))) | arcs.lt(0)
            scores = LogSemiring.zero_mask(self.scores, ~(arcs & mask))
            return self.__class__(scores, self.mask, **self.kwargs).log_partition
        return LogSemiring.prod(LogSemiring.one_mask(self.scores.gather(-1, arcs.unsqueeze(-1)).squeeze(-1), ~self.mask), -1)

    def forward(self, semiring):
        s_arc = self.scores
        batch_size, seq_len = s_arc.shape[:2]
        # [seq_len, seq_len, batch_size, ...], (h->m)
        s_arc = semiring.convert(s_arc.movedim((1, 2), (1, 0)))
        s_i = semiring.zeros_like(s_arc)
        s_c = semiring.zeros_like(s_arc)
        semiring.one_(s_c.diagonal().movedim(-1, 1))

        for w in range(1, seq_len):
            n = seq_len - w

            # [n, batch_size, ...]
            il = ir = semiring.dot(stripe(s_c, n, w), stripe(s_c, n, w, (w, 1)), 1)
            # INCOMPLETE-L: I(j->i) = <C(i->r), C(j->r+1)> * s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i with I(j->i) of n spans
            s_i.diagonal(-w).copy_(semiring.mul(il, s_arc.diagonal(-w).movedim(-1, 0)).movedim(0, -1))
            # INCOMPLETE-R: I(i->j) = <C(i->r), C(j->r+1)> * s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i with I(i->j) of n spans
            s_i.diagonal(w).copy_(semiring.mul(ir, s_arc.diagonal(w).movedim(-1, 0)).movedim(0, -1))

            # [n, batch_size, ...]
            # COMPLETE-L: C(j->i) = <C(r->i), I(j->r)>, i <= r < j
            cl = semiring.dot(stripe(s_c, n, w, (0, 0), 0), stripe(s_i, n, w, (w, 0)), 1)
            s_c.diagonal(-w).copy_(cl.movedim(0, -1))
            # COMPLETE-R: C(i->j) = <I(i->r), C(r->j)>, i < r <= j
            cr = semiring.dot(stripe(s_i, n, w, (0, 1)), stripe(s_c, n, w, (1, w), 0), 1)
            s_c.diagonal(w).copy_(cr.movedim(0, -1))
            if not self.multiroot:
                s_c[0, w][self.lens.ne(w)] = semiring.zero
        return semiring.unconvert(s_c)[0][self.lens, range(batch_size)]


class Dependency2oCRF(StructuredDistribution):
    r"""
    Second-order TreeCRF for projective dependency trees :cite:`mcdonald-pereira-2006-online,zhang-etal-2020-efficient`.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all possible dependent-head pairs.
        lens (~torch.LongTensor): ``[batch_size]``.
            Sentence lengths for masking, regardless of root positions. Default: ``None``.
        multiroot (bool):
            If ``False``, requires the tree to contain only a single root. Default: ``True``.

    Examples:
        >>> from supar import Dependency2oCRF
        >>> batch_size, seq_len = 2, 5
        >>> lens = torch.tensor([3, 4])
        >>> arcs = torch.tensor([[0, 2, 0, 4, 2], [0, 3, 1, 0, 3]])
        >>> sibs = torch.tensor([CoNLL.get_sibs(i) for i in arcs[:, 1:].tolist()])
        >>> s1 = Dependency2oCRF((torch.randn(batch_size, seq_len, seq_len),
                                  torch.randn(batch_size, seq_len, seq_len, seq_len)),
                                 lens)
        >>> s2 = Dependency2oCRF((torch.randn(batch_size, seq_len, seq_len),
                                  torch.randn(batch_size, seq_len, seq_len, seq_len)),
                                 lens)
        >>> s1.max
        tensor([0.7574, 3.3634], grad_fn=<IndexBackward>)
        >>> s1.argmax
        tensor([[0, 3, 3, 0, 0],
                [0, 4, 4, 4, 0]])
        >>> s1.log_partition
        tensor([1.9906, 4.3599], grad_fn=<IndexBackward>)
        >>> s1.log_prob((arcs, sibs))
        tensor([-0.6975, -6.2845], grad_fn=<SubBackward0>)
        >>> s1.entropy
        tensor([1.6436, 2.1717], grad_fn=<IndexBackward>)
        >>> s1.kl(s2)
        tensor([0.4929, 2.0759], grad_fn=<IndexBackward>)
    """

    def __init__(self, scores, lens=None, multiroot=False):
        super().__init__(scores)

        batch_size, seq_len, *_ = scores[0].shape
        self.lens = scores[0].new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.index_fill(1, self.lens.new_tensor(0), 0)

        self.multiroot = multiroot

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    def __add__(self, other):
        return Dependency2oCRF([torch.stack((i, j), -1) for i, j in zip(self.scores, other.scores)], self.lens, self.multiroot)

    @lazy_property
    def argmax(self):
        return self.lens.new_zeros(self.mask.shape).masked_scatter_(self.mask, torch.where(self.backward(self.max.sum()))[2])

    def topk(self, k):
        preds = torch.stack([torch.where(self.backward(i))[2] for i in self.kmax(k).sum(0)], -1)
        return self.lens.new_zeros(*self.mask.shape, k).masked_scatter_(self.mask.unsqueeze(-1), preds)

    def score(self, value, partial=False):
        arcs, sibs = value
        if partial:
            mask, lens = self.mask, self.lens
            mask = mask.index_fill(1, self.lens.new_tensor(0), 1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            arcs = arcs.index_fill(1, lens.new_tensor(0), -1).unsqueeze(-1)
            arcs = arcs.eq(lens.new_tensor(range(mask.shape[1]))) | arcs.lt(0)
            s_arc, s_sib = LogSemiring.zero_mask(self.scores[0], ~(arcs & mask)), self.scores[1]
            return self.__class__((s_arc, s_sib), self.mask, **self.kwargs).log_partition
        s_arc = self.scores[0].gather(-1, arcs.unsqueeze(-1)).squeeze(-1)
        s_arc = LogSemiring.prod(LogSemiring.one_mask(s_arc, ~self.mask), -1)
        s_sib = self.scores[1].gather(-1, sibs.unsqueeze(-1)).squeeze(-1)
        s_sib = LogSemiring.prod(LogSemiring.one_mask(s_sib, ~sibs.gt(0)), (-1, -2))
        return LogSemiring.mul(s_arc, s_sib)

    @torch.enable_grad()
    def forward(self, semiring):
        s_arc, s_sib = self.scores
        batch_size, seq_len = s_arc.shape[:2]
        # [seq_len, seq_len, batch_size, ...], (h->m)
        s_arc = semiring.convert(s_arc.movedim((1, 2), (1, 0)))
        # [seq_len, seq_len, seq_len, batch_size, ...], (h->m->s)
        s_sib = semiring.convert(s_sib.movedim((0, 2), (3, 0)))
        s_i = semiring.zeros_like(s_arc)
        s_s = semiring.zeros_like(s_arc)
        s_c = semiring.zeros_like(s_arc)
        semiring.one_(s_c.diagonal().movedim(-1, 1))

        for w in range(1, seq_len):
            n = seq_len - w

            # INCOMPLETE-L: I(j->i) = <I(j->r), S(j->r, i)> * s(j->i), i < r < j
            #                         <C(j->j), C(i->j-1)>  * s(j->i), otherwise
            # [n, w, batch_size, ...]
            il = semiring.times(stripe(s_i, n, w, (w, 1)),
                                stripe(s_s, n, w, (1, 0), 0),
                                stripe(s_sib[range(w, n+w), range(n), :], n, w, (0, 1)))
            il[:, -1] = semiring.mul(stripe(s_c, n, 1, (w, w)), stripe(s_c, n, 1, (0, w - 1))).squeeze(1)
            il = semiring.sum(il, 1)
            s_i.diagonal(-w).copy_(semiring.mul(il, s_arc.diagonal(-w).movedim(-1, 0)).movedim(0, -1))
            # INCOMPLETE-R: I(i->j) = <I(i->r), S(i->r, j)> * s(i->j), i < r < j
            #                         <C(i->i), C(j->i+1)>  * s(i->j), otherwise
            # [n, w, batch_size, ...]
            ir = semiring.times(stripe(s_i, n, w),
                                stripe(s_s, n, w, (0, w), 0),
                                stripe(s_sib[range(n), range(w, n+w), :], n, w))
            if not self.multiroot:
                semiring.zero_(ir[0])
            ir[:, 0] = semiring.mul(stripe(s_c, n, 1), stripe(s_c, n, 1, (w, 1))).squeeze(1)
            ir = semiring.sum(ir, 1)
            s_i.diagonal(w).copy_(semiring.mul(ir, s_arc.diagonal(w).movedim(-1, 0)).movedim(0, -1))

            # [batch_size, ..., n]
            sl = sr = semiring.dot(stripe(s_c, n, w), stripe(s_c, n, w, (w, 1)), 1).movedim(0, -1)
            # SIB: S(j, i) = <C(i->r), C(j->r+1)>, i <= r < j
            s_s.diagonal(-w).copy_(sl)
            # SIB: S(i, j) = <C(i->r), C(j->r+1)>, i <= r < j
            s_s.diagonal(w).copy_(sr)

            # [n, batch_size, ...]
            # COMPLETE-L: C(j->i) = <C(r->i), I(j->r)>, i <= r < j
            cl = semiring.dot(stripe(s_c, n, w, (0, 0), 0), stripe(s_i, n, w, (w, 0)), 1)
            s_c.diagonal(-w).copy_(cl.movedim(0, -1))
            # COMPLETE-R: C(i->j) = <I(i->r), C(r->j)>, i < r <= j
            cr = semiring.dot(stripe(s_i, n, w, (0, 1)), stripe(s_c, n, w, (1, w), 0), 1)
            s_c.diagonal(w).copy_(cr.movedim(0, -1))
        return semiring.unconvert(s_c)[0][self.lens, range(batch_size)]


class ConstituencyCRF(StructuredDistribution):
    r"""
    Constituency TreeCRF :cite:`zhang-etal-2020-fast,stern-etal-2017-minimal`.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all constituents.
        lens (~torch.LongTensor): ``[batch_size]``.
            Sentence lengths for masking.

    Examples:
        >>> from supar import ConstituencyCRF
        >>> batch_size, seq_len = 2, 5
        >>> lens = torch.tensor([3, 4])
        >>> charts = torch.tensor([[[0, 1, 0, 1, 0],
                                    [0, 0, 1, 1, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0]],
                                   [[0, 1, 1, 0, 1],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 1],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0]]]).bool()
        >>> s1 = ConstituencyCRF(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s2 = ConstituencyCRF(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s1.max
        tensor([ 2.5068, -0.5628], grad_fn=<IndexBackward>)
        >>> s1.argmax
        [[[0, 3], [0, 1], [1, 3], [1, 2], [2, 3]], [[0, 4], [0, 2], [0, 1], [1, 2], [2, 4], [2, 3], [3, 4]]]
        >>> s1.log_partition
        tensor([2.9235, 0.0154], grad_fn=<IndexBackward>)
        >>> s1.log_prob(charts)
        tensor([-0.4167, -0.5781], grad_fn=<SubBackward0>)
        >>> s1.entropy
        tensor([0.6415, 1.2026], grad_fn=<IndexBackward>)
        >>> s1.kl(s2)
        tensor([0.0362, 2.9017], grad_fn=<IndexBackward>)
    """

    def __init__(self, scores, lens=None):
        super().__init__(scores)

        batch_size, seq_len, *_ = scores.shape
        self.lens = scores.new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.unsqueeze(1) & scores.new_ones(scores.shape[:3]).bool().triu_(1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __add__(self, other):
        return ConstituencyCRF(torch.stack((self.scores, other.scores), -1), self.lens)

    @lazy_property
    def argmax(self):
        return [sorted(torch.nonzero(i).tolist(), key=lambda x:(x[0], -x[1])) for i in self.backward(self.max.sum())]

    def topk(self, k):
        return list(zip(*[[sorted(torch.nonzero(j).tolist(), key=lambda x:(x[0], -x[1])) for j in self.backward(i)]
                          for i in self.kmax(k).sum(0)]))

    def score(self, value):
        return LogSemiring.prod(LogSemiring.prod(LogSemiring.one_mask(self.scores, ~(self.mask & value)), -1), -1)

    @torch.enable_grad()
    def forward(self, semiring):
        batch_size, seq_len = self.scores.shape[:2]
        # [seq_len, seq_len, batch_size, ...], (l->r)
        scores = semiring.convert(self.scores.movedim((1, 2), (0, 1)))
        s = semiring.zeros_like(scores)
        s.diagonal(1).copy_(scores.diagonal(1))

        for w in range(2, seq_len):
            n = seq_len - w
            # [n, batch_size, ...]
            s_s = semiring.dot(stripe(s, n, w-1, (0, 1)), stripe(s, n, w-1, (1, w), False), 1)
            s.diagonal(w).copy_(semiring.mul(s_s, scores.diagonal(w).movedim(-1, 0)).movedim(0, -1))
        return semiring.unconvert(s)[0][self.lens, range(batch_size)]


class BiLexicalizedConstituencyCRF(StructuredDistribution):
    r"""
    Grammarless Eisner-Satta Algorithm :cite:`eisner-satta-1999-efficient,yang-etal-2021-neural`.

    Code is revised from `Songlin Yang's implementation <https://github.com/sustcsonglin/span-based-dependency-parsing>`_.

    Args:
        scores (~torch.Tensor): ``[2, batch_size, seq_len, seq_len]``.
            Scores of dependencies and constituents.
        lens (~torch.LongTensor): ``[batch_size]``.
            Sentence lengths for masking.

    Examples:
        >>> from supar import BiLexicalizedConstituencyCRF
        >>> batch_size, seq_len = 2, 5
        >>> lens = torch.tensor([3, 4])
        >>> deps = torch.tensor([[0, 0, 1, 1, 0], [0, 3, 1, 0, 3]])
        >>> cons = torch.tensor([[[0, 1, 1, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0]],
                                 [[0, 1, 1, 1, 1],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0]]]).bool()
        >>> s1 = BiLexicalizedConstituencyCRF(torch.randn(2, batch_size, seq_len, seq_len), lens)
        >>> s2 = BiLexicalizedConstituencyCRF(torch.randn(2, batch_size, seq_len, seq_len), lens)
        >>> s1.max
        tensor([1.3564, 0.9958], grad_fn=<MaxBackward0>)
        >>> s1.argmax[0]
        tensor([[0, 2, 0, 2, 0],
                [0, 0, 1, 1, 3]])
        >>> s1.argmax[1]
        [[[0, 3], [0, 2], [0, 1], [1, 2], [2, 3]], [[0, 4], [0, 2], [0, 1], [1, 2], [2, 4], [2, 3], [3, 4]]]
        >>> s1.log_partition
        tensor([2.4153, 2.3289], grad_fn=<LogsumexpBackward>)
        >>> s1.log_prob((deps, cons))
        tensor([-1.4234, -4.1386], grad_fn=<SubBackward0>)
        >>> s1.entropy
        tensor([1.6121, 2.7814], grad_fn=<SelectBackward>)
        >>> s1.kl(s2)
        tensor([2.1838, 5.6104], grad_fn=<SelectBackward>)
    """

    def __init__(self, scores, lens=None):
        super().__init__(scores)

        batch_size, seq_len, *_ = scores[1].shape
        self.lens = scores[1].new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.unsqueeze(1) & scores[1].new_ones(scores[1].shape[:3]).bool().triu_(1)

    @lazy_property
    def argmax(self):
        marginals = self.backward(self.max.sum())
        dep_mask = self.mask[:, 0]
        dep = self.lens.new_zeros(dep_mask.shape).masked_scatter_(dep_mask, torch.where(marginals[0])[2])
        con = [sorted(torch.nonzero(i).tolist(), key=lambda x:(x[0], -x[1])) for i in marginals[1]]
        return dep, con

    def topk(self, k):
        dep_mask = self.mask[:, 0]
        marginals = [self.backward(i) for i in self.kmax(k).sum(0)]
        dep_preds = torch.stack([torch.where(i[0])[2] for i in marginals], -1)
        dep_preds = self.lens.new_zeros(*dep_mask.shape, k).masked_scatter_(dep_mask.unsqueeze(-1), dep_preds)
        con_preds = list(zip(*[[sorted(torch.nonzero(j).tolist(), key=lambda x:(x[0], -x[1])) for j in i[1]]
                               for i in marginals]))
        return dep_preds, con_preds

    def score(self, value):
        deps, cons = value
        s_dep, s_con = self.scores
        dep_mask, con_mask = self.mask[:, 0], self.mask
        s_dep = LogSemiring.prod(LogSemiring.one_mask(s_dep.gather(-1, deps.unsqueeze(-1)).squeeze(-1), ~dep_mask), -1)
        s_con = LogSemiring.prod(LogSemiring.prod(LogSemiring.one_mask(s_con, ~(con_mask & cons)), -1), -1)
        return LogSemiring.mul(s_dep, s_con)

    def forward(self, semiring):
        s_dep, s_con = self.scores[0], self.scores[1]
        batch_size, seq_len, *_ = s_con.shape
        # [seq_len, seq_len, batch_size, ...], (m<-h)
        s_dep = semiring.convert(s_dep.movedim(0, 2))
        s_root, s_dep = s_dep[1:, 0], s_dep[1:, 1:]
        # [seq_len, seq_len, batch_size, ...], (l->r)
        s_con = semiring.convert(s_con.movedim((1, 2), (0, 1)))
        # [seq_len, seq_len, seq_len, batch_size, ...], (i, j, h)
        s_span = semiring.zero_(s_con.new_empty(seq_len, seq_len, seq_len-1, *s_con.shape[2:]))
        # [seq_len, seq_len, seq_len, batch_size, ...], (i, j<-h)
        s_hook = semiring.zero_(s_con.new_empty(seq_len, seq_len, seq_len-1, *s_con.shape[2:]))
        diagonal_stripe(s_span, 1).copy_(s_con.diagonal(1).movedim(-1, 0).unsqueeze(1))
        s_hook.diagonal(1).copy_(semiring.mul(s_dep, s_con.diagonal(1).movedim(-1, 0).unsqueeze(1)).movedim(0, -1))

        for w in range(2, seq_len):
            n = seq_len - w
            # COMPLETE-L: s_span_l(i, j, h) = <s_span(i, k, h), s_hook(h->k, j)>, i < k < j
            # [n, w, batch_size, ...]
            s_l = stripe(semiring.dot(stripe(s_span, n, w-1, (0, 1)), stripe(s_hook, n, w-1, (1, w), False), 1), n, w)
            # COMPLETE-R: s_span_r(i, j, h) = <s_hook(i, k<-h), s_span(k, j, h)>, i < k < j
            # [n, w, batch_size, ...]
            s_r = stripe(semiring.dot(stripe(s_hook, n, w-1, (0, 1)), stripe(s_span, n, w-1, (1, w), False), 1), n, w)
            # COMPLETE: s_span(i, j, h) = s_span_l(i, j, h) + s_span_r(i, j, h) + s(i, j)
            # [n, w, batch_size, ...]
            s = semiring.mul(semiring.sum(torch.stack((s_l, s_r)), 0), s_con.diagonal(w).movedim(-1, 0).unsqueeze(1))
            diagonal_stripe(s_span, w).copy_(s)

            if w == seq_len - 1:
                continue
            # ATTACH: s_hook(h->i, j) = <s(h->m), s_span(i, j, m)>, i <= m < j
            # [n, seq_len, batch_size, ...]
            s = semiring.dot(expanded_stripe(s_dep, n, w), diagonal_stripe(s_span, w).unsqueeze(2), 1)
            s_hook.diagonal(w).copy_(s.movedim(0, -1))
        return semiring.unconvert(semiring.dot(s_span[0][self.lens, :, range(batch_size)].transpose(0, 1), s_root, 0))
