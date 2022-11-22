# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional

import torch
from supar.structs.dist import StructuredDistribution
from supar.structs.semiring import LogSemiring, Semiring
from torch.distributions.utils import lazy_property


class LinearChainCRF(StructuredDistribution):
    r"""
        Linear-chain CRFs :cite:`lafferty-etal-2001-crf`.

        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, n_tags]``.
                Log potentials.
            trans (~torch.Tensor): ``[n_tags+1, n_tags+1]``.
                Transition scores.
                ``trans[-1, :-1]``/``trans[:-1, -1]`` represent transitions for start/end positions respectively.
            lens (~torch.LongTensor): ``[batch_size]``.
                Sentence lengths for masking. Default: ``None``.

        Examples:
            >>> from supar import LinearChainCRF
            >>> batch_size, seq_len, n_tags = 2, 5, 4
            >>> lens = torch.tensor([3, 4])
            >>> value = torch.randint(n_tags, (batch_size, seq_len))
            >>> s1 = LinearChainCRF(torch.randn(batch_size, seq_len, n_tags),
                                    torch.randn(n_tags+1, n_tags+1),
                                    lens)
            >>> s2 = LinearChainCRF(torch.randn(batch_size, seq_len, n_tags),
                                    torch.randn(n_tags+1, n_tags+1),
                                    lens)
            >>> s1.max
            tensor([4.4120, 8.9672], grad_fn=<MaxBackward0>)
            >>> s1.argmax
            tensor([[2, 0, 3, 0, 0],
                    [3, 3, 3, 2, 0]])
            >>> s1.log_partition
            tensor([ 6.3486, 10.9106], grad_fn=<LogsumexpBackward>)
            >>> s1.log_prob(value)
            tensor([ -8.1515, -10.5572], grad_fn=<SubBackward0>)
            >>> s1.entropy
            tensor([3.4150, 3.6549], grad_fn=<SelectBackward>)
            >>> s1.kl(s2)
            tensor([4.0333, 4.3807], grad_fn=<SelectBackward>)
    """

    def __init__(
        self,
        scores: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
        lens: Optional[torch.LongTensor] = None
    ) -> LinearChainCRF:
        super().__init__(scores, lens=lens)

        batch_size, seq_len, self.n_tags = scores.shape[:3]
        self.lens = scores.new_full((batch_size,), seq_len).long() if lens is None else lens
        self.mask = self.lens.unsqueeze(-1).gt(self.lens.new_tensor(range(seq_len)))

        self.trans = self.scores.new_full((self.n_tags+1, self.n_tags+1), LogSemiring.one) if trans is None else trans

    def __repr__(self):
        return f"{self.__class__.__name__}(n_tags={self.n_tags})"

    def __add__(self, other):
        return LinearChainCRF(torch.stack((self.scores, other.scores), -1),
                              torch.stack((self.trans, other.trans), -1),
                              self.lens)

    @lazy_property
    def argmax(self):
        return self.lens.new_zeros(self.mask.shape).masked_scatter_(self.mask, torch.where(self.backward(self.max.sum()))[2])

    def topk(self, k: int) -> torch.LongTensor:
        preds = torch.stack([torch.where(self.backward(i))[2] for i in self.kmax(k).sum(0)], -1)
        return self.lens.new_zeros(*self.mask.shape, k).masked_scatter_(self.mask.unsqueeze(-1), preds)

    def score(self, value: torch.LongTensor) -> torch.Tensor:
        scores, mask, value = self.scores.transpose(0, 1), self.mask.t(), value.t()
        prev, succ = torch.cat((torch.full_like(value[:1], -1), value[:-1]), 0), value
        # [seq_len, batch_size]
        alpha = scores.gather(-1, value.unsqueeze(-1)).squeeze(-1)
        # [batch_size]
        alpha = LogSemiring.prod(LogSemiring.one_mask(LogSemiring.mul(alpha, self.trans[prev, succ]), ~mask), 0)
        alpha = alpha + self.trans[value.gather(0, self.lens.unsqueeze(0) - 1).squeeze(0), torch.full_like(value[0], -1)]
        return alpha

    def forward(self, semiring: Semiring) -> torch.Tensor:
        # [seq_len, batch_size, n_tags, ...]
        scores = semiring.convert(self.scores.transpose(0, 1))
        trans = semiring.convert(self.trans)
        mask = self.mask.t()

        # [batch_size, n_tags]
        alpha = semiring.mul(trans[-1, :-1], scores[0])
        for i in range(1, len(mask)):
            alpha[mask[i]] = semiring.mul(semiring.dot(alpha.unsqueeze(2), trans[:-1, :-1], 1), scores[i])[mask[i]]
        alpha = semiring.dot(alpha, trans[:-1, -1], 1)
        return semiring.unconvert(alpha)


class SemiMarkovCRF(StructuredDistribution):
    r"""
        Semi-markov CRFs :cite:`sarawagi-cohen-2004-semicrf`.

        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_tags]``.
                Log potentials.
            trans (~torch.Tensor): ``[n_tags, n_tags]``.
                Transition scores.
            lens (~torch.LongTensor): ``[batch_size]``.
                Sentence lengths for masking. Default: ``None``.

        Examples:
            >>> from supar import SemiMarkovCRF
            >>> batch_size, seq_len, n_tags = 2, 5, 4
            >>> lens = torch.tensor([3, 4])
            >>> value = torch.tensor([[[ 0, -1, -1, -1, -1],
                                       [-1, -1,  2, -1, -1],
                                       [-1, -1, -1, -1, -1],
                                       [-1, -1, -1, -1, -1],
                                       [-1, -1, -1, -1, -1]],
                                      [[-1,  1, -1, -1, -1],
                                       [-1, -1,  3, -1, -1],
                                       [-1, -1, -1,  0, -1],
                                       [-1, -1, -1, -1, -1],
                                       [-1, -1, -1, -1, -1]]])
            >>> s1 = SemiMarkovCRF(torch.randn(batch_size, seq_len, seq_len, n_tags),
                                   torch.randn(n_tags, n_tags),
                                   lens)
            >>> s2 = SemiMarkovCRF(torch.randn(batch_size, seq_len, seq_len, n_tags),
                                   torch.randn(n_tags, n_tags),
                                   lens)
            >>> s1.max
            tensor([4.1971, 5.5746], grad_fn=<MaxBackward0>)
            >>> s1.argmax
            [[[0, 0, 1], [1, 1, 0], [2, 2, 1]], [[0, 0, 1], [1, 1, 3], [2, 2, 0], [3, 3, 1]]]
            >>> s1.log_partition
            tensor([6.3641, 8.4384], grad_fn=<LogsumexpBackward0>)
            >>> s1.log_prob(value)
            tensor([-5.7982, -7.4534], grad_fn=<SubBackward0>)
            >>> s1.entropy
            tensor([3.7520, 5.1609], grad_fn=<SelectBackward0>)
            >>> s1.kl(s2)
            tensor([3.5348, 2.2826], grad_fn=<SelectBackward0>)
    """

    def __init__(
        self,
        scores: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
        lens: Optional[torch.LongTensor] = None
    ) -> SemiMarkovCRF:
        super().__init__(scores, lens=lens)

        batch_size, seq_len, _, self.n_tags = scores.shape[:4]
        self.lens = scores.new_full((batch_size,), seq_len).long() if lens is None else lens
        self.mask = self.lens.unsqueeze(-1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.unsqueeze(1) & self.mask.unsqueeze(2)

        self.trans = self.scores.new_full((self.n_tags, self.n_tags), LogSemiring.one) if trans is None else trans

    def __repr__(self):
        return f"{self.__class__.__name__}(n_tags={self.n_tags})"

    def __add__(self, other):
        return SemiMarkovCRF(torch.stack((self.scores, other.scores), -1),
                             torch.stack((self.trans, other.trans), -1),
                             self.lens)

    @lazy_property
    def argmax(self) -> List:
        return [torch.nonzero(i).tolist() for i in self.backward(self.max.sum())]

    def topk(self, k: int) -> List:
        return list(zip(*[[torch.nonzero(j).tolist() for j in self.backward(i)] for i in self.kmax(k).sum(0)]))

    def score(self, value: torch.LongTensor) -> torch.Tensor:
        mask = self.mask & value.ge(0)
        lens = mask.sum((1, 2))
        indices = torch.where(mask)
        batch_size, seq_len = lens.shape[0], lens.max()
        span_mask = lens.unsqueeze(-1).gt(lens.new_tensor(range(seq_len)))
        scores = self.scores.new_full((batch_size, seq_len), LogSemiring.one)
        scores = scores.masked_scatter_(span_mask, self.scores[(*indices, value[indices])])
        scores = LogSemiring.prod(LogSemiring.one_mask(scores, ~span_mask), -1)
        value = value.new_zeros(batch_size, seq_len).masked_scatter_(span_mask, value[indices])
        trans = LogSemiring.prod(LogSemiring.one_mask(self.trans[value[:, :-1], value[:, 1:]], ~span_mask[:, 1:]), -1)
        return LogSemiring.mul(scores, trans)

    def forward(self, semiring: Semiring) -> torch.Tensor:
        # [seq_len, seq_len, batch_size, n_tags, ...]
        scores = semiring.convert(self.scores.movedim((1, 2), (0, 1)))
        trans = semiring.convert(self.trans)
        # [seq_len, batch_size, n_tags, ...]
        alpha = semiring.zeros_like(scores[0])

        alpha[0] = scores[0, 0]
        # [batch_size, n_tags]
        for t in range(1, len(scores)):
            # [batch_size, n_tags, ...]
            s = semiring.dot(semiring.dot(alpha[:t].unsqueeze(3), trans, 2), scores[1:t+1, t], 0)
            alpha[t] = semiring.sum(torch.stack((s, scores[0, t])), 0)
        return semiring.unconvert(semiring.sum(alpha[self.lens - 1, range(len(self.lens))], 1))
