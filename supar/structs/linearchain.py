# -*- coding: utf-8 -*-

import torch
from torch.distributions.utils import lazy_property
from supar.structs.distribution import StructuredDistribution
from supar.structs.semiring import LogSemiring


class LinearChainCRF(StructuredDistribution):
    r"""

        Examples:
            >>> from supar import LinearChainCRF
            >>> batch_size, seq_len, n_tags = 3, 5, 4
            >>> lens = torch.tensor([3, 4, 5])
            >>> value = torch.randint(n_tags, (batch_size, seq_len))
            >>> s1 = LinearChainCRF(torch.randn(batch_size, seq_len, n_tags), torch.randn(n_tags+1, n_tags+1), lens)
            >>> s2 = LinearChainCRF(torch.randn(batch_size, seq_len, n_tags), torch.randn(n_tags+1, n_tags+1), lens)
            >>> s1.max
            tensor([2.4978, 5.7460, 4.9088], grad_fn=<MaxBackward0>)
            >>> s1.argmax
            tensor([[3, 1, 3, 0, 0],
                    [1, 0, 1, 0, 0],
                    [2, 0, 1, 1, 0]])
            >>> s1.log_partition
            tensor([3.7812, 7.9180, 7.8031], grad_fn=<LogsumexpBackward>)
            >>> s1.log_prob(value)
            tensor([ -8.9096, -11.3473,  -9.6189], grad_fn=<SubBackward0>)
            >>> s1.kl(s2)
            tensor([1.9768, 5.1978, 8.6055], grad_fn=<SelectBackward>)
    """

    def __init__(self, scores, trans=None, lens=None):
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

    def score(self, value):
        scores, mask, value = self.scores.transpose(0, 1), self.mask.t(), value.t()
        prev, succ = torch.cat((torch.full_like(value[:1], -1), value[:-1]), 0), value
        # [seq_len, batch_size]
        alpha = scores.gather(-1, value.unsqueeze(-1)).squeeze(-1)
        # [batch_size]
        alpha = LogSemiring.prod(LogSemiring.one_mask(LogSemiring.mul(alpha, self.trans[prev, succ]), ~mask), 0)
        alpha = alpha + self.trans[value.gather(0, self.lens.unsqueeze(0) - 1).squeeze(0), torch.full_like(value[0], -1)]
        return alpha

    def forward(self, semiring):
        # [seq_len, batch_size, n_tags, ...]
        scores = semiring.convert(self.scores.transpose(0, 1))
        trans = semiring.convert(self.trans)
        mask = self.mask.t()

        # [batch_size, n_tags]
        alpha = semiring.mul(trans[-1, :-1], scores[0])
        for i in range(1, len(mask)):
            # [batch_size, n_tags]
            alpha[mask[i]] = semiring.sum(semiring.times(trans[:-1, :-1].unsqueeze(0),
                                                         scores[i].unsqueeze(1),
                                                         alpha.unsqueeze(2)), 1)[mask[i]]
        alpha = semiring.dot(alpha, trans[:-1, -1], 1)
        return semiring.unconvert(alpha)
