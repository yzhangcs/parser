# -*- coding: utf-8 -*-

import torch
from supar.structs.distribution import StructuredDistribution
from supar.structs.semiring import LogSemiring


class LinearChainCRF(StructuredDistribution):
    r"""
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
