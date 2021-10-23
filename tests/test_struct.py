# -*- coding: utf-8 -*-

import itertools

import torch
from supar.structs import (ConstituencyCRF, Dependency2oCRF, DependencyCRF,
                           LinearChainCRF)
from supar.structs.semiring import LogSemiring, MaxSemiring, Semiring
from supar.utils.transform import CoNLL
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property


class BruteForceStructuredDistribution(Distribution):

    def __init__(self, scores, **kwargs):
        self.kwargs = kwargs

        self.scores = scores.requires_grad_() if isinstance(scores, torch.Tensor) else [s.requires_grad_() for s in scores]

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @lazy_property
    def log_partition(self):
        return torch.stack([LogSemiring.sum(i, -1) for i in self.enumerate(LogSemiring)])

    @lazy_property
    def max(self):
        return torch.stack([MaxSemiring.sum(i, -1) for i in self.enumerate(MaxSemiring)])

    def kmax(self, k):
        return torch.stack([i.topk(k)[0] for i in self.enumerate(MaxSemiring)])

    @lazy_property
    def entropy(self):
        ps = [seq - self.log_partition[i] for i, seq in enumerate(self.enumerate(LogSemiring))]
        return -torch.stack([(i.exp() * i).sum() for i in ps])

    @lazy_property
    def count(self):
        structs = self.enumerate(Semiring)
        return torch.tensor([len(i) for i in structs]).to(structs[0].device).long()

    def cross_entropy(self, other):
        ps = [seq - self.log_partition[i] for i, seq in enumerate(self.enumerate(LogSemiring))]
        qs = [seq - other.log_partition[i] for i, seq in enumerate(other.enumerate(LogSemiring))]
        return -torch.stack([(i.exp() * j).sum() for i, j in zip(ps, qs)])

    def kl(self, other):
        return self.cross_entropy(other) - self.entropy

    def enumerate(self, semiring):
        raise NotImplementedError


class BruteForceDependencyCRF(BruteForceStructuredDistribution):

    def __init__(self, scores, lens=None, multiroot=False):
        super().__init__(scores)

        batch_size, seq_len = scores.shape[:2]
        self.lens = scores.new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.index_fill(1, self.lens.new_tensor(0), 0)

        self.multiroot = multiroot

    def enumerate(self, semiring):
        trees = []
        for i, length in enumerate(self.lens.tolist()):
            trees.append([])
            for seq in itertools.product(range(length + 1), repeat=length):
                if not CoNLL.istree(list(seq), True, self.multiroot):
                    continue
                trees[-1].append(semiring.prod(self.scores[i, range(1, length + 1), seq], -1))
        return [torch.stack(seq) for seq in trees]


class BruteForceDependency2oCRF(BruteForceStructuredDistribution):

    def __init__(self, scores, lens=None, multiroot=False):
        super().__init__(scores)

        batch_size, seq_len = scores[0].shape[:2]
        self.lens = scores[0].new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.index_fill(1, self.lens.new_tensor(0), 0)

        self.multiroot = multiroot

    def enumerate(self, semiring):
        trees = []
        for i, length in enumerate(self.lens.tolist()):
            trees.append([])
            for seq in itertools.product(range(length + 1), repeat=length):
                if not CoNLL.istree(list(seq), True, self.multiroot):
                    continue
                sibs = self.lens.new_tensor(CoNLL.get_sibs(seq))
                sib_mask = sibs.gt(0)
                s_arc = self.scores[0][i, :length+1, :length+1]
                s_sib = self.scores[1][i, :length+1, :length+1, :length+1]
                s_arc = semiring.prod(s_arc[range(1, length + 1), seq], -1)
                s_sib = semiring.prod(s_sib[1:][sib_mask].gather(-1, sibs[sib_mask].unsqueeze(-1)).squeeze(-1))
                trees[-1].append(semiring.mul(s_arc, s_sib))
        return [torch.stack(seq) for seq in trees]


class BruteForceConstituencyCRF(BruteForceStructuredDistribution):

    def __init__(self, scores, lens=None, labeled=False):
        super().__init__(scores)

        batch_size, seq_len = scores.shape[:2]
        self.lens = scores.new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.unsqueeze(1) & scores.new_ones(scores.shape[:3]).bool().triu_(1)

        self.labeled = labeled

    def enumerate(self, semiring):
        scores = self.scores if self.labeled else self.scores.unsqueeze(-1)

        def enumerate(s, i, j):
            if i + 1 == j:
                yield from s[i, j].unbind(-1)
            for k in range(i + 1, j):
                for t1 in enumerate(s, i, k):
                    for t2 in enumerate(s, k, j):
                        for t in s[i, j].unbind(-1):
                            yield semiring.times(t, t1, t2)
        return [torch.stack([i for i in enumerate(s, 0, length)]) for s, length in zip(scores, self.lens)]


class BruteForceLinearChainCRF(BruteForceStructuredDistribution):

    def __init__(self, scores, trans=None, lens=None):
        super().__init__(scores, lens=lens)

        batch_size, seq_len, self.n_tags = scores.shape[:3]
        self.lens = scores.new_full((batch_size,), seq_len).long() if lens is None else lens
        self.mask = self.lens.unsqueeze(-1).gt(self.lens.new_tensor(range(seq_len)))

        self.trans = self.scores.new_full((self.n_tags+1, self.n_tags+1), LogSemiring.one) if trans is None else trans

    def enumerate(self, semiring):
        seqs = []
        for i, length in enumerate(self.lens.tolist()):
            seqs.append([])
            for seq in itertools.product(range(self.scores[0].shape[-1]), repeat=length):
                prev, succ = (-1,) + seq,  seq + (-1,)
                seqs[-1].append(semiring.prod(torch.cat((self.scores[i, range(length), seq], self.trans[prev, succ])), -1))
        return [torch.stack(seq) for seq in seqs]


def test_struct():
    torch.manual_seed(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size, seq_len, n_tags, k = 2, 6, 4, 3
    lens = torch.randint(3, seq_len-1, (batch_size,)).to(device)

    def enumerate():
        s1 = torch.randn(batch_size, seq_len, seq_len).to(device)
        s2 = torch.randn(batch_size, seq_len, seq_len).to(device)
        yield (DependencyCRF(s1, lens), DependencyCRF(s2, lens),
               BruteForceDependencyCRF(s1, lens), BruteForceDependencyCRF(s2, lens))
        yield (DependencyCRF(s1, lens, multiroot=True), DependencyCRF(s2, lens, multiroot=True),
               BruteForceDependencyCRF(s1, lens, multiroot=True), BruteForceDependencyCRF(s2, lens, multiroot=True))
        s1 = [torch.randn(batch_size, seq_len, seq_len).to(device),
              torch.randn(batch_size, seq_len, seq_len, seq_len).to(device)]
        s2 = [torch.randn(batch_size, seq_len, seq_len).to(device),
              torch.randn(batch_size, seq_len, seq_len, seq_len).to(device)]
        yield (Dependency2oCRF(s1, lens), Dependency2oCRF(s2, lens),
               BruteForceDependency2oCRF(s1, lens), BruteForceDependency2oCRF(s2, lens))
        yield (Dependency2oCRF(s1, lens, multiroot=True), Dependency2oCRF(s2, lens, multiroot=True),
               BruteForceDependency2oCRF(s1, lens, multiroot=True), BruteForceDependency2oCRF(s2, lens, multiroot=True))
        s1 = torch.randn(batch_size, seq_len, seq_len).to(device)
        s2 = torch.randn(batch_size, seq_len, seq_len).to(device)
        yield (ConstituencyCRF(s1, lens), ConstituencyCRF(s2, lens),
               BruteForceConstituencyCRF(s1, lens), BruteForceConstituencyCRF(s2, lens))
        s1 = torch.randn(batch_size, seq_len, seq_len, n_tags).to(device)
        s2 = torch.randn(batch_size, seq_len, seq_len, n_tags).to(device)
        yield (ConstituencyCRF(s1, lens, labeled=True), ConstituencyCRF(s2, lens, labeled=True),
               BruteForceConstituencyCRF(s1, lens, labeled=True), BruteForceConstituencyCRF(s2, lens, labeled=True))
        s1 = torch.randn(batch_size, seq_len, n_tags).to(device)
        s2 = torch.randn(batch_size, seq_len, n_tags).to(device)
        t1 = torch.randn(n_tags+1, n_tags+1).to(device)
        t2 = torch.randn(n_tags+1, n_tags+1).to(device)
        yield (LinearChainCRF(s1, lens=lens), LinearChainCRF(s2, lens=lens),
               BruteForceLinearChainCRF(s1, lens=lens), BruteForceLinearChainCRF(s2, lens=lens))
        yield (LinearChainCRF(s1, t1, lens=lens), LinearChainCRF(s2, t2, lens=lens),
               BruteForceLinearChainCRF(s1, t1, lens=lens), BruteForceLinearChainCRF(s2, t2, lens=lens))

    for _ in range(5):
        for struct1, struct2, brute1, brute2 in enumerate():
            assert struct1.max.allclose(brute1.max)
            assert struct1.kmax(k).allclose(brute1.kmax(k))
            assert struct1.log_partition.allclose(brute1.log_partition)
            assert struct1.entropy.allclose(brute1.entropy)
            assert struct1.cross_entropy(struct2).allclose(brute1.cross_entropy(brute2))
            assert struct1.kl(struct2).allclose(brute1.kl(brute2))
