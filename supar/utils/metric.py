# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import torch


class Metric(object):

    def __lt__(self, other: 'Metric') -> bool:
        return self.score < other

    def __le__(self, other: 'Metric') -> bool:
        return self.score <= other

    def __ge__(self, other: 'Metric') -> bool:
        return self.score >= other

    def __gt__(self, other: 'Metric') -> bool:
        return self.score > other

    @property
    def score(self):
        return 0.


class AttachmentMetric(Metric):

    def __init__(self, eps: float = 1e-12) -> AttachmentMetric:
        super().__init__()

        self.eps = eps

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
        return s

    def __call__(
        self,
        arc_preds: torch.Tensor,
        rel_preds: torch.Tensor,
        arc_golds: torch.Tensor,
        rel_golds: torch.Tensor,
        mask: torch.BoolTensor
    ) -> AttachmentMetric:
        lens = mask.sum(1)
        arc_mask = arc_preds.eq(arc_golds) & mask
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()
        return self

    @property
    def score(self):
        return self.las

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)


class SpanMetric(Metric):

    def __init__(self, eps: float = 1e-12) -> SpanMetric:
        super().__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds: List[List[Tuple]], golds: List[List[Tuple]]) -> SpanMetric:
        for pred, gold in zip(preds, golds):
            upred, ugold = Counter([tuple(span[:-1]) for span in pred]), Counter([tuple(span[:-1]) for span in gold])
            lpred, lgold = Counter([tuple(span) for span in pred]), Counter([tuple(span) for span in gold])
            utp, ltp = list((upred & ugold).elements()), list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)
        return self

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"

        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)


class ChartMetric(Metric):

    def __init__(self, eps: float = 1e-12) -> ChartMetric:
        super(ChartMetric, self).__init__()

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds: torch.Tensor, golds: torch.Tensor) -> ChartMetric:
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()
        return self

    def __repr__(self):
        return f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)
