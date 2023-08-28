# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import torch

from supar.utils.fn import pad


class Metric(object):

    def __init__(self, reverse: Optional[bool] = None, eps: float = 1e-12) -> Metric:
        super().__init__()

        self.n = 0.0
        self.count = 0.0
        self.total_loss = 0.0
        self.reverse = reverse
        self.eps = eps

    def __repr__(self):
        return f"loss: {self.loss:.4f} - " + ' '.join([f"{key}: {val:6.2%}" for key, val in self.values.items()])

    def __lt__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return True
        if not hasattr(other, 'score'):
            return False
        return (self.score < other.score) if not self.reverse else (self.score > other.score)

    def __le__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return True
        if not hasattr(other, 'score'):
            return False
        return (self.score <= other.score) if not self.reverse else (self.score >= other.score)

    def __gt__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return False
        if not hasattr(other, 'score'):
            return True
        return (self.score > other.score) if not self.reverse else (self.score < other.score)

    def __ge__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return False
        if not hasattr(other, 'score'):
            return True
        return (self.score >= other.score) if not self.reverse else (self.score <= other.score)

    def __add__(self, other: Metric) -> Metric:
        return other

    @property
    def score(self):
        raise AttributeError

    @property
    def loss(self):
        return self.total_loss / (self.count + self.eps)

    @property
    def values(self):
        raise AttributeError


class AttachmentMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[Tuple[Union[List, torch.Tensor], Union[List, torch.Tensor]]] = None,
        golds: Optional[Tuple[Union[List, torch.Tensor], Union[List, torch.Tensor]]] = None,
        mask: Optional[torch.BoolTensor] = None,
        reverse: bool = False,
        eps: float = 1e-12
    ) -> AttachmentMetric:
        super().__init__(reverse=reverse, eps=eps)

        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

        if loss is not None:
            self(loss, preds, golds, mask)

    def __call__(
        self,
        loss: float,
        preds: Tuple[Union[List, torch.Tensor], Union[List, torch.Tensor]],
        golds: Tuple[Union[List, torch.Tensor], Union[List, torch.Tensor]],
        mask: torch.BoolTensor
    ) -> AttachmentMetric:
        lens = mask.sum(1)
        arc_preds, rel_preds, arc_golds, rel_golds = *preds, *golds
        if isinstance(arc_preds, torch.Tensor):
            arc_mask = arc_preds.eq(arc_golds)
            rel_mask = rel_preds.eq(rel_golds)
        else:
            arc_mask = pad([mask.new_tensor([i == j for i, j in zip(pred, gold)]) for pred, gold in zip(arc_preds, arc_golds)])
            rel_mask = pad([mask.new_tensor([i == j for i, j in zip(pred, gold)]) for pred, gold in zip(rel_preds, rel_golds)])
        arc_mask = arc_mask & mask
        rel_mask = rel_mask & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.count += 1
        self.total_loss += float(loss)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()
        return self

    def __add__(self, other: AttachmentMetric) -> AttachmentMetric:
        metric = AttachmentMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss
        metric.n_ucm = self.n_ucm + other.n_ucm
        metric.n_lcm = self.n_lcm + other.n_lcm
        metric.total = self.total + other.total
        metric.correct_arcs = self.correct_arcs + other.correct_arcs
        metric.correct_rels = self.correct_rels + other.correct_rels
        metric.reverse = self.reverse or other.reverse
        return metric

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

    @property
    def values(self) -> Dict:
        return {'UCM': self.ucm,
                'LCM': self.lcm,
                'UAS': self.uas,
                'LAS': self.las}


class SpanMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[List[List[Tuple]]] = None,
        golds: Optional[List[List[Tuple]]] = None,
        reverse: bool = False,
        eps: float = 1e-12
    ) -> SpanMetric:
        super().__init__(reverse=reverse, eps=eps)

        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0

        if loss is not None:
            self(loss, preds, golds)

    def __call__(
        self,
        loss: float,
        preds: List[List[Tuple]],
        golds: List[List[Tuple]]
    ) -> SpanMetric:
        self.n += len(preds)
        self.count += 1
        self.total_loss += float(loss)
        for pred, gold in zip(preds, golds):
            upred, ugold = Counter([tuple(span[:-1]) for span in pred]), Counter([tuple(span[:-1]) for span in gold])
            lpred, lgold = Counter([tuple(span) for span in pred]), Counter([tuple(span) for span in gold])
            utp, ltp = list((upred & ugold).elements()), list((lpred & lgold).elements())
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)
        return self

    def __add__(self, other: SpanMetric) -> SpanMetric:
        metric = SpanMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss
        metric.n_ucm = self.n_ucm + other.n_ucm
        metric.n_lcm = self.n_lcm + other.n_lcm
        metric.utp = self.utp + other.utp
        metric.ltp = self.ltp + other.ltp
        metric.pred = self.pred + other.pred
        metric.gold = self.gold + other.gold
        metric.reverse = self.reverse or other.reverse
        return metric

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

    @property
    def values(self) -> Dict:
        return {'UCM': self.ucm,
                'LCM': self.lcm,
                'UP': self.up,
                'UR': self.ur,
                'UF': self.uf,
                'LP': self.lp,
                'LR': self.lr,
                'LF': self.lf}


class DiscontinuousSpanMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[List[List[Tuple]]] = None,
        golds: Optional[List[List[Tuple]]] = None,
        param: Optional[str] = None,
        reverse: bool = False,
        eps: float = 1e-12
    ) -> DiscontinuousSpanMetric:
        super().__init__(reverse=reverse, eps=eps)

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.dtp = 0.0
        self.dpred = 0.0
        self.dgold = 0.0

        if loss is not None:
            self(loss, preds, golds, param)

    def __call__(
        self,
        loss: float,
        preds: List[List[Tuple]],
        golds: List[List[Tuple]],
        param: str = None
    ) -> DiscontinuousSpanMetric:
        self.n += len(preds)
        self.count += 1
        self.total_loss += float(loss)
        with tempfile.TemporaryDirectory() as ftemp:
            fpred, fgold = os.path.join(ftemp, 'pred'), os.path.join(ftemp, 'gold')
            with open(fpred, 'w') as f:
                for pred in preds:
                    f.write(pred.pformat(1000000) + '\n')
            with open(fgold, 'w') as f:
                for gold in golds:
                    f.write(gold.pformat(1000000) + '\n')

            from discodop.eval import Evaluator, readparam
            from discodop.tree import bitfanout
            from discodop.treebank import DiscBracketCorpusReader
            preds = DiscBracketCorpusReader(fpred, encoding='utf8', functions='remove')
            golds = DiscBracketCorpusReader(fgold, encoding='utf8', functions='remove')
            goldtrees, goldsents = golds.trees(), golds.sents()
            candtrees, candsents = preds.trees(), preds.sents()

            evaluator = Evaluator(readparam(param), max(len(str(key)) for key in candtrees))
            for n, ctree in candtrees.items():
                evaluator.add(n, goldtrees[n], goldsents[n], ctree, candsents[n])
            cpreds, cgolds = evaluator.acc.candb, evaluator.acc.goldb
            dpreds, dgolds = (Counter([i for i in c.elements() if bitfanout(i[1][1]) > 1]) for c in (cpreds, cgolds))
            self.tp += sum((cpreds & cgolds).values())
            self.pred += sum(cpreds.values())
            self.gold += sum(cgolds.values())
            self.dtp += sum((dpreds & dgolds).values())
            self.dpred += sum(dpreds.values())
            self.dgold += sum(dgolds.values())
        return self

    def __add__(self, other: DiscontinuousSpanMetric) -> DiscontinuousSpanMetric:
        metric = DiscontinuousSpanMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss
        metric.tp = self.tp + other.tp
        metric.pred = self.pred + other.pred
        metric.gold = self.gold + other.gold
        metric.dtp = self.dtp + other.dtp
        metric.dpred = self.dpred + other.dpred
        metric.dgold = self.dgold + other.dgold
        metric.reverse = self.reverse or other.reverse
        return metric

    @property
    def score(self):
        return self.f

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)

    @property
    def dp(self):
        return self.dtp / (self.dpred + self.eps)

    @property
    def dr(self):
        return self.dtp / (self.dgold + self.eps)

    @property
    def df(self):
        return 2 * self.dtp / (self.dpred + self.dgold + self.eps)

    @property
    def values(self) -> Dict:
        return {'P': self.p,
                'R': self.r,
                'F': self.f,
                'DP': self.dp,
                'DR': self.dr,
                'DF': self.df}


class ChartMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[torch.Tensor] = None,
        golds: Optional[torch.Tensor] = None,
        reverse: bool = False,
        eps: float = 1e-12
    ) -> ChartMetric:
        super().__init__(reverse=reverse, eps=eps)

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0

        if loss is not None:
            self(loss, preds, golds)

    def __call__(
        self,
        loss: float,
        preds: torch.Tensor,
        golds: torch.Tensor
    ) -> ChartMetric:
        self.n += len(preds)
        self.count += 1
        self.total_loss += float(loss)
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()
        return self

    def __add__(self, other: ChartMetric) -> ChartMetric:
        metric = ChartMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss
        metric.tp = self.tp + other.tp
        metric.utp = self.utp + other.utp
        metric.pred = self.pred + other.pred
        metric.gold = self.gold + other.gold
        metric.reverse = self.reverse or other.reverse
        return metric

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

    @property
    def values(self) -> Dict:
        return {'UP': self.up,
                'UR': self.ur,
                'UF': self.uf,
                'P': self.p,
                'R': self.r,
                'F': self.f}
