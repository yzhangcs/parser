# -*- coding: utf-8 -*-

from typing import Dict, Iterable, Set, Union

import torch

from supar.models.const.crf.parser import CRFConstituencyParser
from supar.models.const.crf.transform import Tree
from supar.models.const.vi.model import VIConstituencyModel
from supar.utils import Config
from supar.utils.logging import get_logger
from supar.utils.metric import SpanMetric
from supar.utils.transform import Batch

logger = get_logger(__name__)


class VIConstituencyParser(CRFConstituencyParser):
    r"""
    The implementation of Constituency Parser using variational inference.
    """

    NAME = 'vi-constituency'
    MODEL = VIConstituencyModel

    def train(
        self,
        train,
        dev,
        test,
        epochs: int = 1000,
        patience: int = 100,
        batch_size: int = 5000,
        update_steps: int = 1,
        buckets: int = 32, workers: int = 0, amp: bool = False, cache: bool = False,
        delete: Set = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
        equal: Dict = {'ADVP': 'PRT'},
        verbose: bool = True,
        **kwargs
    ):
        return super().train(**Config().update(locals()))

    def evaluate(
        self,
        data: Union[str, Iterable],
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        delete: Set = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
        equal: Dict = {'ADVP': 'PRT'},
        verbose: bool = True,
        **kwargs
    ):
        return super().evaluate(**Config().update(locals()))

    def predict(
        self,
        data: Union[str, Iterable],
        pred: str = None,
        lang: str = None,
        prob: bool = False,
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().predict(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, _, charts = batch
        mask = batch.mask[:, 1:]
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
        s_span, s_pair, s_label = self.model(words, feats)
        loss, _ = self.model.loss(s_span, s_pair, s_label, charts, mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> SpanMetric:
        words, *feats, trees, charts = batch
        mask = batch.mask[:, 1:]
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
        s_span, s_pair, s_label = self.model(words, feats)
        loss, s_span = self.model.loss(s_span, s_pair, s_label, charts, mask)
        chart_preds = self.model.decode(s_span, s_label, mask)
        preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                 for tree, chart in zip(trees, chart_preds)]
        return SpanMetric(loss,
                          [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                          [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, *feats, trees = batch
        mask, lens = batch.mask[:, 1:], batch.lens - 2
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
        s_span, s_pair, s_label = self.model(words, feats)
        s_span = self.model.inference((s_span, s_pair), mask)
        chart_preds = self.model.decode(s_span, s_label, mask)
        batch.trees = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                       for tree, chart in zip(trees, chart_preds)]
        if self.args.prob:
            batch.probs = [prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span)]
        return batch
