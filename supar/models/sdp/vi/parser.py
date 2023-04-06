# -*- coding: utf-8 -*-

from typing import Iterable, Union

import torch

from supar.models.dep.biaffine.transform import CoNLL
from supar.models.sdp.biaffine.parser import BiaffineSemanticDependencyParser
from supar.models.sdp.vi.model import VISemanticDependencyModel
from supar.utils import Config
from supar.utils.logging import get_logger
from supar.utils.metric import ChartMetric
from supar.utils.transform import Batch

logger = get_logger(__name__)


class VISemanticDependencyParser(BiaffineSemanticDependencyParser):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference :cite:`wang-etal-2019-second`.
    """

    NAME = 'vi-semantic-dependency'
    MODEL = VISemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(
        self,
        train: Union[str, Iterable],
        dev: Union[str, Iterable],
        test: Union[str, Iterable],
        epochs: int = 1000,
        patience: int = 100,
        batch_size: int = 5000,
        update_steps: int = 1,
        buckets: int = 32,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
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
        words, *feats, labels = batch
        mask = batch.mask
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0
        s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
        loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd, s_label, labels, mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> ChartMetric:
        words, *feats, labels = batch
        mask = batch.mask
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0
        s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
        loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd, s_label, labels, mask)
        label_preds = self.model.decode(s_edge, s_label)
        return ChartMetric(loss, label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, *feats = batch
        mask, lens = batch.mask, (batch.lens - 1).tolist()
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0
        s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
        s_edge = self.model.inference((s_edge, s_sib, s_cop, s_grd), mask)
        label_preds = self.model.decode(s_edge, s_label).masked_fill(~mask, -1)
        batch.labels = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                                               for row in chart[1:i, :i].tolist()])
                        for i, chart in zip(lens, label_preds)]
        if self.args.prob:
            batch.probs = [prob[1:i, :i].cpu() for i, prob in zip(lens, s_edge.unbind())]
        return batch
