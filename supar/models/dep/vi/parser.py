# -*- coding: utf-8 -*-

from typing import Iterable, Union

import torch
from supar.models.dep.biaffine.parser import BiaffineDependencyParser
from supar.models.dep.vi.model import VIDependencyModel
from supar.utils import Config
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger
from supar.utils.metric import AttachmentMetric
from supar.utils.transform import Batch

logger = get_logger(__name__)


class VIDependencyParser(BiaffineDependencyParser):
    r"""
    The implementation of Dependency Parser using Variational Inference :cite:`wang-tu-2020-second`.
    """

    NAME = 'vi-dependency'
    MODEL = VIDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        punct: bool = False,
        tree: bool = False,
        proj: bool = False,
        partial: bool = False,
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
        punct: bool = False,
        tree: bool = True,
        proj: bool = True,
        partial: bool = False,
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
        tree: bool = True,
        proj: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        return super().predict(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, _, *feats, arcs, rels = batch
        mask = batch.mask
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_sib, s_rel = self.model(words, feats)
        loss, *_ = self.model.loss(s_arc, s_sib, s_rel, arcs, rels, mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> AttachmentMetric:
        words, _, *feats, arcs, rels = batch
        mask = batch.mask
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_sib, s_rel = self.model(words, feats)
        loss, s_arc = self.model.loss(s_arc, s_sib, s_rel, arcs, rels, mask)
        arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
        if self.args.partial:
            mask &= arcs.ge(0)
        # ignore all punctuation if not specified
        if not self.args.punct:
            mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
        return AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, _, *feats = batch
        mask, lens = batch.mask, (batch.lens - 1).tolist()
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_sib, s_rel = self.model(words, feats)
        s_arc = self.model.inference((s_arc, s_sib), mask)
        arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
        batch.arcs = [i.tolist() for i in arc_preds[mask].split(lens)]
        batch.rels = [self.REL.vocab[i.tolist()] for i in rel_preds[mask].split(lens)]
        if self.args.prob:
            batch.probs = [prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, s_arc.unbind())]
        return batch
