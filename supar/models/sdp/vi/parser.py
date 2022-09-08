# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.models.sdp.biaffine.parser import BiaffineSemanticDependencyParser
from supar.models.sdp.vi.model import VISemanticDependencyModel
from supar.utils import Config
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ChartMetric
from supar.utils.parallel import parallel, sync
from supar.utils.transform import CoNLL

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

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (Union[str, Iterable]):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, verbose=True, **kwargs):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for evaluation. Both a filename and a list of instances are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, prob=False,
                verbose=True, **kwargs):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for prediction.
                - a filename. If ends with `.txt`, the parser will seek to make predictions line by line from plain texts.
                - a list of instances.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @parallel()
    def _train(self, loader):
        bar, metric = progress_bar(loader), ChartMetric()

        for i, batch in enumerate(bar, 1):
            words, *feats, labels = batch
            mask = batch.mask
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            with sync(self.model, i % self.args.update_steps == 0):
                with torch.autocast(self.device, enabled=self.args.amp):
                    s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
                    loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd, s_label, labels, mask)
                self.backward(loss)
            if i % self.args.update_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

            label_preds = self.model.decode(s_edge, s_label)
            metric + ChartMetric(loss, label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - {metric}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = ChartMetric()

        for batch in progress_bar(loader):
            words, *feats, labels = batch
            mask = batch.mask
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
                loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd, s_label, labels, mask)
            label_preds = self.model.decode(s_edge, s_label)
            metric += ChartMetric(loss, label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        for batch in progress_bar(loader):
            words, *feats = batch
            mask, lens = batch.mask, (batch.lens - 1).tolist()
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
                s_edge = self.model.inference((s_edge, s_sib, s_cop, s_grd), mask)
            label_preds = self.model.decode(s_edge, s_label).masked_fill(~mask, -1)
            batch.labels = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                                                   for row in chart[1:i, :i].tolist()])
                            for i, chart in zip(lens, label_preds)]
            if self.args.prob:
                batch.probs = [prob[1:i, :i].cpu() for i, prob in zip(lens, s_edge.unbind())]
            yield from batch.sentences
