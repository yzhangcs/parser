# -*- coding: utf-8 -*-

import torch
from supar.models.const.crf.parser import CRFConstituencyParser
from supar.models.const.vi.model import VIConstituencyModel
from supar.utils import Config
from supar.utils.logging import get_logger
from supar.utils.metric import SpanMetric
from supar.utils.parallel import parallel
from supar.utils.transform import Batch, Tree

logger = get_logger(__name__)


class VIConstituencyParser(CRFConstituencyParser):
    r"""
    The implementation of Constituency Parser using variational inference.
    """

    NAME = 'vi-constituency'
    MODEL = VIConstituencyModel

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
              equal={'ADVP': 'PRT'},
              verbose=True,
              **kwargs):
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
            delete (Set[str]):
                A set of labels that will not be taken into consideration during evaluation.
                Default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}.
            equal (Dict[str, str]):
                The pairs in the dict are considered equivalent during evaluation.
                Default: {'ADVP': 'PRT'}.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False,
                 delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                 equal={'ADVP': 'PRT'},
                 verbose=True,
                 **kwargs):
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
            delete (Set[str]):
                A set of labels that will not be taken into consideration during evaluation.
                Default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}.
            equal (Dict[str, str]):
                The pairs in the dict are considered equivalent during evaluation.
                Default: {'ADVP': 'PRT'}.
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
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @parallel()
    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, _, charts = batch
        mask = batch.mask[:, 1:]
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
        s_span, s_pair, s_label = self.model(words, feats)
        loss, _ = self.model.loss(s_span, s_pair, s_label, charts, mask)
        return loss

    @parallel(training=False)
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

    @parallel(training=False, op=None)
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
