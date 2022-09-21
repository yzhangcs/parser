# -*- coding: utf-8 -*-

import torch
from supar.models.dep.biaffine.parser import BiaffineDependencyParser
from supar.models.dep.crf.model import CRFDependencyModel
from supar.structs import DependencyCRF, MatrixTree
from supar.utils import Config
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger
from supar.utils.metric import AttachmentMetric
from supar.utils.parallel import parallel
from supar.utils.transform import Batch

logger = get_logger(__name__)


class CRFDependencyParser(BiaffineDependencyParser):
    r"""
    The implementation of first-order CRF Dependency Parser :cite:`zhang-etal-2020-efficient`.
    """

    NAME = 'crf-dependency'
    MODEL = CRFDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              punct=False, mbr=True, tree=False, proj=False, partial=False, verbose=True, **kwargs):
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
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            punct (bool):
                If ``False``, ignores the punctuation during evaluation. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False,
                 punct=False, mbr=True, tree=True, proj=True, partial=False, verbose=True, **kwargs):
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
            punct (bool):
                If ``False``, ignores the punctuation during evaluation. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, prob=False,
                mbr=True, tree=True, proj=True, verbose=True, **kwargs):
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
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
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
        words, _, *feats, arcs, rels = batch
        mask = batch.mask
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_rel = self.model(words, feats)
        loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.mbr, self.args.partial)
        return loss

    @parallel(training=False)
    def eval_step(self, batch: Batch) -> AttachmentMetric:
        words, _, *feats, arcs, rels = batch
        mask = batch.mask
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_rel = self.model(words, feats)
        loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.mbr, self.args.partial)
        arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
        if self.args.partial:
            mask &= arcs.ge(0)
        # ignore all punctuation if not specified
        if not self.args.punct:
            mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
        return AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)

    @parallel(training=False, op=None)
    def pred_step(self, batch: Batch) -> Batch:
        CRF = DependencyCRF if self.args.proj else MatrixTree
        words, _, *feats = batch
        mask, lens = batch.mask, batch.lens - 1
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_rel = self.model(words, feats)
        s_arc = CRF(s_arc, lens).marginals if self.args.mbr else s_arc
        arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
        lens = lens.tolist()
        batch.arcs = [i.tolist() for i in arc_preds[mask].split(lens)]
        batch.rels = [self.REL.vocab[i.tolist()] for i in rel_preds[mask].split(lens)]
        if self.args.prob:
            arc_probs = s_arc if self.args.mbr else s_arc.softmax(-1)
            batch.probs = [prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, arc_probs.unbind())]
        return batch
