# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.models import CRFDependencyModel
from supar.parsers.biaffine_dependency import BiaffineDependencyParser
from supar.utils import Config
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import AttachmentMetric

logger = get_logger(__name__)


class CRFDependencyParser(BiaffineDependencyParser):
    """
    The implementation of first-order CRF Dependency Parser.

    References:
    - Yu Zhang, Zhenghua Li and Min Zhang (ACL'20)
      Efficient Second-Order TreeCRF for Neural Dependency Parsing
      https://www.aclweb.org/anthology/2020.acl-main.302/
    """

    NAME = 'crf-dependency'
    MODEL = CRFDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, train, dev, test, buckets=32, batch_size=5000, punct=False,
              mbr=True, tree=False, proj=False, partial=False, verbose=True, **kwargs):
        """
        Args:
            train, dev, test (List[List] or str):
                the train/dev/test data, both list of instances and filename are allowed.
            buckets (int, default: 32):
                Number of buckets that sentences are assigned to.
            batch_size (int, default: 5000):
                Number of tokens in each batch.
            punct (bool, default: False):
                If False, ignores the punctuations during evaluation.
            mbr (bool, default: True):
                If True, returns marginals for MBR decoding.
            tree (bool, default: False):
                If True, ensures to output well-formed trees.
            proj (bool, default: False):
                If True, ensures to output projective trees.
            partial (bool, default: False):
                True denotes the trees are partially annotated.
            verbose (bool, default: True):
                If True, increases the output verbosity.
            kwargs (Dict):
                A dict holding the unconsumed arguments.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, punct=False,
                 mbr=True, tree=True, proj=True, partial=False, verbose=True, **kwargs):
        """
        Args:
            data (str):
                The data to be evaluated.
            buckets (int, default: 32):
                Number of buckets that sentences are assigned to.
            batch_size (int, default: 5000):
                Number of tokens in each batch.
            punct (bool, default: False):
                If False, ignores the punctuations during evaluation.
            mbr (bool, default: True):
                If True, returns marginals for MBR decoding.
            tree (bool, default: False):
                If True, ensures to output well-formed trees.
            proj (bool, default: False):
                If True, ensures to output projective trees.
            partial (bool, default: False):
                True denotes the trees are partially annotated.
            verbose (bool, default: True):
                If True, increases the output verbosity.
            kwargs (Dict):
                A dict holding the unconsumed arguments.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False,
                mbr=True, tree=True, proj=True, verbose=True, **kwargs):
        """
        Args:
            data (List[List] or str):
                The data to be predicted, both a list of instances and filename are allowed.
            pred (str, default: None):
                If specified, the predicted results will be saved to the file.
            buckets (int, default: 32):
                Number of buckets that sentences are assigned to.
            batch_size (int, default: 5000):
                Number of tokens in each batch.
            prob (bool, default: False):
                If True, outputs the probabilities.
            mbr (bool, default: True):
                If True, returns marginals for MBR decoding.
            tree (bool, default: False):
                If True, ensures to output well-formed trees.
            proj (bool, default: False):
                If True, ensures to output projective trees.
            verbose (bool, default: True):
                If True, increases the output verbosity.
            kwargs (Dict):
                A dict holding the unconsumed arguments.

        Returns:
            A Dataset object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), AttachmentMetric()

        for words, feats, arcs, rels in bar:
            self.optimizer.zero_grad()

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats)
            loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask,
                                          self.args.mbr,
                                          self.args.partial)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, AttachmentMetric()

        for words, feats, arcs, rels in loader:
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats)
            loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask,
                                          self.args.mbr,
                                          self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        arcs, rels, probs = [], [], []
        for words, feats in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(words, feats)
            if self.args.mbr:
                s_arc = self.model.crf(s_arc, mask, mbr=True)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                s_arc = s_arc if self.args.mbr else s_arc.softmax(-1)
                arc_probs = s_arc.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
                probs.extend(arc_probs[mask].split(lens))
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = [seq.tolist() for seq in probs]

        return preds
