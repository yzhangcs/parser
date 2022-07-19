# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import (BiaffineDependencyModel, CRF2oDependencyModel,
                          CRFDependencyModel, VIDependencyModel)
from supar.parsers.parser import Parser
from supar.structs import Dependency2oCRF, DependencyCRF, MatrixTree
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import AttachmentMetric
from supar.utils.parallel import parallel, sync
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import CoNLL

logger = get_logger(__name__)


class BiaffineDependencyParser(Parser):
    r"""
    The implementation of Biaffine Dependency Parser :cite:`dozat-etal-2017-biaffine`.
    """

    NAME = 'biaffine-dependency'
    MODEL = BiaffineDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.TAG = self.transform.CPOS
        self.ARC, self.REL = self.transform.HEAD, self.transform.DEPREL

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              punct=False, tree=False, proj=False, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (str or Iterable):
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
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False,
                 punct=False, tree=True, proj=False, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
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
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, prob=False,
                tree=True, proj=False, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
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
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @classmethod
    def load(cls, path, reload=False, src='github', **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'biaffine-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            src (str):
                Specifies where to download the model.
                ``'github'``: github release page.
                ``'hlt'``: hlt homepage, only accessible from 9:00 to 18:00 (UTC+8).
                Default: ``'github'``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs and initializing the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('biaffine-dep-en')
            >>> parser = Parser.load('./ptb.biaffine.dep.lstm.char')
        """

        return super().load(path, reload, src, **kwargs)

    @parallel()
    def _train(self, loader):
        bar, metric = progress_bar(loader), AttachmentMetric()

        for i, batch in enumerate(bar, 1):
            words, texts, *feats, arcs, rels = batch
            mask = batch.mask
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with sync(self.model, i % self.args.update_steps == 0):
                with torch.autocast(self.device, enabled=self.args.amp):
                    s_arc, s_rel = self.model(words, feats)
                    loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.partial)
                    loss = loss / self.args.update_steps
                self.scaler.scale(loss).backward()
            if i % self.args.update_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
            metric += AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - {metric}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = AttachmentMetric()

        for batch in progress_bar(loader):
            words, texts, *feats, arcs, rels = batch
            mask = batch.mask
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_arc, s_rel = self.model(words, feats)
                loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
            metric += AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        for batch in progress_bar(loader):
            words, texts, *feats = batch
            mask, lens = batch.mask, (batch.lens - 1).tolist()
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_arc, s_rel = self.model(words, feats)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
            batch.arcs = [i.tolist() for i in arc_preds[mask].split(lens)]
            batch.rels = [self.REL.vocab[i.tolist()] for i in rel_preds[mask].split(lens)]
            if self.args.prob:
                batch.probs = [prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, s_arc.softmax(-1).unbind())]
            yield from batch.sentences

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary.
                Required if taking words as encoder input.
                Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(parser.device)
            return parser

        logger.info("Building the fields")
        TAG, CHAR, ELMO, BERT = None, None, None, None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=args.fix_len)
            if 'elmo' in args.feat:
                from allennlp.modules.elmo import batch_to_ids
                ELMO = RawField('elmo')
                ELMO.compose = lambda x: batch_to_ids(x).to(WORD.device)
            if 'bert' in args.feat:
                t = TransformerTokenizer(args.bert)
                BERT = SubwordField('bert', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
                BERT.vocab = t.vocab
        TEXT = RawField('texts')
        ARC = Field('arcs', bos=BOS, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=BOS)
        transform = CoNLL(FORM=(WORD, TEXT, CHAR, ELMO, BERT), CPOS=TAG, HEAD=ARC, DEPREL=REL)

        train = Dataset(transform, args.train, **args)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
        REL.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser


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
            train/dev/test (str or Iterable):
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
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False,
                 punct=False, mbr=True, tree=True, proj=True, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
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
            kwargs (dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, prob=False,
                mbr=True, tree=True, proj=True, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
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
            kwargs (dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @classmethod
    def load(cls, path, reload=False, src='github', **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'crf-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            src (str):
                Specifies where to download the model.
                ``'github'``: github release page.
                ``'hlt'``: hlt homepage, only accessible from 9:00 to 18:00 (UTC+8).
                Default: ``'github'``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs and initializing the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('crf-dep-en')
            >>> parser = Parser.load('./ptb.crf.dep.lstm.char')
        """

        return super().load(path, reload, src, **kwargs)

    @parallel()
    def _train(self, loader):
        bar, metric = progress_bar(loader), AttachmentMetric()

        for i, batch in enumerate(bar, 1):
            words, texts, *feats, arcs, rels = batch
            mask = batch.mask
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with sync(self.model, i % self.args.update_steps == 0):
                with torch.autocast(self.device, enabled=self.args.amp):
                    s_arc, s_rel = self.model(words, feats)
                    loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.mbr, self.args.partial)
                    loss = loss / self.args.update_steps
                self.scaler.scale(loss).backward()
            if i % self.args.update_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
            metric += AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - {metric}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = AttachmentMetric()

        for batch in progress_bar(loader):
            words, texts, *feats, arcs, rels = batch
            mask = batch.mask
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_arc, s_rel = self.model(words, feats)
                loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.mbr, self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
            metric += AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        CRF = DependencyCRF if self.args.proj else MatrixTree
        for batch in progress_bar(loader):
            words, _, *feats = batch
            mask, lens = batch.mask, batch.lens - 1
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_arc, s_rel = self.model(words, feats)
                s_arc = CRF(s_arc, lens).marginals if self.args.mbr else s_arc
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
            lens = lens.tolist()
            batch.arcs = [i.tolist() for i in arc_preds[mask].split(lens)]
            batch.rels = [self.REL.vocab[i.tolist()] for i in rel_preds[mask].split(lens)]
            if self.args.prob:
                arc_probs = s_arc if self.args.mbr else s_arc.softmax(-1)
                batch.probs = [prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, arc_probs.unbind())]
            yield from batch.sentences


class CRF2oDependencyParser(BiaffineDependencyParser):
    r"""
    The implementation of second-order CRF Dependency Parser :cite:`zhang-etal-2020-efficient`.
    """

    NAME = 'crf2o-dependency'
    MODEL = CRF2oDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              punct=False, mbr=True, tree=False, proj=False, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (str or Iterable):
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
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False,
                 punct=False, mbr=True, tree=True, proj=True, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
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
            kwargs (dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, prob=False,
                mbr=True, tree=True, proj=True, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
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
            kwargs (dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @classmethod
    def load(cls, path, reload=False, src='github', **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'crf2o-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            src (str):
                Specifies where to download the model.
                ``'github'``: github release page.
                ``'hlt'``: hlt homepage, only accessible from 9:00 to 18:00 (UTC+8).
                Default: ``'github'``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs and initializing the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('crf2o-dep-en')
            >>> parser = Parser.load('./ptb.crf2o.dep.lstm.char')
        """

        return super().load(path, reload, src, **kwargs)

    @parallel()
    def _train(self, loader):
        bar, metric = progress_bar(loader), AttachmentMetric()

        for i, batch in enumerate(bar, 1):
            words, texts, *feats, arcs, sibs, rels = batch
            mask = batch.mask
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with sync(self.model, i % self.args.update_steps == 0):
                with torch.autocast(self.device, enabled=self.args.amp):
                    s_arc, s_sib, s_rel = self.model(words, feats)
                    loss, s_arc, s_sib = self.model.loss(s_arc, s_sib, s_rel, arcs, sibs, rels, mask,
                                                         self.args.mbr, self.args.partial)
                    loss = loss / self.args.update_steps
                self.scaler.scale(loss).backward()
            if i % self.args.update_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

            arc_preds, rel_preds = self.model.decode(s_arc, s_sib, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
            metric += AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - {metric}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = AttachmentMetric()

        for batch in progress_bar(loader):
            words, texts, *feats, arcs, sibs, rels = batch
            mask = batch.mask
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_arc, s_sib, s_rel = self.model(words, feats)
                loss, s_arc, s_sib = self.model.loss(s_arc, s_sib, s_rel, arcs, sibs, rels, mask,
                                                     self.args.mbr, self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_sib, s_rel, mask, self.args.tree, self.args.mbr, self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
            metric += AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        for batch in progress_bar(loader):
            words, texts, *feats = batch
            mask, lens = batch.mask, batch.lens - 1
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_arc, s_sib, s_rel = self.model(words, feats)
                s_arc, s_sib = Dependency2oCRF((s_arc, s_sib), lens).marginals if self.args.mbr else (s_arc, s_sib)
            arc_preds, rel_preds = self.model.decode(s_arc, s_sib, s_rel, mask, self.args.tree, self.args.mbr, self.args.proj)
            lens = lens.tolist()
            batch.arcs = [i.tolist() for i in arc_preds[mask].split(lens)]
            batch.rels = [self.REL.vocab[i.tolist()] for i in rel_preds[mask].split(lens)]
            if self.args.prob:
                arc_probs = s_arc if self.args.mbr else s_arc.softmax(-1)
                batch.probs = [prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, arc_probs.unbind())]
            yield from batch.sentences

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(parser.device)
            return parser

        logger.info("Building the fields")
        TAG, CHAR, ELMO, BERT = None, None, None, None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=args.fix_len)
            if 'elmo' in args.feat:
                from allennlp.modules.elmo import batch_to_ids
                ELMO = RawField('elmo')
                ELMO.compose = lambda x: batch_to_ids(x).to(WORD.device)
            if 'bert' in args.feat:
                t = TransformerTokenizer(args.bert)
                BERT = SubwordField('bert', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
                BERT.vocab = t.vocab
        TEXT = RawField('texts')
        ARC = Field('arcs', bos=BOS, use_vocab=False, fn=CoNLL.get_arcs)
        SIB = ChartField('sibs', bos=BOS, use_vocab=False, fn=CoNLL.get_sibs)
        REL = Field('rels', bos=BOS)
        transform = CoNLL(FORM=(WORD, TEXT, CHAR, ELMO, BERT), CPOS=TAG, HEAD=(ARC, SIB), DEPREL=REL)

        train = Dataset(transform, args.train, **args)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
        REL.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser


class VIDependencyParser(BiaffineDependencyParser):
    r"""
    The implementation of Dependency Parser using Variational Inference :cite:`wang-tu-2020-second`.
    """

    NAME = 'vi-dependency'
    MODEL = VIDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              punct=False, tree=False, proj=False, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (str or Iterable):
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
            punct (bool):
                If ``False``, ignores the punctuation during evaluation. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False,
                 punct=False, tree=True, proj=True, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
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
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, prob=False,
                tree=True, proj=True, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
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
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @classmethod
    def load(cls, path, reload=False, src='github', **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'vi-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            src (str):
                Specifies where to download the model.
                ``'github'``: github release page.
                ``'hlt'``: hlt homepage, only accessible from 9:00 to 18:00 (UTC+8).
                Default: ``'github'``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs and initializing the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('vi-dep-en')
            >>> parser = Parser.load('./ptb.vi.dep.lstm.char')
        """

        return super().load(path, reload, src, **kwargs)

    @parallel()
    def _train(self, loader):
        bar, metric = progress_bar(loader), AttachmentMetric()

        for i, batch in enumerate(bar, 1):
            words, texts, *feats, arcs, rels = batch
            mask = batch.mask
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with sync(self.model, i % self.args.update_steps == 0):
                with torch.autocast(self.device, enabled=self.args.amp):
                    s_arc, s_sib, s_rel = self.model(words, feats)
                    loss, s_arc = self.model.loss(s_arc, s_sib, s_rel, arcs, rels, mask)
                    loss = loss / self.args.update_steps
                self.scaler.scale(loss).backward()
            if i % self.args.update_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
            metric += AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - {metric}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = AttachmentMetric()

        for batch in progress_bar(loader):
            words, texts, *feats, arcs, rels = batch
            mask = batch.mask
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_arc, s_sib, s_rel = self.model(words, feats)
                loss, s_arc = self.model.loss(s_arc, s_sib, s_rel, arcs, rels, mask)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
            metric += AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        for batch in progress_bar(loader):
            words, texts, *feats = batch
            mask, lens = batch.mask, (batch.lens - 1).tolist()
            # ignore the first token of each sentence
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_arc, s_sib, s_rel = self.model(words, feats)
                s_arc = self.model.inference((s_arc, s_sib), mask)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
            batch.arcs = [i.tolist() for i in arc_preds[mask].split(lens)]
            batch.rels = [self.REL.vocab[i.tolist()] for i in rel_preds[mask].split(lens)]
            if self.args.prob:
                batch.probs = [prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, s_arc.unbind())]
            yield from batch.sentences
