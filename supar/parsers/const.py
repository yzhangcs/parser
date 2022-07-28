# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import CRFConstituencyModel, VIConstituencyModel
from supar.parsers.parser import Parser
from supar.structs import ConstituencyCRF
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, EOS, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import SpanMetric
from supar.utils.parallel import parallel, sync
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Tree

logger = get_logger(__name__)


class CRFConstituencyParser(Parser):
    r"""
    The implementation of CRF Constituency Parser :cite:`zhang-etal-2020-fast`.
    """

    NAME = 'crf-constituency'
    MODEL = CRFConstituencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.TREE = self.transform.TREE
        self.CHART = self.transform.CHART

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              mbr=True,
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
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
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
                 mbr=True,
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
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
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
                mbr=True, verbose=True, **kwargs):
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
    def _train(self, loader):
        bar = progress_bar(loader)

        for i, batch in enumerate(bar, 1):
            words, *feats, trees, charts = batch
            mask = batch.mask[:, 1:]
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            with sync(self.model, i % self.args.update_steps == 0):
                with torch.autocast(self.device, enabled=self.args.amp):
                    s_span, s_label = self.model(words, feats)
                    loss, _ = self.model.loss(s_span, s_label, charts, mask, self.args.mbr)
                    loss = loss / self.args.update_steps
                self.scaler.scale(loss).backward()
            if i % self.args.update_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = SpanMetric()

        for batch in progress_bar(loader):
            words, *feats, trees, charts = batch
            mask = batch.mask[:, 1:]
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            with torch.autocast(self.device, enabled=self.args.amp):
                s_span, s_label = self.model(words, feats)
                loss, s_span = self.model.loss(s_span, s_label, charts, mask, self.args.mbr)
            chart_preds = self.model.decode(s_span, s_label, mask)
            # since the evaluation relies on terminals,
            # the tree should be first built and then factorized
            preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                     for tree, chart in zip(trees, chart_preds)]
            metric += SpanMetric(loss,
                                 [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                                 [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        for batch in progress_bar(loader):
            words, *feats, trees = batch
            mask, lens = batch.mask[:, 1:], batch.lens - 2
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            with torch.autocast(self.device, enabled=self.args.amp):
                s_span, s_label = self.model(words, feats)
                s_span = ConstituencyCRF(s_span, mask[:, 0].sum(-1)).marginals if self.args.mbr else s_span
            chart_preds = self.model.decode(s_span, s_label, mask)
            batch.trees = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                           for tree, chart in zip(trees, chart_preds)]
            if self.args.prob:
                batch.probs = [prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span)]
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
            kwargs (Dict):
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
        WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True)
        TAG, CHAR, ELMO, BERT = None, None, None, None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, eos=t.eos, fix_len=args.fix_len, tokenize=t)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS, eos=EOS)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, eos=EOS, fix_len=args.fix_len)
            if 'elmo' in args.feat:
                from allennlp.modules.elmo import batch_to_ids
                ELMO = RawField('elmo')
                ELMO.compose = lambda x: batch_to_ids(x).to(WORD.device)
            if 'bert' in args.feat:
                t = TransformerTokenizer(args.bert)
                BERT = SubwordField('bert', pad=t.pad, unk=t.unk, bos=t.bos, eos=t.eos, fix_len=args.fix_len, tokenize=t)
                BERT.vocab = t.vocab
        TREE = RawField('trees')
        CHART = ChartField('charts')
        transform = Tree(WORD=(WORD, CHAR, ELMO, BERT), POS=TAG, TREE=TREE, CHART=CHART)

        train = Dataset(transform, args.train, **args)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
        CHART.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_labels': len(CHART.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'eos_index': WORD.eos_index
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser


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
    def _train(self, loader):
        bar = progress_bar(loader)

        for i, batch in enumerate(bar, 1):
            words, *feats, trees, charts = batch
            mask = batch.mask[:, 1:]
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            with sync(self.model, i % self.args.update_steps == 0):
                with torch.autocast(self.device, enabled=self.args.amp):
                    s_span, s_pair, s_label = self.model(words, feats)
                    loss, _ = self.model.loss(s_span, s_pair, s_label, charts, mask)
                    loss = loss / self.args.update_steps
                self.scaler.scale(loss).backward()
            if i % self.args.update_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = SpanMetric()

        for batch in progress_bar(loader):
            words, *feats, trees, charts = batch
            mask = batch.mask[:, 1:]
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            with torch.autocast(self.device, enabled=self.args.amp):
                s_span, s_pair, s_label = self.model(words, feats)
                loss, s_span = self.model.loss(s_span, s_pair, s_label, charts, mask)
            chart_preds = self.model.decode(s_span, s_label, mask)
            # since the evaluation relies on terminals,
            # the tree should be first built and then factorized
            preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                     for tree, chart in zip(trees, chart_preds)]
            metric += SpanMetric(loss,
                                 [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                                 [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        for batch in progress_bar(loader):
            words, *feats, trees = batch
            mask, lens = batch.mask[:, 1:], batch.lens - 2
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            with torch.autocast(self.device, enabled=self.args.amp):
                s_span, s_pair, s_label = self.model(words, feats)
                s_span = self.model.inference((s_span, s_pair), mask)
            chart_preds = self.model.decode(s_span, s_label, mask)
            batch.trees = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                           for tree, chart in zip(trees, chart_preds)]
            if self.args.prob:
                batch.probs = [prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span)]
            yield from batch.sentences
