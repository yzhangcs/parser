# -*- coding: utf-8 -*-

import os
from typing import Dict, Iterable, Set, Union

import torch
from supar.models.const.crf.model import CRFConstituencyModel
from supar.parser import Parser
from supar.structs import ConstituencyCRF
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, EOS, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.logging import get_logger
from supar.utils.metric import SpanMetric
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch, Tree

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
        mbr: bool = True,
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
        mbr: bool = True,
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
        mbr: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        return super().predict(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, _, charts = batch
        mask = batch.mask[:, 1:]
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
        s_span, s_label = self.model(words, feats)
        loss, _ = self.model.loss(s_span, s_label, charts, mask, self.args.mbr)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> SpanMetric:
        words, *feats, trees, charts = batch
        mask = batch.mask[:, 1:]
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
        s_span, s_label = self.model(words, feats)
        loss, s_span = self.model.loss(s_span, s_label, charts, mask, self.args.mbr)
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
        s_span, s_label = self.model(words, feats)
        s_span = ConstituencyCRF(s_span, mask[:, 0].sum(-1)).marginals if self.args.mbr else s_span
        chart_preds = self.model.decode(s_span, s_label, mask)
        batch.trees = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                       for tree, chart in zip(trees, chart_preds)]
        if self.args.prob:
            batch.probs = [prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span)]
        return batch

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
            parser.model.load_pretrained(parser.transform.WORD[0].embed).to(parser.device)
            return parser

        logger.info("Building the fields")
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
