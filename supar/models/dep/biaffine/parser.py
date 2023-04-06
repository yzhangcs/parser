# -*- coding: utf-8 -*-

import os
from typing import Iterable, Union

import torch

from supar.models.dep.biaffine.model import BiaffineDependencyModel
from supar.models.dep.biaffine.transform import CoNLL
from supar.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, PAD, UNK
from supar.utils.field import Field, RawField, SubwordField
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger
from supar.utils.metric import AttachmentMetric
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch

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
        proj: bool = False,
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
        proj: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().predict(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, _, *feats, arcs, rels = batch
        mask = batch.mask
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_rel = self.model(words, feats)
        loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.partial)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> AttachmentMetric:
        words, _, *feats, arcs, rels = batch
        mask = batch.mask
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_rel = self.model(words, feats)
        loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.partial)
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
        s_arc, s_rel = self.model(words, feats)
        arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
        batch.arcs = [i.tolist() for i in arc_preds[mask].split(lens)]
        batch.rels = [self.REL.vocab[i.tolist()] for i in rel_preds[mask].split(lens)]
        if self.args.prob:
            batch.probs = [prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, s_arc.softmax(-1).unbind())]
        return batch

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
            kwargs (Dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.transform.FORM[0].embed).to(parser.device)
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
