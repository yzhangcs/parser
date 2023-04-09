# -*- coding: utf-8 -*-

import os
from typing import Dict, Iterable, Set, Union

import torch

from supar.models.const.tt.model import TetraTaggingConstituencyModel
from supar.models.const.tt.transform import TetraTaggingTree
from supar.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, EOS, PAD, UNK
from supar.utils.field import Field, RawField, SubwordField
from supar.utils.logging import get_logger
from supar.utils.metric import SpanMetric
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch

logger = get_logger(__name__)


class TetraTaggingConstituencyParser(Parser):
    r"""
    The implementation of TetraTagging Constituency Parser :cite:`kitaev-klein-2020-tetra`.
    """

    NAME = 'tetra-tagging-constituency'
    MODEL = TetraTaggingConstituencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.TREE = self.transform.TREE
        self.LEAF = self.transform.LEAF
        self.NODE = self.transform.NODE

        self.left_mask = torch.tensor([*(i.startswith('l') for i in self.LEAF.vocab.itos),
                                       *(i.startswith('L') for i in self.NODE.vocab.itos)]).to(self.device)

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
        depth: int = 1,
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
        depth: int = 1,
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
        depth: int = 1,
        verbose: bool = True,
        **kwargs
    ):
        return super().predict(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, _, leaves, nodes = batch
        mask = batch.mask[:, 2:]
        s_leaf, s_node = self.model(words, feats)
        loss = self.model.loss(s_leaf, s_node, leaves, nodes, mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> SpanMetric:
        words, *feats, trees, leaves, nodes = batch
        mask = batch.mask[:, 2:]
        s_leaf, s_node = self.model(words, feats)
        loss = self.model.loss(s_leaf, s_node, leaves, nodes, mask)
        preds = self.model.decode(s_leaf, s_node, mask, self.left_mask, self.args.depth)
        preds = [TetraTaggingTree.action2tree(tree, (self.LEAF.vocab[i], self.NODE.vocab[j] if len(j) > 0 else []))
                 for tree, i, j in zip(trees, *preds)]
        return SpanMetric(loss,
                          [TetraTaggingTree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                          [TetraTaggingTree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, *feats, trees = batch
        mask = batch.mask[:, 2:]
        s_leaf, s_node = self.model(words, feats)
        preds = self.model.decode(s_leaf, s_node, mask, self.left_mask, self.args.depth)
        batch.trees = [TetraTaggingTree.action2tree(tree, (self.LEAF.vocab[i], self.NODE.vocab[j] if len(j) > 0 else []))
                       for tree, i, j in zip(trees, *preds)]
        if self.args.prob:
            raise NotImplementedError("Returning action probs are currently not supported yet.")
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
        LEAF, NODE = Field('leaf'), Field('node')
        transform = TetraTaggingTree(WORD=(WORD, CHAR, ELMO, BERT), POS=TAG, TREE=TREE, LEAF=LEAF, NODE=NODE)

        train = Dataset(transform, args.train, **args)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
        LEAF, NODE = LEAF.build(train), NODE.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_leaves': len(LEAF.vocab),
            'n_nodes': len(NODE.vocab),
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
