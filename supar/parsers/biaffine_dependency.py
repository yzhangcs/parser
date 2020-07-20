# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import BiaffineDependencyModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import Field, SubwordField
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import AttachmentMetric
from supar.utils.transform import CoNLL

logger = get_logger(__name__)


class BiaffineDependencyParser(Parser):
    """
    The implementation of Biaffine Dependency Parser.

    References:
    - Timothy Dozat and Christopher D. Manning (ICLR'17)
      Deep Biaffine Attention for Neural Dependency Parsing
      https://openreview.net/pdf?id=Hk95PK9le/
    """

    NAME = 'biaffine-dependency'
    MODEL = BiaffineDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.FORM
        else:
            self.WORD, self.FEAT = self.transform.FORM, self.transform.CPOS
        self.ARC, self.REL = self.transform.HEAD, self.transform.DEPREL
        self.puncts = torch.tensor([i
                                    for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(self.args.device)

    def train(self, train, dev, test, buckets=32, batch_size=5000,
              punct=False, tree=False, proj=False, verbose=True, **kwargs):
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
            tree (bool, default: False):
                If True, ensures to output well-formed trees.
            proj (bool, default: False):
                If True, ensures to output projective trees.
            verbose (bool, default: True):
                If True, increases the output verbosity.
            kwargs (Dict):
                A dict holding the unconsumed arguments.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000,
                 punct=False, tree=True, proj=False, verbose=True, **kwargs):
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
            tree (bool, default: False):
                If True, ensures to output well-formed trees.
            proj (bool, default: False):
                If True, ensures to output projective trees.
            verbose (bool, default: True):
                If True, increases the output verbosity.
            kwargs (Dict):
                A dict holding the unconsumed arguments.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000,
                prob=False, tree=True, proj=False, verbose=True, **kwargs):
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
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
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
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
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
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                arc_probs = s_arc.softmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
                probs.extend(arc_probs[mask].split(lens))
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = [seq.tolist() for seq in probs]

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        """
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str, default: 2):
                The minimum frequency needed to include a token in the vocabulary.
            fix_len (int, default: 20):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
            kwargs (Dict):
                A dict holding the unconsumed arguments.

        Returns:
            The created parser.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Build the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            tokenizer.bos_token = tokenizer.bos_token or tokenizer.cls_token
            tokenizer.eos_token = tokenizer.eos_token or tokenizer.sep_token
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            WORD.bos = FEAT.bos # ensure representations have the same length
            if hasattr(tokenizer, 'vocab'):
                FEAT.vocab = tokenizer.get_vocab()
            else:
                FEAT.vocab = FieldVocab(tokenizer.unk_token_id,
                                        {tokenizer._convert_id_to_token(i): i
                                         for i in range(len(tokenizer))})
        else:
            FEAT = Field('tags', bos=bos)
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        if args.feat in ('char', 'bert'):
            transform = CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL)
        else:
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)

        train = Dataset(transform, args.train)
        WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
        FEAT.build(train)
        REL.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_rels': len(REL.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'feat_pad_index': FEAT.pad_index
        })
        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform)
