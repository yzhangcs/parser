# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import CRFConstituencyModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, eos, pad, unk
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.logging import logger, progress_bar
from supar.utils.metric import BracketMetric
from supar.utils.transform import Tree


class CRFConstituencyParser(Parser):

    NAME = 'crf-constituency'
    MODEL = CRFConstituencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.WORD
        else:
            self.WORD, self.FEAT = self.transform.WORD, self.transform.POS
        self.TREE = self.transform.TREE
        self.CHART = self.transform.CHART

    def train(self, train, dev, test, buckets=32, mbr=True,
              delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
              equal={'ADVP': 'PRT'},
              ** kwargs,):
        super().train(train, dev, test, buckets, mbr=mbr, delete=delete, equal=equal, **kwargs)

    def evaluate(self, data, buckets=8,  mbr=True,
                 delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                 equal={'ADVP': 'PRT'},
                 ** kwargs,):
        return super().evaluate(data, buckets, mbr=mbr, delete=delete, equal=equal, **kwargs)

    def predict(self, data, pred=None, buckets=8, prob=False, mbr=True, **kwargs):
        return super().predict(data, pred, buckets, prob, mbr=mbr, **kwargs)

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for words, feats, trees, (spans, labels) in bar:
            self.optimizer.zero_grad()

            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            loss, _ = self.model.loss(s_span, s_label, spans, labels, mask, self.args.mbr)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            bar.set_postfix_str(f"lr: {self.scheduler.get_lr()[0]:.4e} - "
                                f"loss: {loss:.4f}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, BracketMetric()

        for words, feats, trees, (spans, labels) in loader:
            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            loss, s_span = self.model.loss(s_span, s_label, spans, labels, mask, self.args.mbr)
            chart_preds = self.model.decode(s_span, s_label, mask)
            # since the evaluation relies on terminals,
            # the tree should be first built and then factorized
            preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                     for tree, chart in zip(trees, chart_preds)]
            total_loss += loss.item()
            metric([Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                   [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds, probs = {'trees': []}, []

        for words, feats, trees in progress_bar(loader):
            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            if self.args.mbr:
                s_span = self.model.crf(s_span, mask, mbr=True)
            chart_preds = self.model.decode(s_span, s_label, mask)
            preds['trees'].extend([Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                                   for tree, chart in zip(trees, chart_preds)])
            if self.args.prob:
                probs.extend(s_span.tolist())
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, verbose=True, **kwargs):
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Build the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, eos=eos, fix_len=args.fix_len)
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.cls_token or tokenizer.cls_token,
                                eos=tokenizer.sep_token or tokenizer.sep_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        else:
            FEAT = Field('tags', bos=bos, eos=eos)
        TREE = RawField('trees')
        CHART = ChartField('charts')
        if args.feat in ('char', 'bert'):
            transform = Tree(WORD=(WORD, FEAT), TREE=TREE, CHART=CHART)
        else:
            transform = Tree(WORD=WORD, POS=FEAT, TREE=TREE, CHART=CHART)

        train = Dataset(transform, args.train)
        embed = None
        if args.embed:
            embed = Embedding.load(args.embed, args.unk)
        WORD.build(train, args.min_freq, embed)
        FEAT.build(train)
        CHART.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_labels': len(CHART.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'eos_index': WORD.eos_index,
            'feat_pad_index': FEAT.pad_index
        })
        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform, verbose)
