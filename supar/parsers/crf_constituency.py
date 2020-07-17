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
    """
    The implementation of CRF Constituency Parser.
    This parser is also called FANCY (abbr. of Fast and Accurate Neural Crf constituencY) Parser.

    References:
    - Yu Zhang, houquan Zhou and Zhenghua Li (IJCAI'20)
      Fast and Accurate Neural CRF Constituency Parsing
      https://www.ijcai.org/Proceedings/2020/560/
    """

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

    def train(self, train, dev, test, buckets=32, batch_size=5000, mbr=True,
              delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
              equal={'ADVP': 'PRT'},
              **kwargs):
        r"""the function of training

        Args:
            train, dev, test (List[List] or str):
                the train\dev\test data, input as a filename 'str' or a list of filenames (required).
            buckets (int, default: 32):
                The dict that maps each centroid to the indices of the clustering sentences.
                The centroid corresponds to the average length of all sentences in the bucket (optional).
            batch_size (int, default: 5000): batch size in training (optional).
            mbr (bool, default: True): using Minimum Bayes Risk decoding (optional).
            delete (Set[str], default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}):
                a set of labels need to delete from evaluation, those labels may be predict but evaluate.
                the delete set corresponds to the wildly used PTB dataset (optional).
            equal (Dict[str, str], default: {'ADVP': 'PRT'}):
                labels that should be treat as same label.
                the delete equal map corresponds to the wildly used PTB dataset (optional).
            kwargs: other args (optional).

        Examples:
            >>> parser.train(train_filename, dev_filename, test_filename)
        """
        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, mbr=True,
                 delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                 equal={'ADVP': 'PRT'},
                 **kwargs):
        r"""the function of evaluating

        Args:
            data (List[List] or str):
                the data you want to evaluate, input as a filename 'str' or a list of filenames (required).
            buckets (int, default: 32):
                The dict that maps each centroid to the indices of the clustering sentences.
                The centroid corresponds to the average length of all sentences in the bucket (optional).
            batch_size (int, default: 5000): batch size in training (optional).
            mbr (bool, default: True): using Minimum Bayes Risk decoding (optional).
            delete (Set, default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}):
                a set of labels need to delete from evaluation, those labels may be predict but evaluate.
                the delete set corresponds to the wildly used PTB dataset (optional).
            equal (Dict, default: {'ADVP': 'PRT'}):
                labels that should be treat as same label.
                the delete equal map corresponds to the wildly used PTB dataset (optional).
            kwargs: other args (optional).

        Examples:
            >>> parser.evaluate(data_filename)
        """
        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False, mbr=True, **kwargs):
        r"""the function of predicting

        Args:
            data (List[List] or str):
                the data you want to predict, input as a filename 'str' or a list of filenames (required).
            pred (str, default: None):
                the output file name that you want to save the predict results (optional).
            buckets (int, default: 32):
                The dict that maps each centroid to the indices of the clustering sentences.
                The centroid corresponds to the average length of all sentences in the bucket (optional).
            batch_size (int, default: 5000): batch size in training (optional).
            prob (bool, default: False): whether output the probability or not (optional).
            mbr (bool, default: True): using Minimum Bayes Risk decoding (optional).
            kwargs: other args (optional).

        Examples:
            >>> parser.predict(data_filename, pred=output_filename)
        """
        return super().predict(**Config().update(locals()))

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

            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - "
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
        r"""The first choice to build a brand-new Parser

        Args:
            path (str):
                the path you want to save your model (required).
            min_freq (str, default: 2):
                the minimum freqency of words.
                words appear less than it will be ignored (optional).
            fix_len (int, default: 20):
                A fixed length that all subword pieces will be padded to.
                This is used for truncating the subword pieces that exceed the length.
                To save the memory, the final length will be the smaller value
                between the max length of subword pieces in a batch and fix_len (optional).
            verbose (bool, default: True): if you don't want the verbose logging, set it to False (optional).
            kwargs: other args (optional).

        Examples:
            >>> parser.build(model_path)
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
        WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
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
