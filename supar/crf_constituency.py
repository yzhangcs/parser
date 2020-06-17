# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from supar.config import Config
from supar.models import CRFConstituencyModel
from supar.utils import Embedding
from supar.utils.common import bos, eos, pad, unk
from supar.utils.corpus import Treebank, TreebankCorpus
from supar.utils.data import TextDataset, batchify
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.fn import build, factorize
from supar.utils.logging import init_logger, logger, progress_bar
from supar.utils.metric import BracketMetric


class CRFConstituencyParser(object):

    def __init__(self, args, model, fields):
        super(CRFConstituencyParser, self).__init__()

        self.args = args
        self.model = model
        self.fields = fields
        self.TREE = self.fields.TREE
        if args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.fields.WORD
        else:
            self.WORD, self.FEAT = self.fields.WORD, self.fields.POS
        self.CHART = self.fields.CHART

    def train(self, train, dev, test, logger=None, **kwargs):
        args = self.args.update({'train': train,
                                 'dev': dev,
                                 'test': test,
                                 **kwargs})
        logger = logger or init_logger(path=args.path)

        train = TreebankCorpus.load(args.train, self.fields, args.max_len)
        dev = TreebankCorpus.load(args.dev, self.fields)
        test = TreebankCorpus.load(args.test, self.fields)
        train = TextDataset(train, self.fields, args.buckets)
        dev = TextDataset(dev, self.fields, args.buckets)
        test = TextDataset(test, self.fields, args.buckets)
        # set the data loaders
        train.loader = batchify(train, args.batch_size, True)
        dev.loader = batchify(dev, args.batch_size)
        test.loader = batchify(test, args.batch_size)
        logger.info(f"{'train:':6} {len(train):5} sentences, "
                    f"{len(train.loader):3} batches, "
                    f"{len(train.buckets)} buckets")
        logger.info(f"{'dev:':6} {len(dev):5} sentences, "
                    f"{len(dev.loader):3} batches, "
                    f"{len(train.buckets)} buckets")
        logger.info(f"{'test:':6} {len(test):5} sentences, "
                    f"{len(test.loader):3} batches, "
                    f"{len(train.buckets)} buckets\n")

        logger.info(f"{self.model}\n")
        self.optimizer = Adam(self.model.parameters(),
                              args.lr,
                              (args.mu, args.nu),
                              args.epsilon)
        self.scheduler = ExponentialLR(self.optimizer,
                                       args.decay**(1/args.decay_steps))

        total_time = timedelta()
        best_e, best_metric = 1, BracketMetric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train.loader)
            loss, dev_metric = self._evaluate(dev.loader)
            logger.info(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
            loss, test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':6} - loss: {loss:.4f} - {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                self.save(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= args.patience:
                break
        loss, metric = self.load(args.path)._evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:': 6} - {best_metric}")
        logger.info(f"{'test:':6} - {metric}")
        logger.info(f"{total_time}s elapsed, {total_time / epoch}s/epoch")

    def evaluate(self, data, logger=None, **kwargs):
        args = self.args.update(kwargs)
        logger = logger or init_logger()

        logger.info("Load the dataset")
        corpus = TreebankCorpus.load(data, self.fields)
        dataset = TextDataset(corpus, self.fields, args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        logger.info(f"{len(dataset)} sentences, "
                    f"{len(dataset.loader)} batches, "
                    f"{len(dataset.buckets)} buckets")

        logger.info("Evaluate the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        total_time = datetime.now() - start
        logger.info(f"loss: {loss:.4f} {metric}")
        logger.info(f"{total_time}s elapsed, "
                    f"{len(dataset)/total_time.total_seconds():.2f} Sents/s")

    def predict(self, data, pred=None, prob=True, logger=None, **kwargs):
        args = self.args.update({'prob': prob, **kwargs})
        logger = logger or init_logger()

        corpus = TreebankCorpus.load(data, self.fields)
        dataset = TextDataset(corpus,
                              [self.TREE, self.WORD, self.FEAT],
                              args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        logger.info(f"Load the dataset: "
                    f"{len(dataset)} sentences, "
                    f"{len(dataset.loader)} batches")

        logger.info("Make predictions on the dataset")
        start = datetime.now()
        pred_arcs, pred_rels, pred_probs = self._predict(dataset.loader)
        total_time = datetime.now() - start
        # restore the order of sentences in the buckets
        indices = torch.tensor([i
                                for bucket in dataset.buckets.values()
                                for i in bucket]).argsort()
        corpus.arcs = [pred_arcs[i] for i in indices]
        corpus.rels = [pred_rels[i] for i in indices]
        if args.prob:
            corpus.probs = [pred_probs[i] for i in indices]
        if pred is not None:
            logger.info(f"Save predicted results to {pred}")
            corpus.save(pred)
        logger.info(f"{total_time}s elapsed, "
                    f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")
        return corpus

    def _train(self, loader):
        self.model.train()

        progress = progress_bar(loader)

        for trees, words, feats, (spans, labels) in progress:
            self.optimizer.zero_grad()

            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            loss, _ = self.model.loss(s_span, s_label,
                                      spans, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            progress.set_postfix_str(f"lr: {self.scheduler.get_lr()[0]:.4e} - "
                                     f"loss: {loss:.4f}")

    @ torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, BracketMetric()

        for trees, words, feats, (spans, labels) in loader:
            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            loss, s_span = self.model.loss(s_span, s_label,
                                           spans, labels, mask)
            preds = self.model.decode(s_span, s_label, mask)
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            total_loss += loss.item()
            metric([factorize(tree, self.args.delete, self.args.equal)
                    for tree in preds],
                   [factorize(tree, self.args.delete, self.args.equal)
                    for tree in trees])
        total_loss /= len(loader)

        return total_loss, metric

    @ torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        trees, progress = [], progress_bar(loader)

        for trees, words, feats in progress:
            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            if self.args.mbr:
                s_span = self.model.crf(s_span, mask, mbr=True)
            preds = self.model.decode(s_span, s_label, mask)
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            trees.extend(preds)

        return trees

    @ classmethod
    def build(cls, path, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        args = Config().update({'path': path, **kwargs})
        if not os.path.exists(path) or args.build:
            logger.info("Build the fields")
            TREE = RawField('trees')
            WORD = Field('words', pad=pad, unk=unk,
                         bos=bos, eos=eos, lower=True)
            if args.feat == 'char':
                FEAT = SubwordField('chars',
                                    pad=pad, unk=unk, bos=bos, eos=eos,
                                    fix_len=args.fix_len, tokenize=list)
            elif args.feat == 'bert':
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.bert)
                if args.bert.startswith('bert'):
                    tokenizer.bos_token = tokenizer.cls_token
                    tokenizer.eos_token = tokenizer.sep_token
                FEAT = SubwordField('bert',
                                    pad=tokenizer.pad_token,
                                    unk=tokenizer.unk_token,
                                    bos=tokenizer.cls_token,
                                    eos=tokenizer.sep_token,
                                    fix_len=args.fix_len,
                                    tokenize=tokenizer.tokenize)
                FEAT.vocab = tokenizer.get_vocab()
            else:
                FEAT = Field('tags', bos=bos, eos=eos)
            CHART = ChartField('charts')
            if args.feat in ('char', 'bert'):
                fields = Treebank(TREE=TREE, WORD=(WORD, FEAT), CHART=CHART)
            else:
                fields = Treebank(TREE=TREE, WORD=WORD, POS=FEAT, CHART=CHART)

            train = TreebankCorpus.load(args.train, fields)
            if args.embed:
                embed = Embedding.load(args.embed, args.unk)
            else:
                embed = None
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
            model = CRFConstituencyModel(args)
            model.load_pretrained(WORD.embed).to(args.device)
            return cls(args, model, fields)
        else:
            parser = cls.load(**args)
            parser.model = CRFConstituencyModel(parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

    @classmethod
    def load(cls, path, **kwargs):
        if os.path.exists(path):
            state = torch.load(path, map_location='cpu')
        else:
            state = torch.hub.load_state_dict_from_url(path,
                                                       map_location='cpu')
        args = state['args']
        args.update({'path': path, **kwargs})
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CRFConstituencyModel(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        fields = state['fields']
        return cls(args, model, fields)

    def save(self, path):
        state_dict = self.model.state_dict()
        if hasattr(self.model, 'pretrained'):
            state_dict.pop('pretrained.weight')
        state = {'args': self.args,
                 'state_dict': state_dict,
                 'pretrained': self.WORD.embed,
                 'fields': self.fields}
        torch.save(state, path)


def run():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--path', '-p', default='exp/ptb.char/model',
                             help='path to model file')
    base_parser.add_argument('--conf', '-c', default='config.ini',
                             help='path to config file')
    base_parser.add_argument('--device', '-d', default='-1',
                             help='ID of GPU to use')
    base_parser.add_argument('--seed', '-s', default=1, type=int,
                             help='seed for generating random numbers')
    base_parser.add_argument('--threads', '-t', default=16, type=int,
                             help='max num of threads')
    base_parser.add_argument('--batch-size', default=5000, type=int,
                             help='batch size')
    base_parser.add_argument('--buckets', default=32, type=int,
                             help='max num of buckets to use')
    base_parser.add_argument('--mbr', action='store_true',
                             help='whether to use mbr decoding')

    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subparser = subparsers.add_parser(
        'train',
        help='Train a model.',
        parents=[base_parser]
    )
    subparser.add_argument('--feat', '-f', default='char',
                           choices=['tag', 'char', 'bert'],
                           help='choices of additional features')
    subparser.add_argument('--build', '-b', action='store_true',
                           help='whether to build the model first')
    subparser.add_argument('--max-len', default=None, type=int,
                           help='max length of the sentences')
    subparser.add_argument('--train', default='data/ptb/train.pid',
                           help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.pid',
                           help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.pid',
                           help='path to test file')
    subparser.add_argument('--embed', default='data/glove.6B.100d.txt',
                           help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk',
                           help='unk token in pretrained embeddings')
    subparser.add_argument('--bert', default='bert-base-cased',
                           help='which bert model to use')
    # evaluate
    subparser = subparsers.add_parser(
        'evaluate',
        help='Evaluate the specified model and dataset.',
        parents=[base_parser]
    )
    subparser.add_argument('--data', default='data/ptb/test.pid',
                           help='path to dataset')
    # predict
    subparser = subparsers.add_parser(
        'predict',
        help='Use a trained model to make predictions.',
        parents=[base_parser]
    )
    subparser.add_argument('--prob', action='store_true',
                           help='whether to output probs')
    subparser.add_argument('--data', default='data/ptb/test.pid',
                           help='path to dataset')
    subparser.add_argument('--pred', default='pred.pid',
                           help='path to predicted result')
    args = parser.parse_args()

    logger = init_logger(path=args.path)
    logger.info(f"Set the max num of threads to {args.threads}")
    logger.info(f"Set the seed for generating random numbers to {args.seed}")
    logger.info(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = Config(args.conf).update(vars(args))
    logger.info('\n' + str(args))

    if args.mode == 'train':
        parser = CRFConstituencyParser.build(**args)
        parser.train(**args, logger=logger)
    elif args.mode == 'evaluate':
        parser = CRFConstituencyParser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = CRFConstituencyParser.load(args.path)
        parser.predict(**args)
