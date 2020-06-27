# -*- coding: utf-8 -*-

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from supar import Config
from supar.models import MODELS
from supar.parsers.parser import Parser
from supar.utils import Dataset, Embedding
from supar.utils.common import bos, eos, pad, unk
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.fn import build, factorize
from supar.utils.logging import init_logger, progress_bar
from supar.utils.metric import BracketMetric
from supar.utils.transform import Tree


class CRFConstituencyParser(Parser):

    def __init__(self, args, model, fields):
        super(CRFConstituencyParser, self).__init__(args, model, fields)

        if args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.WORD
        else:
            self.WORD, self.FEAT = self.transform.WORD, self.transform.POS
        self.TREE = self.transform.TREE
        self.CHART = self.transform.CHART

    def _train(self, loader):
        self.model.train()

        progress = progress_bar(loader)

        for words, feats, trees, (spans, labels) in progress:
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

        for words, feats, trees, (spans, labels) in loader:
            batch_size, seq_len = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(words, feats)
            loss, s_span = self.model.loss(s_span, s_label,
                                           spans, labels, mask)
            chart_preds = self.model.decode(s_span, s_label, mask)
            # since the evaluation relies on terminals,
            # the tree should be first built and then factorized
            preds = [build(tree,
                           [(i, j, self.CHART.vocab[label])
                            for i, j, label in chart])
                     for tree, chart in zip(trees, chart_preds)]
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
            preds['trees'].extend([build(tree,
                                         [(i, j, self.CHART.vocab[label])
                                          for i, j, label in chart])
                                   for tree, chart in zip(trees, chart_preds)])
            if self.args.prob:
                probs.extend(s_span.tolist())
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @ classmethod
    def build(cls, path, logger=None, **kwargs):
        args = Config().update(locals())
        logger = logger or init_logger()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path) or args.build:
            logger.info("Build the fields")
            WORD = Field('words', pad=pad, unk=unk,
                         bos=bos, eos=eos, lower=True)
            if args.feat == 'char':
                FEAT = SubwordField('chars',
                                    pad=pad,
                                    unk=unk,
                                    bos=bos,
                                    eos=eos,
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
            model = MODELS[args.model](args)
            model.load_pretrained(WORD.embed).to(args.device)
            return cls(args, model, transform)
        else:
            parser = cls.load(**args)
            parser.model = MODELS[args.model](parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser


def run(args):
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
    base_parser.add_argument('--num-workers', '-w', default=4, type=int,
                             help='num of processes to build the dataset')
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
    args.update(vars(parser.parse_known_args()[0]))

    dist.init_process_group(backend='nccl')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.cuda.set_device(args.local_rank)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = init_logger(path=args.path)
    logger.info(f"Set the max num of threads to {args.threads}")
    logger.info(f"Set the seed for generating random numbers to {args.seed}")
    logger.info(f"Set the device with ID {args.device} visible")
    logger.info('\n' + str(args))

    if args.mode == 'train':
        parser = CRFConstituencyParser.build(**args, logger=logger)
        parser.train(**args, logger=logger)
    elif args.mode == 'evaluate':
        parser = CRFConstituencyParser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = CRFConstituencyParser.load(args.path)
        parser.predict(**args)
