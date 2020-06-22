# -*- coding: utf-8 -*-

import argparse
import os

import torch
import torch.nn as nn
from supar import Config
from supar.models import MODELS
from supar.parsers.biaffine_parser import BiaffineParser
from supar.utils import Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import Field, SubwordField
from supar.utils.logging import init_logger, logger, progress_bar
from supar.utils.metric import AttachmentMetric
from supar.utils.transform import CoNLL


class CRF2oDependencyParser(BiaffineParser):

    def __init__(self, *args, **kwargs):
        super(CRF2oDependencyParser, self).__init__(*args, **kwargs)

    def _train(self, loader):
        self.model.train()

        metric = AttachmentMetric()
        progress = progress_bar(loader)

        for words, feats, arcs, sibs, rels in progress:
            self.optimizer.zero_grad()

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_sib, s_rel = self.model(words, feats)
            loss, s_arc = self.model.loss(s_arc, s_sib, s_rel,
                                          arcs, sibs, rels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
            progress.set_postfix_str(f"lr: {self.scheduler.get_lr()[0]:.4e} - "
                                     f"loss: {loss:.4f} - "
                                     f"{metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, AttachmentMetric()

        for words, feats, arcs, sibs, rels in loader:
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_sib, s_rel = self.model(words, feats)
            loss, s_arc = self.model.loss(s_arc, s_sib, s_rel,
                                          arcs, sibs, rels, mask)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
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
            s_arc, s_sib, s_rel = self.model(words, feats)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                s_arc = s_arc.softmax(-1)
                arc_probs = s_arc.gather(-1, arc_preds.unsqueeze(-1))
                probs.extend(arc_probs.squeeze(-1)[mask].split(lens))
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = [seq.tolist() for seq in probs]

        return preds

    @classmethod
    def build(cls, path, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        args = Config().update({'path': path, **kwargs})
        if not os.path.exists(path) or args.build:
            logger.info("Build the fields")
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            if args.feat == 'char':
                FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos,
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
                                    bos=tokenizer.bos_token,
                                    fix_len=args.fix_len,
                                    tokenize=tokenizer.tokenize)
                FEAT.vocab = tokenizer.get_vocab()
            else:
                FEAT = Field('tags', bos=bos)
            ARC = Field('arcs', bos=bos, use_vocab=False,
                        fn=CoNLL.numericalize)
            SIB = Field('sibs', bos=bos, use_vocab=False,
                        fn=CoNLL.numericalize_sibs)
            REL = Field('rels', bos=bos)
            if args.feat in ('char', 'bert'):
                fields = CoNLL(FORM=(WORD, FEAT),
                               HEAD=(ARC, SIB), DEPREL=REL)
            else:
                fields = CoNLL(FORM=WORD, CPOS=FEAT,
                               HEAD=(ARC, SIB), DEPREL=REL)

            train = CoNLL.load(args.train)
            embed = None
            if args.embed:
                embed = Embedding.load(args.embed, args.unk)
            WORD.build(train, args.min_freq, embed)
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
            model = MODELS[args.model](args)
            model = model.load_pretrained(WORD.embed).to(args.device)
            return cls(args, model, fields)
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
    base_parser.add_argument('--batch-size', default=5000, type=int,
                             help='batch size')
    base_parser.add_argument('--buckets', default=32, type=int,
                             help='max num of buckets to use')
    base_parser.add_argument('--partial', action='store_true',
                             help='whether partial annotation is included')
    base_parser.add_argument('--mbr', action='store_true',
                             help='whether to use mbr decoding')
    base_parser.add_argument('--tree', action='store_true',
                             help='whether to ensure well-formedness')
    base_parser.add_argument('--proj', action='store_true',
                             help='whether to projectivise the data')

    parser = argparse.ArgumentParser(
        description='Create the CRF Dependency Parser model.'
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
    subparser.add_argument('--punct', action='store_true',
                           help='whether to include punctuation')
    subparser.add_argument('--max-len', default=None, type=int,
                           help='max length of the sentences')
    subparser.add_argument('--train', default='data/ptb/train.conllx',
                           help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx',
                           help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx',
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
    subparser.add_argument('--punct', action='store_true',
                           help='whether to include punctuation')
    subparser.add_argument('--data', default='data/ptb/test.conllx',
                           help='path to dataset')
    # predict
    subparser = subparsers.add_parser(
        'predict',
        help='Use a trained model to make predictions.',
        parents=[base_parser]
    )
    subparser.add_argument('--prob', action='store_true',
                           help='whether to output probs')
    subparser.add_argument('--data', default='data/ptb/test.conllx',
                           help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx',
                           help='path to predicted result')
    args.update(vars(parser.parse_known_args()[0]))

    logger = init_logger(path=args.path)
    logger.info(f"Set the max num of threads to {args.threads}")
    logger.info(f"Set the seed for generating random numbers to {args.seed}")
    logger.info(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('\n' + str(args))

    if args.mode == 'train':
        parser = CRF2oDependencyParser.build(**args)
        parser.train(**args, logger=logger)
    elif args.mode == 'evaluate':
        parser = CRF2oDependencyParser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = CRF2oDependencyParser.load(args.path)
        parser.predict(**args)
