# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from supar.config import Config
from supar.model import Model
from supar.utils import Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.corpus import CoNLL, Corpus
from supar.utils.data import TextDataset, batchify
from supar.utils.field import Field, SubwordField
from supar.utils.fn import ispunct, numericalize
from supar.utils.metric import AttachmentMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class BiaffineParser(object):

    def __init__(self, args, model, fields):
        super(BiaffineParser, self).__init__()

        self.args = args
        self.model = model
        self.fields = fields
        if args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.fields.FORM
        else:
            self.WORD, self.FEAT = self.fields.FORM, self.fields.CPOS
        self.ARC, self.REL = self.fields.HEAD, self.fields.DEPREL
        self.puncts = torch.tensor([i
                                    for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(args.device)

        print(f"{self.WORD}\n{self.FEAT}\n{self.ARC}\n{self.REL}")
        print(f"{model}\n")

    def train(self, train, dev, test, **kwargs):
        args = self.args.update(kwargs)

        train = Corpus.load(args.train, self.fields)
        dev = Corpus.load(args.dev, self.fields)
        test = Corpus.load(args.test, self.fields)
        train = TextDataset(train, self.fields, args.buckets)
        dev = TextDataset(dev, self.fields, args.buckets)
        test = TextDataset(test, self.fields, args.buckets)
        # set the data loaders
        train.loader = batchify(train, args.batch_size, True)
        dev.loader = batchify(dev, args.batch_size)
        test.loader = batchify(test, args.batch_size)
        print(f"{'train:':6} {len(train):5} sentences, "
              f"{len(train.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'dev:':6} {len(dev):5} sentences, "
              f"{len(dev.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'test:':6} {len(test):5} sentences, "
              f"{len(test.loader):3} batches, "
              f"{len(train.buckets)} buckets\n")

        print("Create the model")
        self.model = Model(args).load_pretrained(self.WORD.embed)
        print(f"{self.model}\n")
        self.model = self.model.to(args.device)
        self.optimizer = Adam(self.model.parameters(),
                              args.lr,
                              (args.mu, args.nu),
                              args.epsilon)
        self.scheduler = ExponentialLR(self.optimizer,
                                       args.decay**(1/args.decay_steps))

        total_time = timedelta()
        best_e, best_metric = 1, AttachmentMetric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            print(f"Epoch {epoch} / {args.epochs}:")
            loss, train_metric = self._train(train.loader)
            print(f"{'train:':6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = self._evaluate(dev.loader)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = self._evaluate(test.loader)
            print(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric and epoch > args.patience//10:
                best_e, best_metric = epoch, dev_metric
                self.save(args.path)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= args.patience:
                break
        self = self.load(args.path)
        loss, metric = self.evaluate(test.loader)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

    def evaluate(self, data, **kwargs):
        args = self.args.update(kwargs)

        print("Load the dataset")
        corpus = Corpus.load(data, self.fields)
        dataset = TextDataset(corpus, self.fields, args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches, "
              f"{len(dataset.buckets)} buckets")

        print("Evaluate the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        total_time = datetime.now() - start
        print(f"Loss: {loss:.4f} {metric}")
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")

    def predict(self, data, pred=None, **kwargs):
        args = self.args.update(kwargs)
        print("Load the dataset")
        if args.prob:
            self.fields = self.fields._replace(PHEAD=Field('probs'))
        corpus = Corpus.load(data, self.fields)
        dataset = TextDataset(corpus, [self.WORD, self.FEAT], args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches")

        self.model.args = args

        print("Make predictions on the dataset")
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
        print(f"Save the predicted result to {pred}")
        corpus.save(pred)
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")
        return corpus

    def _train(self, loader):
        self.model.train()

        total_loss, metric = 0, AttachmentMetric()

        for words, feats, arcs, rels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

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

        arcs, rels, probs = [], [], []
        for words, feats in loader:
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(words, feats)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                s_arc = s_arc.softmax(-1)
                arc_probs = s_arc.gather(-1, arc_preds.unsqueeze(-1))
                probs.extend(arc_probs.squeeze(-1)[mask].split(lens))
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        probs = [[round(p, 4) for p in seq.tolist()] for seq in probs]

        return arcs, rels, probs

    @classmethod
    def build(cls, path, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        args = Config().update({'path': path, **kwargs})
        if not os.path.exists(path) or args.build:
            print("Build the fields")
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            if args.feat == 'char':
                FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos,
                                    fix_len=args.fix_len, tokenize=list)
            elif args.feat == 'bert':
                from transformers import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                FEAT = SubwordField('bert',
                                    pad=tokenizer.pad_token,
                                    unk=tokenizer.unk_token,
                                    bos=tokenizer.cls_token,
                                    tokenize=tokenizer.tokenize)
                FEAT.vocab = tokenizer.vocab
            else:
                FEAT = Field('tags', bos=bos)
            ARC = Field('arcs', bos=bos, use_vocab=False, fn=numericalize)
            REL = Field('rels', bos=bos)
            if args.feat in ('char', 'bert'):
                fields = CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL)
            else:
                fields = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)

            train = Corpus.load(args.train, fields)
            if args.embed:
                embed = Embedding.load(args.embed, args.unk)
            else:
                embed = None
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
            model = Model(args).load_pretrained(WORD.embed).to(args.device)
            return cls(args, model, fields)

        return cls.load(**args)

    @classmethod
    def load(cls, path, **kwargs):
        state = torch.load(path, map_location='cpu')
        args = state['args']
        args.update({'path': path, **kwargs})
        model = Model(state['args'])
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
    base_parser.add_argument('--path', '-p', default='exp/ptb.char',
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
    base_parser.add_argument('--tree', action='store_true',
                             help='whether to ensure well-formedness')
    base_parser.add_argument('--proj', action='store_true',
                             help='whether to projectivise the data')

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
    subparser.add_argument('--punct', action='store_true',
                           help='whether to include punctuation')
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
    subparser.add_argument('--bert-model', default='bert-base-cased',
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
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    print(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = Config(args.conf).update(vars(args))
    print(args)

    if args.mode == 'train':
        parser = BiaffineParser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = BiaffineParser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = BiaffineParser.load(args.path)
        parser.evaluate(**args)
