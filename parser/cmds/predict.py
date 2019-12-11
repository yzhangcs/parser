# -*- coding: utf-8 -*-

from datetime import datetime
from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.utils.field import Field

import torch


class Predict(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--prob', action='store_true',
                               help='whether to output probs')
        subparser.add_argument('--fdata', default='data/ptb/test.conllx',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='pred.conllx',
                               help='path to predicted result')

        return subparser

    def __call__(self, args):
        super(Predict, self).__call__(args)

        print("Load the dataset")
        if args.prob:
            self.fields = self.fields._replace(PHEAD=Field('probs'))
        corpus = Corpus.load(args.fdata, self.fields)
        dataset = TextDataset(corpus, [self.WORD, self.FEAT], args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches")

        print("Load the model")
        self.model = Model.load(args.model)
        print(f"{self.model}\n")

        print("Make predictions on the dataset")
        start = datetime.now()
        pred_arcs, pred_rels, pred_probs = self.predict(dataset.loader)
        total_time = datetime.now() - start
        # restore the order of sentences in the buckets
        indices = torch.tensor([i for bucket in dataset.buckets.values()
                                for i in bucket]).argsort()
        corpus.arcs = [pred_arcs[i] for i in indices]
        corpus.rels = [pred_rels[i] for i in indices]
        if args.prob:
            corpus.probs = [pred_probs[i] for i in indices]
        print(f"Save the predicted result to {args.fpred}")
        corpus.save(args.fpred)
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")
