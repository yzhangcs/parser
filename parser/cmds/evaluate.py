# -*- coding: utf-8 -*-

from datetime import datetime
from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify


class Evaluate(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--buckets', default=32, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--fdata', default='data/ptb/test.conllx',
                               help='path to dataset')

        return subparser

    def __call__(self, args):
        super(Evaluate, self).__call__(args)

        print("Load the dataset")
        corpus = Corpus.load(args.fdata, self.fields)
        dataset = TextDataset(corpus, self.fields, args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches, "
              f"{len(dataset.buckets)} buckets")

        print("Load the model")
        self.model = Model.load(args.model)
        print(f"{self.model}\n")

        print("Evaluate the dataset")
        start = datetime.now()
        loss, metric = self.evaluate(dataset.loader)
        total_time = datetime.now() - start
        print(f"Loss: {loss:.4f} {metric}")
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")
