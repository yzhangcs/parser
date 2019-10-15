# -*- coding: utf-8 -*-

from datetime import datetime
from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify


class Predict(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--fdata', default='data/ptb/test.conllx',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='pred.conllx',
                               help='path to predicted result')

        return subparser

    def __call__(self, args):
        super(Predict, self).__call__(args)

        print("Load the dataset")
        corpus = Corpus.load(args.fdata, self.fields)
        dataset = TextDataset(corpus, [self.WORD, self.FEAT])
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches")

        print("Load the model")
        self.model = Model.load(args.model)
        print(f"{self.model}\n")

        print("Make predictions on the dataset")
        start = datetime.now()
        corpus.heads, corpus.rels = self.predict(dataset.loader)
        print(f"Save the predicted result to {args.fpred}")
        corpus.save(args.fpred)
        total_time = datetime.now() - start
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")
