# -*- coding: utf-8 -*-

import os
from parser import BiaffineParser, Model
from parser.utils import Corpus, TextDataset, collate_fn

import torch
from torch.utils.data import DataLoader


class Evaluate(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--batch-size', default=200, type=int,
                               help='batch size')
        subparser.add_argument('--fdata', default='data/test.conllx',
                               help='path to dataset')
        subparser.add_argument('--file', '-f', default='model.pt',
                               help='path to model file')
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=4, type=int,
                               help='max num of threads')
        subparser.add_argument('--device', '-d', default='-1',
                               help='id of GPU to use')
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):
        print(f"Set the max num of threads to {args.threads}")
        print(f"Set the seed for generating random numbers to {args.seed}")
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)

        print(f"Set the device with ID {args.device} visible")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

        print("Load the model")
        parser = BiaffineParser.load(args.file)
        vocab = parser.vocab

        print("Load the dataset")
        corpus = Corpus.load(args.fdata)
        dataset = TextDataset(vocab.numericalize(corpus))
        # set the data loader
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            collate_fn=collate_fn)

        print("Evaluate the dataset")
        model = Model(parser=parser)
        loss, metric = model.evaluate(loader)
        print(f"Loss: {loss:.4f} {metric}")
