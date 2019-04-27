# -*- coding: utf-8 -*-

import os
from parser import BiaffineParser, Model
from parser.data import Corpus, Embedding, TextDataset, Vocab, collate_fn

import torch
from torch.utils.data import DataLoader


class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--epsilon', default=1e-12, type=float,
                               help='epsilon')
        subparser.add_argument('--betas', default=(0.9, 0.9),
                               help='betas')
        subparser.add_argument('--annealing', default=lambda x: .75**(x/5000),
                               help='annealing')
        subparser.add_argument('--batch-size', default=200, type=int,
                               help='batch size')
        subparser.add_argument('--epochs', default=1000, type=int,
                               help='max num of epochs')
        subparser.add_argument('--patience', default=100, type=int,
                               help='patience for early stop')
        subparser.add_argument('--lr', default=2e-3, type=float,
                               help='learning rate of training')
        subparser.add_argument('--n-embed', default=100, type=int,
                               help='dim of word embedding')
        subparser.add_argument('--n-char-embed', default=50, type=int,
                               help='dim of char embedding')
        subparser.add_argument('--n-char-out', default=100, type=int,
                               help='dim of char level lstm output')
        subparser.add_argument('--n-lstm-hidden', default=400, type=int,
                               help='dim of lstm hidden state')
        subparser.add_argument('--n-lstm-layers', default=3, type=int,
                               help='num of lstm layers')
        subparser.add_argument('--n-mlp-arc', default=500, type=int,
                               help='arc mlp size')
        subparser.add_argument('--n-mlp-lab', default=100, type=int,
                               help='label mlp size')
        subparser.add_argument('--dropout', default=0.33, type=float,
                               help='dropout ratio')
        subparser.add_argument('--ftrain', default='data/train.conllx',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/dev.conllx',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/test.conllx',
                               help='path to test file')
        subparser.add_argument('--fembed', default='data/glove.6B.100d.txt',
                               help='path to pretrained embedding file')
        subparser.add_argument('--file', '-f', default='model.pt',
                               help='path to model file')
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=4, type=int,
                               help='max num of threads')
        subparser.add_argument('--device', '-d', default='-1',
                               help='ID of GPU to use')
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):
        print(f"Set the max num of threads to {args.threads}")
        print(f"Set the seed for generating random numbers to {args.seed}")
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)

        print(f"Set the device with ID {args.device} visible")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

        print("Preprocess the data")
        train = Corpus.load(args.ftrain)
        dev = Corpus.load(args.fdev)
        test = Corpus.load(args.ftest)
        embed = Embedding.load(args.fembed)
        vocab = Vocab.from_corpus(corpus=train, min_freq=2)
        vocab.read_embeddings(embed=embed, unk='unk')
        print(vocab)

        print("Load the dataset")
        trainset = TextDataset(vocab.numericalize(train))
        devset = TextDataset(vocab.numericalize(dev))
        testset = TextDataset(vocab.numericalize(test))
        # set the data loaders
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
        dev_loader = DataLoader(dataset=devset,
                                batch_size=args.batch_size,
                                collate_fn=collate_fn)
        test_loader = DataLoader(dataset=testset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate_fn)
        print(f"  size of trainset: {len(trainset)}")
        print(f"  size of devset: {len(devset)}")
        print(f"  size of testset: {len(testset)}")

        print("Create the model")
        params = {
            'n_embed': args.n_embed,
            'n_char_embed': args.n_char_embed,
            'n_char_out': args.n_char_out,
            'n_lstm_hidden': args.n_lstm_hidden,
            'n_lstm_layers': args.n_lstm_layers,
            'n_mlp_arc': args.n_mlp_arc,
            'n_mlp_lab': args.n_mlp_lab,
            'dropout': args.dropout
        }
        for k, v in params.items():
            print(f"  {k}: {v}")
        parser = BiaffineParser(vocab, params)
        if torch.cuda.is_available():
            parser = parser.cuda()
        print(f"{parser}\n")

        model = Model(parser=parser)
        model(loaders=(train_loader, dev_loader, test_loader),
              epochs=args.epochs,
              patience=args.patience,
              betas=args.betas,
              lr=args.lr,
              epsilon=args.epsilon,
              annealing=args.annealing,
              file=args.file)
