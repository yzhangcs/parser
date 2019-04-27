# -*- coding: utf-8 -*-

import argparse
import os
from parser import BiaffineParser, Trainer
from parser.data import Corpus, Embedding, TextDataset, Vocab, collate_fn
from parser.utils import numericalize

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    parser.add_argument('--dropout', action='store', default=0.33, type=float,
                        help='set the prob of dropout')
    parser.add_argument('--batch-size', action='store', default=200, type=int,
                        help='set the size of batch')
    parser.add_argument('--epochs', action='store', default=1000, type=int,
                        help='set the max num of epochs')
    parser.add_argument('--patience', action='store', default=100, type=int,
                        help='set the num of epochs to be patient')
    parser.add_argument('--lr', action='store', default=2e-3, type=float,
                        help='set the learning rate of training')
    parser.add_argument('--device', '-d', action='store', default='-1',
                        help='set which device to use')
    parser.add_argument('--file', '-f', action='store', default='model.pt',
                        help='set where to store the model')
    parser.add_argument('--seed', '-s', action='store', default=1, type=int,
                        help='set the seed for generating random numbers')
    parser.add_argument('--threads', '-t', action='store', default=4, type=int,
                        help='set the max num of threads')
    parser.add_argument('--ftrain', action='store',
                        default='data/train.conllx',
                        help='set the path to train file')
    parser.add_argument('--fdev', action='store',
                        default='data/dev.conllx',
                        help='set the path to dev file')
    parser.add_argument('--ftest', action='store',
                        default='data/test.conllx',
                        help='set the path to test file')
    parser.add_argument('--fembed', action='store',
                        default='data/glove.6B.100d.txt',
                        help='set the path to the pretrained embedding file')
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    print("Preprocess the data")
    train = Corpus(fname=args.ftrain)
    dev = Corpus(fname=args.fdev)
    test = Corpus(fname=args.ftest)
    embed = Embedding(fname=args.fembed)
    vocab = Vocab.from_corpus(corpus=train, min_freq=2)
    vocab.read_embeddings(embed=embed, unk='unk')
    print(vocab)

    print("Load the dataset")
    trainset = TextDataset(numericalize(vocab=vocab, corpus=train))
    devset = TextDataset(numericalize(vocab=vocab, corpus=dev))
    testset = TextDataset(numericalize(vocab=vocab, corpus=test))
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

    print("Create Neural Network")
    params = {
        'n_embed': Config.n_embed,
        'n_char_embed': Config.n_char_embed,
        'n_char_out': Config.n_char_out,
        'n_lstm_hidden': Config.n_lstm_hidden,
        'n_lstm_layers': Config.n_lstm_layers,
        'n_mlp_arc': Config.n_mlp_arc,
        'n_mlp_lab': Config.n_mlp_lab,
        'dropout': args.dropout
    }
    for k, v in params.items():
        print(f"  {k}: {v}")
    model = BiaffineParser(vocab, **params)
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"{model}\n")

    optimizer = optim.Adam(params=model.parameters(),
                           betas=Config.betas, lr=args.lr,
                           eps=Config.epsilon)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda x: .75**(x/5000))

    trainer = Trainer(model=model,
                      vocab=vocab,
                      optimizer=optimizer,
                      scheduler=scheduler)
    trainer.fit(train_loader=train_loader,
                dev_loader=dev_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                patience=args.patience,
                file=args.file)
