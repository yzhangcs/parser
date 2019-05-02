# -*- coding: utf-8 -*-

import argparse
import os
from parser.cmds import Evaluate, Predict, Train

import torch

from config import Config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--conf', '-c', default='config.ini',
                               help='path to config file')
        subparser.add_argument('--model', '-m', default='exp/ptb/model.tag',
                               help='path to model file')
        subparser.add_argument('--vocab', '-v', default='exp/ptb/vocab.tag',
                               help='path to vocab file')
        subparser.add_argument('--device', '-d', default='-1',
                               help='ID of GPU to use')
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=4, type=int,
                               help='max num of threads')
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    print(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    print(f"Override the default configs with parsed arguments")
    config = Config(args.conf)
    config.update(vars(args))
    print(config)

    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(config)
