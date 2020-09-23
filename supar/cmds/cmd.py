# -*- coding: utf-8 -*-

import torch
from supar.utils import Config
from supar.utils.logging import init_logger, logger
from supar.utils.parallel import init_device


def parse(parser):
    parser.add_argument('--conf', '-c', help='path to config file')
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int, help='max num of threads')
    parser.add_argument('--batch-size', default=5000, type=int, help='batch size')
    parser.add_argument("--local_rank", type=int, default=-1, help='node rank for distributed training')
    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))
    Parser = args.pop('Parser')

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    init_device(args.device, args.local_rank)
    init_logger(logger, f"{args.path}.{args.mode}.log")
    logger.info('\n' + str(args))

    if args.mode == 'train':
        parser = Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(args.path)
        parser.predict(**args)
