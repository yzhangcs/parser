# -*- coding: utf-8 -*-

import logging
import os
from datetime import datetime

import torch.distributed as dist
from tqdm import tqdm

logger = logging.getLogger('supar')


def init_logger(path=None,
                mode='a',
                level=None,
                handlers=None):
    if level is None:
        level = logging.INFO
        if dist.is_initialized() and dist.get_rank() != 0:
            level = logging.WARNING
    if not handlers:
        handlers = [logging.StreamHandler()]
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            name = datetime.now().strftime("%y-%m-%d_%H.%M.%S")
            handlers.append(logging.FileHandler(f"{path}.{name}.log", mode))
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=level,
                        handlers=handlers)


def progress_bar(iterator,
                 ncols=None,
                 bar_format='{l_bar}{bar:36}| {n_fmt}/{total_fmt} '
                 '{elapsed}<{remaining}, {rate_fmt}{postfix}'):
    return tqdm(iterator,
                ncols=ncols,
                bar_format=bar_format,
                ascii=True,
                disable=(dist.is_initialized() and dist.get_rank() != 0))
