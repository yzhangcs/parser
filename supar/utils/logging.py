# -*- coding: utf-8 -*-

import logging
import os

from supar.utils.parallel import is_master
from tqdm import tqdm


def get_logger(name):
    return logging.getLogger(name)


def init_logger(logger,
                path=None,
                mode='w',
                level=None,
                handlers=None,
                verbose=True):
    level = level or logging.WARNING
    if not handlers:
        handlers = [logging.StreamHandler()]
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handlers.append(logging.FileHandler(path, mode))
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=level,
                        handlers=handlers)
    logger.setLevel(logging.INFO if is_master() and verbose else logging.WARNING)


def progress_bar(iterator,
                 ncols=None,
                 bar_format='{l_bar}{bar:18}| {n_fmt}/{total_fmt} {elapsed}<{remaining}, {rate_fmt}{postfix}',
                 leave=True):
    return tqdm(iterator,
                ncols=ncols,
                bar_format=bar_format,
                ascii=True,
                disable=(not (logger.level == logging.INFO and is_master())),
                leave=leave)


logger = get_logger('supar')
