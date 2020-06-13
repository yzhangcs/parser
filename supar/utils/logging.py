# -*- coding: utf-8 -*-

import logging
import os
import sys
from datetime import datetime

from tqdm import tqdm


def init_logger(name=datetime.now().strftime("%y-%m-%d_%H.%M.%S"),
                path=None,
                tqdm=True,
                level=None,
                mode='a'):
    level = level or logging.INFO
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.propagate = False

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    attached_to_std = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stderr or handler.stream == sys.stdout:
                attached_to_std = True
                break
    if not attached_to_std:
        logger.addHandler(consoleHandler)
    logger.setLevel(level)
    consoleHandler.setLevel(level)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fileHandler = logging.FileHandler(f"{path}.{name}.log", mode=mode)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        fileHandler.setLevel(level)

    return logger


def progress_bar(iterator,
                 ncols=None,
                 bar_format='{l_bar}{bar:36}| {n_fmt}/{total_fmt} '
                 '{elapsed}<{remaining}, {rate_fmt}{postfix}'):
    return tqdm(iterator, ncols=ncols, bar_format=bar_format, ascii=True)


logger = init_logger(name='supar')
