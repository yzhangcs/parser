# -*- coding: utf-8 -*-

import logging
import os
from logging import FileHandler, Formatter, Handler, Logger, StreamHandler
from typing import Iterable, Optional

from supar.utils.parallel import is_master
from tqdm import tqdm


def get_logger(name: Optional[str] = None) -> Logger:
    logger = logging.getLogger(name)
    # init the root logger
    if name is None:
        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[TqdmHandler()])
    return logger


class TqdmHandler(StreamHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def init_logger(
    logger: Logger,
    path: Optional[str] = None,
    mode: str = 'w',
    handlers: Optional[Iterable[Handler]] = None,
    verbose: bool = True
) -> Logger:
    if not handlers:
        if path:
            os.makedirs(os.path.dirname(path) or './', exist_ok=True)
            logger.addHandler(FileHandler(path, mode))
    for handler in logger.handlers:
        handler.setFormatter(ColoredFormatter(colored=not isinstance(handler, FileHandler)))
    logger.setLevel(logging.INFO if is_master() and verbose else logging.WARNING)
    return logger


def progress_bar(
    iterator: Iterable,
    ncols: Optional[int] = None,
    bar_format: str = '{l_bar}{bar:20}| {n_fmt}/{total_fmt} {elapsed}<{remaining}, {rate_fmt}{postfix}',
    leave: bool = False,
    **kwargs
) -> tqdm:
    return tqdm(iterator,
                ncols=ncols,
                bar_format=bar_format,
                ascii=True,
                disable=(not (logger.level == logging.INFO and is_master())),
                leave=leave,
                **kwargs)


class ColoredFormatter(Formatter):

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    GREY = '\033[37m'
    RESET = '\033[0m'

    COLORS = {
        logging.ERROR: RED,
        logging.WARNING: RED,
        logging.INFO: GREEN,
        logging.DEBUG: BLACK,
        logging.NOTSET: BLACK
    }

    def __init__(self, colored=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.colored = colored

    def format(self, record):
        fmt = '[%(asctime)s %(levelname)s] %(message)s'
        if self.colored:
            fmt = f'{self.COLORS[record.levelno]}[%(asctime)s %(levelname)s]{self.RESET} %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        return Formatter(fmt=fmt, datefmt=datefmt).format(record)


logger = get_logger()
