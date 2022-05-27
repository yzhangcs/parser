# -*- coding: utf-8 -*-

import logging
import os
from logging import Logger, Handler
from typing import Iterable, Optional

from supar.utils.parallel import is_master
from tqdm import tqdm


def get_logger(name: str) -> Logger:
    return logging.getLogger(name)


class TqdmHandler(logging.StreamHandler):

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
    level: Optional[int] = None,
    handlers: Optional[Iterable[Handler]] = None,
    verbose: bool = True
) -> None:
    level = level or logging.WARNING
    if not handlers:
        handlers = [TqdmHandler()]
        if path:
            os.makedirs(os.path.dirname(path) or './', exist_ok=True)
            handlers.append(logging.FileHandler(path, mode))
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=level,
                        handlers=handlers)
    logger.setLevel(logging.INFO if is_master() and verbose else logging.WARNING)


def progress_bar(
    iterator: Iterable,
    ncols: Optional[int] = None,
    bar_format: str = '{l_bar}{bar:18}| {n_fmt}/{total_fmt} {elapsed}<{remaining}, {rate_fmt}{postfix}',
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


logger = get_logger('supar')
