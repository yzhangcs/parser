# -*- coding: utf-8 -*-

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn

if TYPE_CHECKING:
    from supar.parsers import Parser


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)

    def __getattr__(self, name):
        wrapped = super().__getattr__('module')
        if hasattr(wrapped, name):
            return getattr(wrapped, name)
        return super().__getattr__(name)


class parallel(object):

    def __init__(self, training=True, op='sum'):
        self.training = training
        self.op = op

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        ...

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(parser: Parser, *args, **kwargs):
            parser.model.train(self.training)
            if not dist.is_initialized():
                return fn(parser, *args, **kwargs)
            if self.training:
                with parser.model.join():
                    results = fn(parser, *args, **kwargs)
            else:
                with torch.no_grad():
                    dist_model = parser.model
                    # https://github.com/pytorch/pytorch/issues/54059
                    if hasattr(parser.model, 'module'):
                        parser.model = parser.model.module
                    results = fn(parser, *args, **kwargs)
                    parser.model = dist_model
                    dist.barrier()
            if results is None:
                return results
            if self.op is None:
                return results
            elif self.op == 'sum':
                return functools.reduce(lambda x, y: x + y, gather(results))
            else:
                raise NotImplementedError(f"Op {self.op} not supported yet")
        return wrapper


def is_master():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))
    port = str(s.getsockname()[1])
    s.close()
    return port


def gather(obj: Any) -> Iterable[Any]:
    objs = [None] * dist.get_world_size()
    dist.all_gather_object(objs, obj)
    return objs
