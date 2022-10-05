# -*- coding: utf-8 -*-

from __future__ import annotations

import functools
import os
import re
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)

    def __getattr__(self, name):
        wrapped = super().__getattr__('module')
        if hasattr(wrapped, name):
            return getattr(wrapped, name)
        return super().__getattr__(name)


def wait(fn) -> Any:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        value = None
        if is_master():
            value = fn(*args, **kwargs)
        if is_dist():
            dist.barrier()
            value = gather(value)[0]
        return value
    return wrapper


def gather(obj: Any) -> Iterable[Any]:
    objs = [None] * dist.get_world_size()
    dist.all_gather_object(objs, obj)
    return objs


def reduce(obj: Any, reduction: str = 'sum') -> Any:
    objs = gather(obj)
    if reduction == 'sum':
        return functools.reduce(lambda x, y: x + y, objs)
    elif reduction == 'mean':
        return functools.reduce(lambda x, y: x + y, objs) / len(objs)
    elif reduction == 'min':
        return min(objs)
    elif reduction == 'max':
        return max(objs)
    else:
        raise NotImplementedError(f"Unsupported reduction {reduction}")


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return not is_dist() or dist.get_rank() == 0


def get_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))
    port = str(s.getsockname()[1])
    s.close()
    return port


def get_device_count():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        return len(re.findall(r'\d+', os.environ['CUDA_VISIBLE_DEVICES']))
    return torch.cuda.device_count()
