# -*- coding: utf-8 -*-

import os
import socket

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


def init_device(device, local_rank=-1, backend='nccl', host=None, port=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    if torch.cuda.device_count() > 1:
        if host is None:
            host = os.environ.get('MASTER_ADDR', 'localhost')
        if port is None:
            s = socket.socket()
            s.bind(('', 0))
            port = os.environ.get('MASTER_PORT', str(s.getsockname()[1]))
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = port
        dist.init_process_group(backend)
        torch.cuda.set_device(local_rank)


def is_master():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
