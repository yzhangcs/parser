# -*- coding: utf-8 -*-

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


def is_master():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))
    port = str(s.getsockname()[1])
    s.close()
    return port
