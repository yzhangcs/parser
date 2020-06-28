# -*- coding: utf-8 -*-

import torch.nn as nn


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __init__(self, module, **kwargs):
        super(DistributedDataParallel, self).__init__(module, **kwargs)

    def __getattr__(self, name):
        wrapped = super().__getattr__('module')
        if hasattr(wrapped, name):
            return getattr(wrapped, name)
        return super().__getattr__(name)
