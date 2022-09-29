# -*- coding: utf-8 -*-

import argparse
import os
from ast import literal_eval
from configparser import ConfigParser

import supar
from omegaconf import OmegaConf
from supar.utils.fn import download


class Config(object):

    def __init__(self, **kwargs):
        super(Config, self).__init__()

        self.update(kwargs)

    def __repr__(self):
        return OmegaConf.to_yaml(vars(self))

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def update(self, kwargs):
        for key in ('self', 'cls', '__class__'):
            kwargs.pop(key, None)
        kwargs.update(kwargs.pop('kwargs', dict()))
        for name, value in kwargs.items():
            setattr(self, name, value)
        return self

    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default

    def pop(self, key, val=None):
        return self.__dict__.pop(key, val)

    @classmethod
    def load(cls, conf='', unknown=None, **kwargs):
        config = ConfigParser()
        config.read(conf if not conf or os.path.exists(conf) else download(supar.CONFIG['github'].get(conf, conf)))
        config = dict((name, literal_eval(value))
                      for section in config.sections()
                      for name, value in config.items(section))
        if unknown is not None:
            parser = argparse.ArgumentParser()
            for name, value in config.items():
                parser.add_argument('--'+name.replace('_', '-'), type=type(value), default=value)
            config.update(vars(parser.parse_args(unknown)))
        config.update(kwargs)
        return cls(**config)
