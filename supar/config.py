# -*- coding: utf-8 -*-

from argparse import Namespace
from ast import literal_eval
from configparser import ConfigParser


class Config(ConfigParser):

    def __init__(self, path=[]):
        super(Config, self).__init__()

        self.read(path)
        self.namespace = Namespace()
        self.update(dict((name, literal_eval(value))
                         for section in self.sections()
                         for name, value in self.items(section)))

    def __repr__(self):
        s = line = "-" * 15 + "-+-" + "-" * 25 + "\n"
        s += f"{'Param':15} | {'Value':^25}\n" + line
        for name, value in vars(self.namespace).items():
            s += f"{name:15} | {str(value):^25}\n"
        s += line

        return s

    def __getitem__(self, key):
        return getattr(self.namespace, key)

    def __getattr__(self, attr):
        return getattr(self.namespace, attr)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def keys(self):
        return vars(self.namespace).keys()

    def update(self, kwargs):
        for name, value in kwargs.items():
            setattr(self.namespace, name, value)

        return self
