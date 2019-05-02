# -*- coding: utf-8 -*-

from ast import literal_eval
from configparser import ConfigParser


class Config(object):

    def __init__(self, fname):
        super(Config, self).__init__()

        self.config = ConfigParser()
        self.config.read(fname)
        self.kwargs = dict((option, literal_eval(value))
                           for section in self.config.sections()
                           for option, value in self.config.items(section))

    def __repr__(self):
        info = f"{self.__class__.__name__}:\n"
        for i, (option, value) in enumerate(self.kwargs.items()):
            info += f"{option:15} {value:<25}" + ('\n' if i % 2 > 0 else '')
        if i % 2 == 0:
            info += '\n'

        return info

    def __getattr__(self, attr):
        return self.kwargs.get(attr, None)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update(self, kwargs):
        self.kwargs.update(kwargs)
