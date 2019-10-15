# -*- coding: utf-8 -*-

from collections.abc import Iterable
from parser.utils.common import unk


class Vocab(object):

    def __init__(self, counter, min_freq=1, specials=[]):
        self.itos = specials
        self.stoi = {token: i for i, token in enumerate(self.itos)}

        self.extend([token for token, freq in counter.items()
                     if freq >= min_freq])
        self.unk_index = self.stoi.get(unk, 0)
        self.n_init = len(self)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, key):
        return self.stoi[key]

    def __contains__(self, token):
        return token in self.stoi

    def token2id(self, sequence):
        return [self.stoi.get(token, self.unk_index) for token in sequence]

    def id2token(self, ids):
        if isinstance(ids, Iterable):
            return [self.itos[i] for i in ids]
        else:
            return self.itos[ids]

    def extend(self, tokens):
        self.itos.extend(sorted(set(tokens).difference(self.stoi)))
        self.stoi = {token: i for i, token in enumerate(self.itos)}
