# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Tuple, Union


class Vocab(object):
    r"""
    Defines a vocabulary object that will be used to numericalize a field.

    Args:
        counter (~collections.Counter):
            :class:`~collections.Counter` object holding the frequencies of each value found in the data.
        min_freq (int):
            The minimum frequency needed to include a token in the vocabulary. Default: 1.
        specials (Tuple[str]):
            The list of special tokens (e.g., pad, unk, bos and eos) that will be prepended to the vocabulary. Default: ``[]``.
        unk_index (int):
            The index of unk token. Default: 0.

    Attributes:
        itos:
            A list of token strings indexed by their numerical identifiers.
        stoi:
            A :class:`~collections.defaultdict` object mapping token strings to numerical identifiers.
    """

    def __init__(self, counter: Counter, min_freq: int = 1, specials: Tuple = tuple(), unk_index: int = 0) -> Vocab:
        self.itos = list(specials)
        self.stoi = defaultdict(lambda: unk_index)
        self.stoi.update({token: i for i, token in enumerate(self.itos)})
        self.update([token for token, freq in counter.items() if freq >= min_freq])
        self.unk_index = unk_index
        self.n_init = len(self)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, key: Union[int, str, Iterable]) -> Union[str, int, Iterable]:
        if isinstance(key, str):
            return self.stoi[key]
        elif not isinstance(key, Iterable):
            return self.itos[key]
        elif isinstance(key[0], str):
            return [self.stoi[i] for i in key]
        else:
            return [self.itos[i] for i in key]

    def __contains__(self, token):
        return token in self.stoi

    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        stoi = defaultdict(lambda: self.unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def items(self):
        return self.stoi.items()

    def update(self, vocab: Union[Iterable[str], Vocab, Counter]) -> Vocab:
        if isinstance(vocab, Vocab):
            vocab = vocab.itos
        # NOTE: PAY CAREFUL ATTENTION TO DICT ORDER UNDER DISTRIBUTED TRAINING!
        vocab = sorted(set(vocab).difference(self.stoi))
        self.itos.extend(vocab)
        self.stoi.update({token: i for i, token in enumerate(vocab, len(self.stoi))})
        return self
