# -*- coding: utf-8 -*-

from collections import Counter
from parser.utils.vocab import Vocab

import torch


class Field(object):

    def __init__(self, name, pad=None, unk=None, bos=None, eos=None,
                 lower=False, use_vocab=True, tokenize=None, fn=None):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn

        self.specials = [token for token in [pad, unk, bos, eos]
                         if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        s += f", ".join(params)
        s += f")"

        return s

    @property
    def pad_index(self):
        return self.specials.index(self.pad) if self.pad is not None else 0

    @property
    def unk_index(self):
        return self.specials.index(self.unk) if self.unk is not None else 0

    @property
    def bos_index(self):
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        return self.specials.index(self.eos)

    def transform(self, sequence):
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]
        if self.fn is not None:
            sequence = [self.fn(token) for token in sequence]

        return sequence

    def build(self, corpus, min_freq=1, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter(token for sequence in sequences
                          for token in self.transform(sequence))
        self.vocab = Vocab(counter, min_freq, self.specials)

        if not embed:
            self.embed = None
        else:
            tokens = self.transform(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(tokens)] = embed.vectors
            self.embed /= torch.std(self.embed)

    def numericalize(self, sequences):
        sequences = [self.transform(sequence) for sequence in sequences]
        if self.use_vocab:
            sequences = [self.vocab.token2id(sequence)
                         for sequence in sequences]
        if self.bos:
            sequences = [[self.bos_index] + sequence for sequence in sequences]
        if self.eos:
            sequences = [sequence + [self.eos_index] for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]

        return sequences


class CharField(Field):

    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else -1
        super(CharField, self).__init__(*args, **kwargs)

    def build(self, corpus, min_freq=1, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter(char for sequence in sequences for token in sequence
                          for char in self.transform(token))
        self.vocab = Vocab(counter, min_freq, self.specials)

        if not embed:
            self.embed = None
        else:
            tokens = self.transform(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(tokens)] = embed.vectors

    def numericalize(self, sequences):
        sequences = [[self.transform(token) for token in sequence]
                     for sequence in sequences]
        if self.fix_len <= 0:
            self.fix_len = max(len(token) for sequence in sequences
                               for token in sequence)
        if self.use_vocab:
            sequences = [[self.vocab.token2id(token) for token in sequence]
                         for sequence in sequences]
        if self.bos:
            sequences = [[self.vocab.token2id(self.bos)] + sequence
                         for sequence in sequences]
        if self.eos:
            sequences = [sequence + [self.vocab.token2id(self.eos)]
                         for sequence in sequences]
        sequences = [
            torch.tensor([ids[:self.fix_len] + [0] * (self.fix_len - len(ids))
                          for ids in sequence])
            for sequence in sequences
        ]

        return sequences


class BertField(Field):

    def numericalize(self, sequences):
        subwords, lens = [], []
        sequences = [([self.bos] if self.bos else []) + list(sequence) +
                     ([self.eos] if self.eos else [])
                     for sequence in sequences]

        for sequence in sequences:
            sequence = [self.transform(token) for token in sequence]
            sequence = [piece if piece else self.transform(self.pad)
                        for piece in sequence]
            subwords.append(sum(sequence, []))
            lens.append(torch.tensor([len(piece) for piece in sequence]))
        subwords = [torch.tensor(pieces) for pieces in subwords]
        mask = [torch.ones(len(pieces)).ge(0) for pieces in subwords]

        return list(zip(subwords, lens, mask))
