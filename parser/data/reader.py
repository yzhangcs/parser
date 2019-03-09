# -*- coding: utf-8 -*-

from collections import Counter

import torch


class Corpus(object):
    ROOT = '<ROOT>'

    def __init__(self, fname):
        super(Corpus, self).__init__()

        self.fname = fname
        self.word_seqs, self.head_seqs, self.label_seqs = self.read(fname)

    @property
    def words(self):
        return Counter(w for seq in self.word_seqs for w in seq)

    @property
    def labels(self):
        return Counter(t for seq in self.label_seqs for t in seq)

    @classmethod
    def read(cls, fname):
        start = 0
        word_seqs, head_seqs, label_seqs = [], [], []
        with open(fname, 'r') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(lines[i]) <= 1:
                cols = list(zip(*[l.split() for l in lines[start:i]]))
                word_seqs.append([cls.ROOT] + list(cols[1]))
                head_seqs.append([0] + list(map(int, cols[6])))
                label_seqs.append([cls.ROOT] + list(cols[7]))
                start = i + 1

        return word_seqs, head_seqs, label_seqs


class Embedding(object):

    def __init__(self, fname):
        super(Embedding, self).__init__()

        self.fname = fname
        self.words, self.vectors = self.read(fname)
        self.pretrained = {w: v for w, v in zip(self.words, self.vectors)}

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return torch.tensor(self.pretrained[word], dtype=torch.float)

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def read(cls, fname):
        with open(fname, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        reprs = [
            (split[0], list(map(float, split[1:]))) for split in splits
        ]
        words, vectors = map(list, zip(*reprs))

        return words, vectors
