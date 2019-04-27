# -*- coding: utf-8 -*-

from collections import Counter, namedtuple

import torch


CONLL = namedtuple(typename='CONLL',
                   field_names=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS',
                                'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC'],
                   defaults=[None]*10)


class Sentence(object):
    ROOT = '<ROOT>'

    def __init__(self, *args, **kwargs):
        super(Sentence, self).__init__()

        self.conll = CONLL(*args, **kwargs)

    def __getitem__(self, index):
        return tuple(field[index] for field in self.conll)

    def __repr__(self):
        return '\n'.join('\t'.join(map(str, field)) for field in self) + '\n'

    @property
    def words(self):
        return [self.ROOT] + [word.lower() for word in self.conll.FORM]

    @property
    def heads(self):
        return [0] + list(map(int, self.conll.HEAD))

    @property
    def labels(self):
        return [self.ROOT] + list(self.conll.DEPREL)

    @heads.setter
    def heads(self, sequence):
        self.conll = self.conll._replace(HEAD=sequence)

    @labels.setter
    def labels(self, sequence):
        self.conll = self.conll._replace(DEPREL=sequence)


class Corpus(object):

    def __init__(self, sentences):
        super(Corpus, self).__init__()

        self.sentences = sentences

    def __getitem__(self, index):
        return self.sentences[index]

    @property
    def words(self):
        return Counter(word for sentence in self.sentences
                       for word in sentence.words[1:])

    @property
    def labels(self):
        return Counter(label for sentence in self.sentences
                       for label in sentence.labels[1:])

    @classmethod
    def load(cls, fname, field_names=CONLL._fields):
        start, sentences = 0, []
        with open(fname, 'r') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(line) <= 1:
                cols = zip(*[l.split() for l in lines[start:i]])
                fields = dict(zip(field_names, cols))
                sentences.append(Sentence(**fields))
                start = i + 1
        corpus = cls(sentences)

        return corpus

    def dump(self, fname):
        with open(fname, 'w') as f:
            for sentence in self:
                f.write(f"{sentence}\n")


class Embedding(object):

    def __init__(self, words, vectors):
        super(Embedding, self).__init__()

        self.words = words
        self.vectors = vectors
        self.pretrained = {w: v for w, v in zip(words, vectors)}

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return torch.tensor(self.pretrained[word])

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        reprs = [(s[0], list(map(float, s[1:]))) for s in splits]
        embedding = cls(*map(list, zip(*reprs)))

        return embedding
