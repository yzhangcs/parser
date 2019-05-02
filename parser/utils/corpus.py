# -*- coding: utf-8 -*-

from collections import namedtuple


Sentence = namedtuple(typename='Sentence',
                      field_names=['ID', 'FORM', 'LEMMA', 'CPOS',
                                   'POS', 'FEATS', 'HEAD', 'DEPREL',
                                   'PHEAD', 'PDEPREL'],
                      defaults=[None]*10)


class Corpus(object):
    ROOT = '<ROOT>'

    def __init__(self, sentences):
        super(Corpus, self).__init__()

        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(
            '\n'.join('\t'.join(map(str, i))
                      for i in zip(*(f for f in sentence if f))) + '\n'
            for sentence in self
        )

    def __getitem__(self, index):
        return self.sentences[index]

    @property
    def words(self):
        return [[self.ROOT] + list(sentence.FORM) for sentence in self]

    @property
    def tags(self):
        return [[self.ROOT] + list(sentence.CPOS) for sentence in self]

    @property
    def heads(self):
        return [[0] + list(map(int, sentence.HEAD)) for sentence in self]

    @property
    def rels(self):
        return [[self.ROOT] + list(sentence.DEPREL) for sentence in self]

    @heads.setter
    def heads(self, sequences):
        self.sentences = [sentence._replace(HEAD=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @rels.setter
    def rels(self, sequences):
        self.sentences = [sentence._replace(DEPREL=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @classmethod
    def load(cls, fname):
        start, sentences = 0, []
        with open(fname, 'r') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(line) <= 1:
                sentence = Sentence(*zip(*[l.split() for l in lines[start:i]]))
                sentences.append(sentence)
                start = i + 1
        corpus = cls(sentences)

        return corpus

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write(f"{self}\n")
