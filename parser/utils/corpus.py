# -*- coding: utf-8 -*-

from collections import namedtuple
from collections.abc import Iterable
from parser.utils.field import Field


CoNLL = namedtuple(typename='CoNLL',
                   field_names=['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
                                'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'],
                   defaults=[None]*10)


class Sentence(object):

    def __init__(self, fields, values):
        for field, value in zip(fields, values):
            if isinstance(field, Iterable):
                for j in range(len(field)):
                    setattr(self, field[j].name, value)
            else:
                setattr(self, field.name, value)
        self.fields = fields

    @property
    def values(self):
        for field in self.fields:
            if isinstance(field, Iterable):
                yield getattr(self, field[0].name)
            else:
                yield getattr(self, field.name)

    def __len__(self):
        return len(next(iter(self.values)))

    def __repr__(self):
        return '\n'.join('\t'.join(map(str, line))
                         for line in zip(*self.values)) + '\n'


class Corpus(object):

    def __init__(self, fields, sentences):
        super(Corpus, self).__init__()

        self.fields = fields
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(str(sentence) for sentence in self)

    def __getitem__(self, index):
        return self.sentences[index]

    def __getattr__(self, name):
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for sentence in self.sentences:
            yield getattr(sentence, name)

    def __setattr__(self, name, value):
        if name in ['fields', 'sentences']:
            self.__dict__[name] = value
        else:
            for i, sentence in enumerate(self.sentences):
                setattr(sentence, name, value[i])

    @classmethod
    def load(cls, path, fields):
        start, sentences = 0, []
        fields = [field if field is not None else Field(str(i))
                  for i, field in enumerate(fields)]
        with open(path, 'r') as f:
            lines = [line.strip() for line in f
                     if not line.startswith('#')
                     and (len(line) == 1 or line.split()[0].isdigit())]
        for i, line in enumerate(lines):
            if not line:
                values = list(zip(*[j.split('\t') for j in lines[start:i]]))
                sentences.append(Sentence(fields, values))
                start = i + 1

        return cls(fields, sentences)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(f"{self}\n")
