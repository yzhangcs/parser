# -*- coding: utf-8 -*-

from collections import namedtuple
from collections.abc import Iterable

from nltk.tree import Tree
from supar.utils.field import Field
from supar.utils.fn import binarize, factorize, isprojective, toconll, totree

CoNLL = namedtuple(typename='CoNLL',
                   field_names=['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
                                'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'],
                   )  # defaults=[None]*10)
CoNLL.__new__.__defaults__ = (None,) * 10


Treebank = namedtuple(typename='Treebank',
                      field_names=['TREE', 'WORD', 'POS', 'CHART'],
                      )  # defaults=[None]*4)
Treebank.__new__.__defaults__ = (None,) * 4


class CoNLLSentence(object):

    def __init__(self, fields, lines):
        self.annotations = dict()
        values = []
        for i, line in enumerate(lines):
            if line.startswith('#'):
                self.annotations[-i-1] = line
            else:
                value = line.split('\t')
                if value[0].isdigit():
                    values.append(value)
                    self.annotations[int(value[0])] = ''  # placeholder
                else:
                    self.annotations[-i] = line
        for field, value in zip(fields, list(zip(*values))):
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
        merged = {**self.annotations,
                  **{i+1: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values))}}
        return '\n'.join(merged.values()) + '\n'


class TreebankSentence(object):

    def __init__(self, fields, tree):
        self.tree = tree
        self.fields = [field if isinstance(field, Iterable) else [field]
                       for field in fields]
        self.values = [tree, *zip(*tree.pos()), factorize(binarize(tree)[0])]
        for field, value in zip(self.fields, self.values):
            for f in field:
                setattr(self, f.name, value)

    def __len__(self):
        return len(list(self.tree.leaves()))

    def __repr__(self):
        return self.tree.pformat(1000000)

    def __setattr__(self, name, value):
        if isinstance(value, Tree) and hasattr(self, name):
            tree = getattr(self, name)
            tree.clear()
            tree.extend([value[0]])
        else:
            self.__dict__[name] = value


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

    def save(self, path):
        with open(path, 'w') as f:
            f.write(f"{self}\n")


class CoNLLCorpus(Corpus):

    @classmethod
    def load(cls, data, fields, proj=False, max_len=None):
        start, sentences = 0, []
        fields = [field or Field(str(i)) for i, field in enumerate(fields)]
        if isinstance(data, str):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]
        else:
            data = [data] if isinstance(data[0], str) else data
            lines = '\n'.join([toconll(i) for i in data]).split('\n')
        for i, line in enumerate(lines):
            if not line:
                sentences.append(CoNLLSentence(fields, lines[start:i]))
                start = i + 1
        if proj:
            sentences = [sentence for sentence in sentences
                         if isprojective([0] + list(map(int, sentence.arcs)))]
        if max_len is not None:
            sentences = [sentence for sentence in sentences
                         if len(sentence) < max_len]

        return cls(fields, sentences)


class TreebankCorpus(Corpus):

    @classmethod
    def load(cls, data, fields, max_len=None):
        fields = [field or Field(str(i)) for i, field in enumerate(fields)]
        if isinstance(data, str):
            with open(data, 'r') as f:
                trees = [Tree.fromstring(string) for string in f]
        else:
            data = [data] if isinstance(data[0], str) else data
            trees = [totree(i) for i in data]
        sentences = [TreebankSentence(fields, tree) for tree in trees
                     if not len(tree) == 1 or isinstance(tree[0][0], Tree)]
        if max_len is not None:
            sentences = [sentence for sentence in sentences
                         if len(sentence) < max_len]

        return cls(fields, sentences)
