# -*- coding: utf-8 -*-

from collections.abc import Iterable

import nltk
from supar.utils.fn import binarize, factorize, isprojective


class Transform(object):

    fields = []

    def __init__(self):
        self.training = True

    def __call__(self, sentences):
        pairs = dict()
        for field in self:
            if field not in self.src and field not in self.tgt:
                continue
            if not self.training and field in self.tgt:
                continue
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None:
                    pairs[f] = f.transform([getattr(i, f.name)
                                            for i in sentences])

        return pairs

    def __getitem__(self, index):
        return getattr(self, self.fields[index])

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(False)

    def append(self, field):
        self.fields.append(field.name)
        setattr(self, field.name, field)

    @property
    def src(self):
        raise AttributeError

    @property
    def tgt(self):
        raise AttributeError

    def save(self, path, sentences):
        with open(path, 'w') as f:
            f.write('\n'.join([str(i) for i in sentences]) + '\n')


class Sentence(object):

    def __init__(self, transform):
        self.transform = transform

        # the mapping from each nested field to their proper position
        self.maps = dict()
        # the names of each field
        self.keys = set()
        # the values of each position
        self.values = []
        for i, field in enumerate(self.transform):
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None:
                    self.maps[f.name] = i
                    self.keys.add(f.name)

    def __len__(self):
        return len(self.values[0])

    def __contains__(self, key):
        return key in self.keys

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return self.values[self.maps[name]]

    def __setattr__(self, name, value):
        if 'keys' in self.__dict__ and name in self:
            index = self.maps[name]
            if index >= len(self.values):
                self.__dict__[name] = value
            else:
                self.values[index] = value
        else:
            self.__dict__[name] = value


class CoNLL(Transform):

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
              'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self,
                 ID=None, FORM=None, LEMMA=None, CPOS=None, POS=None,
                 FEATS=None, HEAD=None, DEPREL=None, PHEAD=None, PDEPREL=None):
        super(CoNLL, self).__init__()

        self.ID = ID
        self.FORM = FORM
        self.LEMMA = LEMMA
        self.CPOS = CPOS
        self.POS = POS
        self.FEATS = FEATS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.PHEAD = PHEAD
        self.PDEPREL = PDEPREL

    @property
    def src(self):
        return self.FORM, self.CPOS

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL

    @classmethod
    def toconll(cls, tokens):
        if isinstance(tokens[0], str):
            s = '\n'.join([f"{i}\t{word}\t" + '\t'.join(['_']*8)
                           for i, word in enumerate(tokens, 1)])
        else:
            s = '\n'.join([f"{i}\t{word}\t_\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, tag) in enumerate(tokens, 1)])
        return s + '\n'

    @classmethod
    def numericalize(cls, sequence):
        return [int(i) for i in sequence]

    @classmethod
    def numericalize_sibs(cls, sequence):
        sibs = [-1] * (len(sequence) + 1)
        heads = [0] + [int(i) for i in sequence]

        for i in range(1, len(heads)):
            hi = heads[i]
            for j in range(i + 1, len(heads)):
                hj = heads[j]
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i] = j
                    else:
                        sibs[j] = i
                    break
        return sibs[1:]

    def load(self, data, proj=False, max_len=None, **kwargs):
        start, sentences = 0, []
        if isinstance(data, str):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]
        else:
            data = [data] if isinstance(data[0], str) else data
            lines = '\n'.join([self.toconll(i) for i in data]).split('\n')
        for i, line in enumerate(lines):
            if not line:
                sentences.append(CoNLLSentence(self, lines[start:i]))
                start = i + 1
        if proj:
            sentences = [i for i in sentences
                         if isprojective([0] + list(map(int, i.arcs)))]
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences


class CoNLLSentence(Sentence):

    def __init__(self, transform, lines):
        super(CoNLLSentence, self).__init__(transform)

        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i-1] = line
            else:
                self.annotations[len(self.values)] = line
                self.values.append(value)
        self.values = list(zip(*self.values))

    def __repr__(self):
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values))}}
        return '\n'.join(merged.values()) + '\n'


class Tree(Transform):

    root = ''
    fields = ['WORD', 'POS', 'TREE', 'CHART']

    def __init__(self, WORD=None, POS=None, TREE=None, CHART=None):
        super(Tree, self).__init__()

        self.WORD = WORD
        self.POS = POS
        self.TREE = TREE
        self.CHART = CHART

    @property
    def src(self):
        return self.WORD, self.POS, self.TREE

    @property
    def tgt(self):
        return self.CHART,

    @classmethod
    def totree(cls, tokens, root=''):
        if isinstance(tokens[0], str):
            tokens = [(token, '_') for token in tokens]
        tree = ' '.join([f"({pos} {word})" for word, pos in tokens])
        return nltk.Tree.fromstring(f"({root} {tree})")

    def load(self, data, max_len=None, **kwargs):
        if isinstance(data, str):
            with open(data, 'r') as f:
                trees = [nltk.Tree.fromstring(string) for string in f]
            self.root = trees[0].label()
        else:
            data = [data] if isinstance(data[0], str) else data
            trees = [self.totree(i, self.root) for i in data]
        sentences = [TreeSentence(self, tree) for tree in trees
                     if not len(tree) == 1
                     or isinstance(tree[0][0], nltk.Tree)]
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences


class TreeSentence(Sentence):

    def __init__(self, transform, tree):
        super(TreeSentence, self).__init__(transform)

        # the values contain words, pos tags, raw trees, and spans
        # the tree is first left-binarized before factorized
        # spans are the factorization of tree traversed in pre-order
        self.values = [*zip(*tree.pos()), tree, factorize(binarize(tree)[0])]

    def __repr__(self):
        return self.values[-2].pformat(1000000)
