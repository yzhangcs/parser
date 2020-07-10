# -*- coding: utf-8 -*-

from collections.abc import Iterable

import nltk


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

        # mapping from each nested field to their proper position
        self.maps = dict()
        # names of each field
        self.keys = set()
        # values of each position
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

    """
    CoNLL format template.

    Parameters
    ----------
    id : int
        Token counter, starting at 1 for each new sentence.
    form : str
        Word form or punctuation symbol.
    lemma : str
        Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
    cpos : str
        Coarse-grained part-of-speech tag, where the tagset depends on the treebank.
    pos : str
        Fine-grained part-of-speech tag, where the tagset depends on the treebank.
    feats : str
        Unordered set of syntactic and/or morphological features (depending on the particular treebank),
        or an underscore if not available.
    head : Union[int, List[int]]
        Head of the current token, which is either a value of ID,
        or zero (’0’) if the token links to the virtual root node of the sentence.
    deprel : Union[str, List[str]]
        Dependency relation to the HEAD.
    phead : int
        Projective head of current token, which is either a value of ID or zero (’0’),
        or an underscore if not available.
    pdeprel : str
        Dependency relation to the PHEAD, or an underscore if not available.

    References::
    - Sabine Buchholz and Erwin Marsi (CoNLL'06)
        CoNLL-X Shared Task on Multilingual Dependency Parsing
        http://anthology.aclweb.org/W/W06/W06-2920/302/
    """

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
              'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self,
                 ID=None, FORM=None, LEMMA=None, CPOS=None, POS=None,
                 FEATS=None, HEAD=None, DEPREL=None, PHEAD=None, PDEPREL=None):
        super().__init__()

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
    def get_arcs(cls, sequence):
        return [int(i) for i in sequence]

    @classmethod
    def get_sibs(cls, sequence):
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

    @classmethod
    def isprojective(cls, sequence):
        arcs = [(h, d) for d, h in enumerate(sequence[1:], 1) if h >= 0]
        for i, (hi, di) in enumerate(arcs):
            for hj, dj in arcs[i+1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    return False
                if lj <= hi <= rj and hj == di:
                    return False
                if (li < lj < ri or li < rj < ri) and (li - lj)*(ri - rj) > 0:
                    return False
        return True

    @classmethod
    def istree(cls, sequence, proj=False, multiroot=False):
        from supar.utils.alg import tarjan
        if proj and not cls.isprojective(sequence):
            return False
        n_roots = sum(head == 0 for head in sequence[1:])
        if n_roots == 0:
            return False
        if not multiroot and n_roots > 1:
            return False
        return next(tarjan(sequence), None) is None

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
                         if self.isprojective([0] + list(map(int, i.arcs)))]
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences


class CoNLLSentence(Sentence):

    def __init__(self, transform, lines):
        super().__init__(transform)

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
        super().__init__()

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

    @classmethod
    def binarize(cls, tree):
        tree = tree.copy(True)
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, nltk.Tree):
                nodes.extend([child for child in node])
                if len(node) > 1:
                    for i, child in enumerate(node):
                        if not isinstance(child[0], nltk.Tree):
                            node[i] = nltk.Tree(f"{node.label()}|<>", [child])
        tree.chomsky_normal_form('left', 0, 0)
        tree.collapse_unary()

        return tree

    @classmethod
    def factorize(cls, tree, delete_labels=None, equal_labels=None):
        def track(tree, i):
            label = tree.label()
            if delete_labels is not None and label in delete_labels:
                label = None
            if equal_labels is not None:
                label = equal_labels.get(label, label)
            if len(tree) == 1 and not isinstance(tree[0], nltk.Tree):
                return (i+1 if label is not None else i), []
            j, spans = i, []
            for child in tree:
                j, s = track(child, j)
                spans += s
            if label is not None and j > i:
                spans = [(i, j, label)] + spans
            return j, spans
        return track(tree, 0)[1]

    @classmethod
    def build(cls, tree, sequence):
        root = tree.label()
        leaves = [subtree for subtree in tree.subtrees()
                  if not isinstance(subtree[0], nltk.Tree)]

        def track(node):
            i, j, label = next(node)
            if j == i+1:
                children = [leaves[i]]
            else:
                children = track(node) + track(node)
            if label.endswith('|<>'):
                return children
            labels = label.split('+')
            tree = nltk.Tree(labels[-1], children)
            for label in reversed(labels[:-1]):
                tree = nltk.Tree(label, [tree])
            return [tree]
        return nltk.Tree(root, track(iter(sequence)))

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
        super().__init__(transform)

        # the values contain words, pos tags, raw trees, and spans
        # the tree is first left-binarized before factorized
        # spans are the factorization of tree traversed in pre-order
        self.values = [*zip(*tree.pos()),
                       tree,
                       Tree.factorize(Tree.binarize(tree)[0])]

    def __repr__(self):
        return self.values[-2].pformat(1000000)
