# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from io import StringIO
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

import nltk
import torch
from supar.utils.fn import debinarize
from supar.utils.logging import logger, progress_bar
from supar.utils.tokenizer import Tokenizer
from torch.distributions.utils import lazy_property

if TYPE_CHECKING:
    from supar.utils import Field


class Transform(object):
    r"""
    A Transform object corresponds to a specific data format, which holds several instances of data fields
    that provide instructions for preprocessing and numericalization, etc.

    Attributes:
        training (bool):
            Sets the object in training mode.
            If ``False``, some data fields not required for predictions won't be returned.
            Default: ``True``.
    """

    fields = []

    def __init__(self):
        self.training = True

    def __len__(self):
        return len(self.fields)

    def __repr__(self):
        s = '\n' + '\n'.join([f" {f}" for f in self.flattened_fields]) + '\n'
        return f"{self.__class__.__name__}({s})"

    def __call__(self, sentences: Iterable[Sentence]) -> Iterable[Sentence]:
        return [sentence.numericalize(self.flattened_fields) for sentence in progress_bar(sentences)]

    def __getitem__(self, index):
        return getattr(self, self.fields[index])

    @property
    def flattened_fields(self):
        flattened = []
        for field in self:
            if field not in self.src and field not in self.tgt:
                continue
            if not self.training and field in self.tgt:
                continue
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None:
                    flattened.append(f)
        return flattened

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


class CoNLL(Transform):
    r"""
    The CoNLL object holds ten fields required for CoNLL-X data format :cite:`buchholz-marsi-2006-conll`.
    Each field can be bound to one or more :class:`~supar.utils.field.Field` objects. For example,
    ``FORM`` can contain both :class:`~supar.utils.field.Field` and :class:`~supar.utils.field.SubwordField`
    to produce tensors for words and subwords.

    Attributes:
        ID:
            Token counter, starting at 1.
        FORM:
            Words in the sentence.
        LEMMA:
            Lemmas or stems (depending on the particular treebank) of words, or underscores if not available.
        CPOS:
            Coarse-grained part-of-speech tags, where the tagset depends on the treebank.
        POS:
            Fine-grained part-of-speech tags, where the tagset depends on the treebank.
        FEATS:
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or underscores if not available.
        HEAD:
            Heads of the tokens, which are either values of ID or zeros.
        DEPREL:
            Dependency relations to the HEAD.
        PHEAD:
            Projective heads of tokens, which are either values of ID or zeros, or underscores if not available.
        PDEPREL:
            Dependency relations to the PHEAD, or underscores if not available.
    """

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(
        self,
        ID: Optional[Union[Field, Iterable[Field]]] = None,
        FORM: Optional[Union[Field, Iterable[Field]]] = None,
        LEMMA: Optional[Union[Field, Iterable[Field]]] = None,
        CPOS: Optional[Union[Field, Iterable[Field]]] = None,
        POS: Optional[Union[Field, Iterable[Field]]] = None,
        FEATS: Optional[Union[Field, Iterable[Field]]] = None,
        HEAD: Optional[Union[Field, Iterable[Field]]] = None,
        DEPREL: Optional[Union[Field, Iterable[Field]]] = None,
        PHEAD: Optional[Union[Field, Iterable[Field]]] = None,
        PDEPREL: Optional[Union[Field, Iterable[Field]]] = None
    ) -> CoNLL:
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
        return self.FORM, self.LEMMA, self.CPOS, self.POS, self.FEATS

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL, self.PHEAD, self.PDEPREL

    @classmethod
    def get_arcs(cls, sequence, placeholder='_'):
        return [-1 if i == placeholder else int(i) for i in sequence]

    @classmethod
    def get_sibs(cls, sequence, placeholder='_'):
        sibs = [[0] * (len(sequence) + 1) for _ in range(len(sequence) + 1)]
        heads = [0] + [-1 if i == placeholder else int(i) for i in sequence]

        for i, hi in enumerate(heads[1:], 1):
            for j, hj in enumerate(heads[i+1:], i + 1):
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i][hi] = j
                    else:
                        sibs[j][hj] = i
                    break
        return sibs[1:]

    @classmethod
    def get_edges(cls, sequence):
        edges = [[0]*(len(sequence)+1) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edges[i][int(pair.split(':')[0])] = 1
        return edges

    @classmethod
    def get_labels(cls, sequence):
        labels = [[None]*(len(sequence)+1) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edge, label = pair.split(':', 1)
                    labels[i][int(edge)] = label
        return labels

    @classmethod
    def build_relations(cls, chart):
        sequence = ['_'] * len(chart)
        for i, row in enumerate(chart):
            pairs = [(j, label) for j, label in enumerate(row) if label is not None]
            if len(pairs) > 0:
                sequence[i] = '|'.join(f"{head}:{label}" for head, label in pairs)
        return sequence

    @classmethod
    def toconll(cls, tokens: List[Union[str, Tuple]]) -> str:
        r"""
        Converts a list of tokens to a string in CoNLL-X format.
        Missing fields are filled with underscores.

        Args:
            tokens (list[str] or list[tuple]):
                This can be either a list of words, word/pos pairs or word/lemma/pos triples.

        Returns:
            A string in CoNLL-X format.

        Examples:
            >>> print(CoNLL.toconll(['She', 'enjoys', 'playing', 'tennis', '.']))
            1       She     _       _       _       _       _       _       _       _
            2       enjoys  _       _       _       _       _       _       _       _
            3       playing _       _       _       _       _       _       _       _
            4       tennis  _       _       _       _       _       _       _       _
            5       .       _       _       _       _       _       _       _       _

            >>> print(CoNLL.toconll([('She',     'she',    'PRP'),
                                     ('enjoys',  'enjoy',  'VBZ'),
                                     ('playing', 'play',   'VBG'),
                                     ('tennis',  'tennis', 'NN'),
                                     ('.',       '_',      '.')]))
            1       She     she     PRP     _       _       _       _       _       _
            2       enjoys  enjoy   VBZ     _       _       _       _       _       _
            3       playing play    VBG     _       _       _       _       _       _
            4       tennis  tennis  NN      _       _       _       _       _       _
            5       .       _       .       _       _       _       _       _       _

        """

        if isinstance(tokens[0], str):
            s = '\n'.join([f"{i}\t{word}\t" + '\t'.join(['_']*8)
                           for i, word in enumerate(tokens, 1)])
        elif len(tokens[0]) == 2:
            s = '\n'.join([f"{i}\t{word}\t_\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, tag) in enumerate(tokens, 1)])
        elif len(tokens[0]) == 3:
            s = '\n'.join([f"{i}\t{word}\t{lemma}\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, lemma, tag) in enumerate(tokens, 1)])
        else:
            raise RuntimeError(f"Invalid sequence {tokens}. Only list of str or list of word/pos/lemma tuples are support.")
        return s + '\n'

    @classmethod
    def isprojective(cls, sequence: List[int]) -> bool:
        r"""
        Checks if a dependency tree is projective.
        This also works for partial annotation.

        Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
        which are hard to detect in the scenario of partial annotation.

        Args:
            sequence (list[int]):
                A list of head indices.

        Returns:
            ``True`` if the tree is projective, ``False`` otherwise.

        Examples:
            >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
            False
            >>> CoNLL.isprojective([3, -1, 2])
            False
        """

        pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
        for i, (hi, di) in enumerate(pairs):
            for hj, dj in pairs[i+1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    return False
                if lj <= hi <= rj and hj == di:
                    return False
                if (li < lj < ri or li < rj < ri) and (li - lj)*(ri - rj) > 0:
                    return False
        return True

    @classmethod
    def istree(cls, sequence: List[int], proj: bool = False, multiroot: bool = False) -> bool:
        r"""
        Checks if the arcs form an valid dependency tree.

        Args:
            sequence (list[int]):
                A list of head indices.
            proj (bool):
                If ``True``, requires the tree to be projective. Default: ``False``.
            multiroot (bool):
                If ``False``, requires the tree to contain only a single root. Default: ``True``.

        Returns:
            ``True`` if the arcs form an valid tree, ``False`` otherwise.

        Examples:
            >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
            True
            >>> CoNLL.istree([3, 0, 0, 3], proj=True)
            False
        """

        from supar.structs.fn import tarjan
        if proj and not cls.isprojective(sequence):
            return False
        n_roots = sum(head == 0 for head in sequence)
        if n_roots == 0:
            return False
        if not multiroot and n_roots > 1:
            return False
        if any(i == head for i, head in enumerate(sequence, 1)):
            return False
        return next(tarjan(sequence), None) is None

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        proj: bool = False,
        max_len: Optional[int] = None,
        **kwargs
    ) -> Iterable[CoNLLSentence]:
        r"""
        Loads the data in CoNLL-X format.
        Also supports for loading data from CoNLL-U file with comments and non-integer IDs.

        Args:
            data (str or Iterable):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            proj (bool):
                If ``True``, discards all non-projective sentences. Default: ``False``.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`CoNLLSentence` instances.
        """

        isconll = False
        if lang is not None:
            tokenizer = Tokenizer(lang)
        if isinstance(data, str) and os.path.exists(data):
            f = open(data)
            if data.endswith('.txt'):
                lines = (i
                         for s in f
                         if len(s) > 1
                         for i in StringIO(self.toconll(s.split() if lang is None else tokenizer(s)) + '\n'))
            else:
                lines, isconll = f, True
        else:
            if lang is not None:
                data = [tokenizer(s) for s in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = (i for s in data for i in StringIO(self.toconll(s) + '\n'))

        index, sentence = 0, []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                sentence = CoNLLSentence(self, sentence, index)
                if isconll and proj and not self.isprojective(list(map(int, sentence.arcs))):
                    logger.warning(f"Sentence {index} is not projective. Discarding it!")
                elif max_len is not None and len(sentence) >= max_len:
                    logger.warning(f"Sentence {index} has {len(sentence)} tokens, exceeding {max_len}. Discarding it!")
                else:
                    yield sentence
                    index += 1
                sentence = []
            else:
                sentence.append(line)


class Tree(Transform):
    r"""
    The Tree object factorize a constituency tree into four fields,
    each associated with one or more :class:`~supar.utils.field.Field` objects.

    Attributes:
        WORD:
            Words in the sentence.
        POS:
            Part-of-speech tags, or underscores if not available.
        TREE:
            The raw constituency tree in :class:`nltk.tree.Tree` format.
        CHART:
            The factorized sequence of binarized tree traversed in pre-order.
    """

    root = ''
    fields = ['WORD', 'POS', 'TREE', 'CHART']

    def __init__(
        self,
        WORD: Optional[Union[Field, Iterable[Field]]] = None,
        POS: Optional[Union[Field, Iterable[Field]]] = None,
        TREE: Optional[Union[Field, Iterable[Field]]] = None,
        CHART: Optional[Union[Field, Iterable[Field]]] = None
    ) -> Tree:
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
    def totree(
        cls,
        tokens: List[Union[str, Tuple]],
        root: str = '',
    ) -> nltk.Tree:
        r"""
        Converts a list of tokens to a :class:`nltk.tree.Tree`.
        Missing fields are filled with underscores.

        Args:
            tokens (list[str] or list[tuple]):
                This can be either a list of words or word/pos pairs.
            root (str):
                The root label of the tree. Default: ''.

        Returns:
            A :class:`nltk.tree.Tree` object.

        Examples:
            >>> print(Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], 'TOP'))
            (TOP ( (_ She)) ( (_ enjoys)) ( (_ playing)) ( (_ tennis)) ( (_ .)))
        """

        if isinstance(tokens[0], str):
            tokens = [(token, '_') for token in tokens]
        return nltk.Tree(root, [nltk.Tree('', [nltk.Tree(pos, [word])]) for word, pos in tokens])

    @classmethod
    def binarize(cls, tree: nltk.Tree) -> nltk.Tree:
        r"""
        Conducts binarization over the tree.

        First, the tree is transformed to satisfy `Chomsky Normal Form (CNF)`_.
        Here we call :meth:`~nltk.tree.Tree.chomsky_normal_form` to conduct left-binarization.
        Second, all unary productions in the tree are collapsed.

        Args:
            tree (nltk.tree.Tree):
                The tree to be binarized.

        Returns:
            The binarized tree.

        Examples:
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ She))
                                                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                                (_ .)))
                                            ''')
            >>> print(Tree.binarize(tree))
            (TOP
              (S
                (S|<>
                  (NP (_ She))
                  (VP
                    (VP|<> (_ enjoys))
                    (S::VP (VP|<> (_ playing)) (NP (_ tennis)))))
                (S|<> (_ .))))

        .. _Chomsky Normal Form (CNF):
            https://en.wikipedia.org/wiki/Chomsky_normal_form
        """

        tree = tree.copy(True)
        if len(tree) == 1 and not isinstance(tree[0][0], nltk.Tree):
            tree[0] = nltk.Tree(f"{tree.label()}|<>", [tree[0]])
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
        tree.collapse_unary(joinChar='::')

        return tree

    @classmethod
    def factorize(
        cls,
        tree: nltk.Tree,
        delete_labels: Optional[Set[str]] = None,
        equal_labels: Optional[Dict[str, str]] = None
    ) -> List[Tuple]:
        r"""
        Factorizes the tree into a sequence.
        The tree is traversed in pre-order.

        Args:
            tree (nltk.tree.Tree):
                The tree to be factorized.
            delete_labels (set[str]):
                A set of labels to be ignored. This is used for evaluation.
                If it is a pre-terminal label, delete the word along with the brackets.
                If it is a non-terminal label, just delete the brackets (don't delete children).
                In `EVALB`_, the default set is:
                {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
                Default: ``None``.
            equal_labels (dict[str, str]):
                The key-val pairs in the dict are considered equivalent (non-directional). This is used for evaluation.
                The default dict defined in `EVALB`_ is: {'ADVP': 'PRT'}
                Default: ``None``.

        Returns:
            The sequence of the factorized tree.

        Examples:
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ She))
                                                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                                (_ .)))
                                            ''')
            >>> Tree.factorize(tree)
            [(0, 5, 'TOP'), (0, 5, 'S'), (0, 1, 'NP'), (1, 4, 'VP'), (2, 4, 'S'), (2, 4, 'VP'), (3, 4, 'NP')]
            >>> Tree.factorize(tree, delete_labels={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''})
            [(0, 5, 'S'), (0, 1, 'NP'), (1, 4, 'VP'), (2, 4, 'S'), (2, 4, 'VP'), (3, 4, 'NP')]

        .. _EVALB:
            https://nlp.cs.nyu.edu/evalb/
        """

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
    def build(cls, tree: nltk.Tree, sequence: List[Tuple]) -> nltk.Tree:
        r"""
        Builds a constituency tree from the sequence. The sequence is generated in pre-order.
        During building the tree, the sequence is de-binarized to the original format (i.e.,
        the suffixes ``|<>`` are ignored, the collapsed labels are recovered).

        Args:
            tree (nltk.tree.Tree):
                An empty tree that provides a base for building a result tree.
            sequence (list[tuple]):
                A list of tuples used for generating a tree.
                Each tuple consits of the indices of left/right boundaries and label of the constituent.

        Returns:
            A result constituency tree.

        Examples:
            >>> tree = Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], 'TOP')
            >>> sequence = [(0, 5, 'S'), (0, 4, 'S|<>'), (0, 1, 'NP'), (1, 4, 'VP'), (1, 2, 'VP|<>'),
                            (2, 4, 'S::VP'), (2, 3, 'VP|<>'), (3, 4, 'NP'), (4, 5, 'S|<>')]
            >>> print(Tree.build(tree, sequence))
            (TOP
              (S
                (NP (_ She))
                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                (_ .)))
        """

        root = tree.label()
        leaves = [subtree for subtree in tree.subtrees()
                  if not isinstance(subtree[0], nltk.Tree)]

        def track(node):
            i, j, label = next(node)
            if j == i+1:
                children = [leaves[i]]
            else:
                children = track(node) + track(node)
            if label is None or label.endswith('|<>'):
                return children
            labels = label.split('::')
            tree = nltk.Tree(labels[-1], children)
            for label in reversed(labels[:-1]):
                tree = nltk.Tree(label, [tree])
            return [tree]
        return nltk.Tree(root, track(iter(sequence)))

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        max_len: Optional[int] = None,
        **kwargs
    ) -> List[TreeSentence]:
        r"""
        Args:
            data (str or Iterable):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`TreeSentence` instances.
        """

        if lang is not None:
            tokenizer = Tokenizer(lang)
        if isinstance(data, str) and os.path.exists(data):
            if data.endswith('.txt'):
                data = (s.split() if lang is None else tokenizer(s) for s in open(data) if len(s) > 1)
            else:
                data = open(data)
        else:
            if lang is not None:
                data = [tokenizer(i) for i in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data

        index = 0
        for s in data:
            try:
                tree = nltk.Tree.fromstring(s) if isinstance(s, str) else self.totree(s, self.root)
                sentence = TreeSentence(self, tree, index)
            except ValueError:
                logger.warning(f"Error found while converting Sentence {index} to a tree:\n{s}\nDiscarding it!")
                continue
            if max_len is not None and len(sentence) >= max_len:
                logger.warning(f"Sentence {index} has {len(sentence)} tokens, exceeding {max_len}. Discarding it!")
            else:
                yield sentence
                index += 1
        self.root = tree.label()


class Batch(object):

    def __init__(self, sentences: Iterable[Sentence]) -> Batch:
        self.sentences = sentences

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([f"{name}" for name in self.names])})'

    def __getattr__(self, name):
        return [getattr(s, name) for s in self.sentences]

    def __setattr__(self, name: str, value: Iterable[Any]):
        if name not in ('sentences', 'names'):
            for s, v in zip(self.sentences, value):
                setattr(s, name, v)
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @lazy_property
    def names(self):
        return [name for name in self.sentences[0].fields]

    @lazy_property
    def lens(self):
        return torch.tensor([len(i) for i in self.sentences]).to(self.device, non_blocking=True)

    @lazy_property
    def mask(self):
        return self.lens.unsqueeze(-1).gt(self.lens.new_tensor(range(self.lens.max())))

    def compose(self, transform: Transform):
        return [f.compose([s.fields[f.name] for s in self.sentences]) for f in transform.flattened_fields]

    def pin_memory(self):
        for s in self.sentences:
            for i in s.fields.values():
                if isinstance(i, torch.Tensor):
                    i.pin_memory()
        return self


class Sentence(object):

    def __init__(self, transform, index: Optional[int] = None) -> Sentence:
        self.index = index
        # mapping from each nested field to their proper position
        self.maps = dict()
        # original values and numericalized values of each position
        self.values, self.fields = [], {}
        for i, field in enumerate(transform):
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None:
                    self.maps[f.name] = i
                    self.fields[f.name] = None

    def __contains__(self, name):
        return name in self.fields

    def __getattr__(self, name):
        if name in self.fields:
            return self.values[self.maps[name]]
        raise AttributeError(f"`{name}` not found")

    def __setattr__(self, name, value):
        if 'fields' in self.__dict__ and name in self:
            index = self.maps[name]
            if index >= len(self.values):
                self.__dict__[name] = value
            else:
                self.values[index] = value
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        try:
            return len(next(iter(self.fields.values())))
        except Exception:
            raise AttributeError("Cannot get size of a sentence with no fields")

    @lazy_property
    def size(self):
        # number of subwords in the sentence, mainly used for clustering
        # this is equivalent to __len__ for normal tokens without further subword tokenization
        try:
            return next(iter(self.fields.values())).ne(self.pad_index).sum().item()
        except Exception:
            raise AttributeError("Cannot get size of a sentence with no fields")

    def numericalize(self, fields):
        for f in fields:
            self.fields[f.name] = next(f.transform([getattr(self, f.name)]))
        self.pad_index = fields[0].pad_index
        return self

    @classmethod
    def from_cache(cls, fbin: str, pos: Tuple[int, int]) -> Sentence:
        return debinarize(fbin, pos)


class CoNLLSentence(Sentence):
    r"""
    Sencence in CoNLL-X format.

    Args:
        transform (CoNLL):
            A :class:`~supar.utils.transform.CoNLL` object.
        lines (list[str]):
            A list of strings composing a sentence in CoNLL-X format.
            Comments and non-integer IDs are permitted.
        index (Optional[int]):
            Index of the sentence in the corpus. Default: ``None``.

    Examples:
        >>> lines = ['# text = But I found the location wonderful and the neighbors very kind.',
                     '1\tBut\t_\t_\t_\t_\t_\t_\t_\t_',
                     '2\tI\t_\t_\t_\t_\t_\t_\t_\t_',
                     '3\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '4\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_',
                     '6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7\tand\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '8\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_',
                     '10\tvery\t_\t_\t_\t_\t_\t_\t_\t_',
                     '11\tkind\t_\t_\t_\t_\t_\t_\t_\t_',
                     '12\t.\t_\t_\t_\t_\t_\t_\t_\t_']
        >>> sentence = CoNLLSentence(transform, lines)  # fields in transform are built from ptb.
        >>> sentence.arcs = [3, 3, 0, 5, 6, 3, 6, 9, 11, 11, 6, 3]
        >>> sentence.rels = ['cc', 'nsubj', 'root', 'det', 'nsubj', 'xcomp',
                             'cc', 'det', 'dep', 'advmod', 'conj', 'punct']
        >>> sentence
        # text = But I found the location wonderful and the neighbors very kind.
        1       But     _       _       _       _       3       cc      _       _
        2       I       _       _       _       _       3       nsubj   _       _
        3       found   _       _       _       _       0       root    _       _
        4       the     _       _       _       _       5       det     _       _
        5       location        _       _       _       _       6       nsubj   _       _
        6       wonderful       _       _       _       _       3       xcomp   _       _
        7       and     _       _       _       _       6       cc      _       _
        7.1     found   _       _       _       _       _       _       _       _
        8       the     _       _       _       _       9       det     _       _
        9       neighbors       _       _       _       _       11      dep     _       _
        10      very    _       _       _       _       11      advmod  _       _
        11      kind    _       _       _       _       6       conj    _       _
        12      .       _       _       _       _       3       punct   _       _
    """

    def __init__(self, transform: CoNLL, lines: List[str], index: Optional[int] = None) -> CoNLLSentence:
        super().__init__(transform, index)

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


class TreeSentence(Sentence):
    r"""
    Args:
        transform (Tree):
            A :class:`Tree` object.
        tree (nltk.tree.Tree):
            A :class:`nltk.tree.Tree` object.
        index (Optional[int]):
            Index of the sentence in the corpus. Default: ``None``.
    """

    def __init__(self, transform: Tree, tree: nltk.Tree, index: Optional[int] = None) -> TreeSentence:
        super().__init__(transform, index)

        words, tags, chart = *zip(*tree.pos()), None
        if transform.training:
            chart = [[None]*(len(words)+1) for _ in range(len(words)+1)]
            for i, j, label in Tree.factorize(Tree.binarize(tree)[0]):
                chart[i][j] = label
        self.values = [words, tags, tree, chart]

    def __repr__(self):
        return self.values[-2].pformat(1000000)

    def pretty_print(self):
        self.values[-2].pretty_print()
