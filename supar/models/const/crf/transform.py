# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import (TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple,
                    Union)

import nltk

from supar.utils.logging import get_logger
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import Sentence, Transform

if TYPE_CHECKING:
    from supar.utils import Field

logger = get_logger(__name__)


class Tree(Transform):
    r"""
    A :class:`Tree` object factorize a constituency tree into four fields,
    each associated with one or more :class:`~supar.utils.field.Field` objects.

    Attributes:
        WORD:
            Words in the sentence.
        POS:
            Part-of-speech tags, or underscores if not available.
        TREE:
            The raw constituency tree in :class:`nltk.tree.Tree` format.
        CHART:
            The factorized sequence of binarized tree traversed in post-order.
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
        normalize: Dict[str, str] = {'(': '-LRB-', ')': '-RRB-'}
    ) -> nltk.Tree:
        r"""
        Converts a list of tokens to a :class:`nltk.tree.Tree`, with missing fields filled in with underscores.

        Args:
            tokens (List[Union[str, Tuple]]):
                This can be either a list of words or word/pos pairs.
            root (str):
                The root label of the tree. Default: ''.
            normalize (Dict):
                Keys within the dict in each token will be replaced by the values. Default: ``{'(': '-LRB-', ')': '-RRB-'}``.

        Returns:
            A :class:`nltk.tree.Tree` object.

        Examples:
            >>> from supar.models.const.crf.transform import Tree
            >>> Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], 'TOP').pprint()
            (TOP ( (_ She)) ( (_ enjoys)) ( (_ playing)) ( (_ tennis)) ( (_ .)))
            >>> Tree.totree(['(', 'If', 'You', 'Let', 'It', ')'], 'TOP').pprint()
            (TOP
              ( (_ -LRB-))
              ( (_ If))
              ( (_ You))
              ( (_ Let))
              ( (_ It))
              ( (_ -RRB-)))
        """

        normalize = str.maketrans(normalize)
        if isinstance(tokens[0], str):
            tokens = [(token, '_') for token in tokens]
        return nltk.Tree(root, [nltk.Tree('', [nltk.Tree(pos, [word.translate(normalize)])]) for word, pos in tokens])

    @classmethod
    def binarize(
        cls,
        tree: nltk.Tree,
        left: bool = True,
        mark: str = '*',
        join: str = '::',
        implicit: bool = False
    ) -> nltk.Tree:
        r"""
        Conducts binarization over the tree.

        First, the tree is transformed to satisfy `Chomsky Normal Form (CNF)`_.
        Here we call :meth:`~nltk.tree.Tree.chomsky_normal_form` to conduct left-binarization.
        Second, all unary productions in the tree are collapsed.

        Args:
            tree (nltk.tree.Tree):
                The tree to be binarized.
            left (bool):
                If ``True``, left-binarization is conducted. Default: ``True``.
            mark (str):
                A string used to mark newly inserted nodes, working if performing explicit binarization. Default: ``'*'``.
            join (str):
                A string used to connect collapsed node labels. Default: ``'::'``.
            implicit (bool):
                If ``True``, performs implicit binarization. Default: ``False``.

        Returns:
            The binarized tree.

        Examples:
            >>> from supar.models.const.crf.transform import Tree
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ She))
                                                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                                (_ .)))
                                            ''')
            >>> tree.pretty_print()
                         TOP
                          |
                          S
              ____________|________________
             |            VP               |
             |     _______|_____           |
             |    |             S          |
             |    |             |          |
             |    |             VP         |
             |    |        _____|____      |
             NP   |       |          NP    |
             |    |       |          |     |
             _    _       _          _     _
             |    |       |          |     |
            She enjoys playing     tennis  .

            >>> Tree.binarize(tree).pretty_print()
                             TOP
                              |
                              S
                         _____|__________________
                        S*                       |
              __________|_____                   |
             |                VP                 |
             |     ___________|______            |
             |    |                S::VP         |
             |    |            ______|_____      |
             NP  VP*         VP*           NP    S*
             |    |           |            |     |
             _    _           _            _     _
             |    |           |            |     |
            She enjoys     playing       tennis  .

            >>> Tree.binarize(tree, implicit=True).pretty_print()
                             TOP
                              |
                              S
                         _____|__________________
                                                 |
              __________|_____                   |
             |                VP                 |
             |     ___________|______            |
             |    |                S::VP         |
             |    |            ______|_____      |
             NP                            NP
             |    |           |            |     |
             _    _           _            _     _
             |    |           |            |     |
            She enjoys     playing       tennis  .

            >>> Tree.binarize(tree, left=False).pretty_print()
                         TOP
                          |
                          S
              ____________|______
             |                   S*
             |             ______|___________
             |            VP                 |
             |     _______|______            |
             |    |            S::VP         |
             |    |        ______|_____      |
             NP  VP*     VP*           NP    S*
             |    |       |            |     |
             _    _       _            _     _
             |    |       |            |     |
            She enjoys playing       tennis  .

        .. _Chomsky Normal Form (CNF):
            https://en.wikipedia.org/wiki/Chomsky_normal_form
        """

        tree = tree.copy(True)
        nodes = [tree]
        if len(tree) == 1:
            if not isinstance(tree[0][0], nltk.Tree):
                tree[0] = nltk.Tree(f'{tree.label()}{mark}', [tree[0]])
            nodes = [tree[0]]
        while nodes:
            node = nodes.pop()
            if isinstance(node, nltk.Tree):
                if implicit:
                    label = ''
                else:
                    label = node.label()
                    if mark not in label:
                        label = f'{label}{mark}'
                # ensure that only non-terminals can be attached to a n-ary subtree
                if len(node) > 1:
                    for child in node:
                        if not isinstance(child[0], nltk.Tree):
                            child[:] = [nltk.Tree(child.label(), child[:])]
                            child.set_label(label)
                # chomsky normal form factorization
                if len(node) > 2:
                    if left:
                        node[:-1] = [nltk.Tree(label, node[:-1])]
                    else:
                        node[1:] = [nltk.Tree(label, node[1:])]
                nodes.extend(node)
        # collapse unary productions, shoule be conducted after binarization
        tree.collapse_unary(joinChar=join)
        return tree

    @classmethod
    def factorize(
        cls,
        tree: nltk.Tree,
        delete_labels: Optional[Set[str]] = None,
        equal_labels: Optional[Dict[str, str]] = None
    ) -> Iterable[Tuple]:
        r"""
        Factorizes the tree into a sequence traversed in post-order.

        Args:
            tree (nltk.tree.Tree):
                The tree to be factorized.
            delete_labels (Optional[Set[str]]):
                A set of labels to be ignored. This is used for evaluation.
                If it is a pre-terminal label, delete the word along with the brackets.
                If it is a non-terminal label, just delete the brackets (don't delete children).
                In `EVALB`_, the default set is:
                {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
                Default: ``None``.
            equal_labels (Optional[Dict[str, str]]):
                The key-val pairs in the dict are considered equivalent (non-directional). This is used for evaluation.
                The default dict defined in `EVALB`_ is: {'ADVP': 'PRT'}
                Default: ``None``.

        Returns:
            The sequence of the factorized tree.

        Examples:
            >>> from supar.models.const.crf.transform import Tree
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ She))
                                                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                                (_ .)))
                                            ''')
            >>> Tree.factorize(tree)
            [(0, 1, 'NP'), (3, 4, 'NP'), (2, 4, 'VP'), (2, 4, 'S'), (1, 4, 'VP'), (0, 5, 'S'), (0, 5, 'TOP')]
            >>> Tree.factorize(tree, delete_labels={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''})
            [(0, 1, 'NP'), (3, 4, 'NP'), (2, 4, 'VP'), (2, 4, 'S'), (1, 4, 'VP'), (0, 5, 'S')]

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
                return (i + 1 if label is not None else i), []
            j, spans = i, []
            for child in tree:
                j, s = track(child, j)
                spans += s
            if label is not None and j > i:
                spans = spans + [(i, j, label)]
            return j, spans
        return track(tree, 0)[1]

    @classmethod
    def build(
        cls,
        sentence: Union[nltk.Tree, Iterable],
        spans: Iterable[Tuple],
        delete_labels: Optional[Set[str]] = None,
        mark: Union[str, Tuple[str]] = ('*', '|<>'),
        root: str = '',
        join: str = '::',
        postorder: bool = True
    ) -> nltk.Tree:
        r"""
        Builds a constituency tree from a span sequence.
        During building, the sequence is recovered, i.e., de-binarized to the original format.

        Args:
            sentence (Union[nltk.tree.Tree, Iterable]):
                Sentence to provide a base for building a result tree, both `nltk.tree.Tree` and tokens are allowed.
            spans (Iterable[Tuple]):
                A list of spans, each consisting of the indices of left/right boundaries and label of the constituent.
            delete_labels (Optional[Set[str]]):
                A set of labels to be ignored. Default: ``None``.
            mark (Union[str, List[str]]):
                A string used to mark newly inserted nodes. Non-terminals containing this will be removed.
                Default: ``('*', '|<>')``.
            root (str):
                The root label of the tree, needed if input a list of tokens. Default: ''.
            join (str):
                A string used to connect collapsed node labels. Non-terminals containing this will be expanded to unary chains.
                Default: ``'::'``.
            postorder (bool):
                If ``True``, enforces the sequence is sorted in post-order. Default: ``True``.

        Returns:
            A result constituency tree.

        Examples:
            >>> from supar.models.const.crf.transform import Tree
            >>> Tree.build(['She', 'enjoys', 'playing', 'tennis', '.'],
                           [(0, 5, 'S'), (0, 4, 'S*'), (0, 1, 'NP'), (1, 4, 'VP'), (1, 2, 'VP*'),
                            (2, 4, 'S::VP'), (2, 3, 'VP*'), (3, 4, 'NP'), (4, 5, 'S*')],
                           root='TOP').pretty_print()
                         TOP
                          |
                          S
              ____________|________________
             |            VP               |
             |     _______|_____           |
             |    |             S          |
             |    |             |          |
             |    |             VP         |
             |    |        _____|____      |
             NP   |       |          NP    |
             |    |       |          |     |
             _    _       _          _     _
             |    |       |          |     |
            She enjoys playing     tennis  .

            >>> Tree.build(['She', 'enjoys', 'playing', 'tennis', '.'],
                           [(0, 1, 'NP'), (3, 4, 'NP'), (2, 4, 'VP'), (2, 4, 'S'), (1, 4, 'VP'), (0, 5, 'S')],
                           root='TOP').pretty_print()
                         TOP
                          |
                          S
              ____________|________________
             |            VP               |
             |     _______|_____           |
             |    |             S          |
             |    |             |          |
             |    |             VP         |
             |    |        _____|____      |
             NP   |       |          NP    |
             |    |       |          |     |
             _    _       _          _     _
             |    |       |          |     |
            She enjoys playing     tennis  .

        """

        tree = sentence if isinstance(sentence, nltk.Tree) else Tree.totree(sentence, root)
        leaves = [subtree for subtree in tree.subtrees() if not isinstance(subtree[0], nltk.Tree)]
        if postorder:
            spans = sorted(spans, key=lambda x: (x[1], x[1] - x[0]))

        root = tree.label()
        start, stack = 0, []
        for span in spans:
            i, j, label = span
            if delete_labels is not None and label in delete_labels:
                continue
            stack.extend([(n, n + 1, leaf) for n, leaf in enumerate(leaves[start:i], start)])
            children = []
            while len(stack) > 0 and i <= stack[-1][0]:
                children = [stack.pop()] + children
            start = children[-1][1] if len(children) > 0 else i
            children.extend([(n, n + 1, leaf) for n, leaf in enumerate(leaves[start:j], start)])
            start = j
            if not label or label.endswith(mark):
                stack.extend(children)
                continue
            labels = label.split(join)
            tree = nltk.Tree(labels[-1], [child[-1] for child in children])
            for label in reversed(labels[:-1]):
                tree = nltk.Tree(label, [tree])
            stack.append((i, j, tree))
        stack.extend([(n, n + 1, leaf) for n, leaf in enumerate(leaves[start:], start)])
        return nltk.Tree(root, [i[-1] for i in stack])

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        **kwargs
    ) -> List[TreeSentence]:
        r"""
        Args:
            data (Union[str, Iterable]):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.

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
                sentence = TreeSentence(self, tree, index, **kwargs)
            except ValueError:
                logger.warning(f"Error found while converting Sentence {index} to a tree:\n{s}\nDiscarding it!")
                continue
            else:
                yield sentence
                index += 1
        self.root = tree.label()


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

    def __init__(
        self,
        transform: Tree,
        tree: nltk.Tree,
        index: Optional[int] = None,
        **kwargs
    ) -> TreeSentence:
        super().__init__(transform, index)

        words, tags, chart = *zip(*tree.pos()), None
        if transform.training:
            chart = [[None] * (len(words) + 1) for _ in range(len(words) + 1)]
            for i, j, label in Tree.factorize(Tree.binarize(tree, implicit=kwargs.get('implicit', False))[0]):
                chart[i][j] = label
        self.values = [words, tags, tree, chart]

    def __repr__(self):
        return self.values[-2].pformat(1000000)

    def pretty_print(self):
        self.values[-2].pretty_print()
