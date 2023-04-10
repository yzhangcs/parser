# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union, Sequence

import nltk

from supar.models.const.crf.transform import Tree
from supar.utils.logging import get_logger
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import Sentence

if TYPE_CHECKING:
    from supar.utils import Field

logger = get_logger(__name__)


class TetraTaggingTree(Tree):
    r"""
    :class:`TetraTaggingTree` is derived from the :class:`Tree` class and is defined for supporting the transition system of
    tetra tagger :cite:`kitaev-klein-2020-tetra`.

    Attributes:
        WORD:
            Words in the sentence.
        POS:
            Part-of-speech tags, or underscores if not available.
        TREE:
            The raw constituency tree in :class:`nltk.tree.Tree` format.
        LEAF:
            Action labels in tetra tagger transition system.
        NODE:
            Non-terminal labels.
    """

    fields = ['WORD', 'POS', 'TREE', 'LEAF', 'NODE']

    def __init__(
        self,
        WORD: Optional[Union[Field, Iterable[Field]]] = None,
        POS: Optional[Union[Field, Iterable[Field]]] = None,
        TREE: Optional[Union[Field, Iterable[Field]]] = None,
        LEAF: Optional[Union[Field, Iterable[Field]]] = None,
        NODE: Optional[Union[Field, Iterable[Field]]] = None
    ) -> Tree:
        super().__init__()

        self.WORD = WORD
        self.POS = POS
        self.TREE = TREE
        self.LEAF = LEAF
        self.NODE = NODE

    @property
    def tgt(self):
        return self.LEAF, self.NODE

    @classmethod
    def tree2action(cls, tree: nltk.Tree) -> Tuple[Sequence, Sequence]:
        r"""
        Converts a (binarized) constituency tree into tetra-tagging actions.

        Args:
            tree (nltk.tree.Tree):
                A constituency tree in :class:`nltk.tree.Tree` format.

        Returns:
            Tetra-tagging actions for leaves and non-terminals.

        Examples:
            >>> from supar.models.const.tt.transform import TetraTaggingTree
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

            >>> tree = TetraTaggingTree.binarize(tree, left=False, implicit=True)
            >>> tree.pretty_print()
                         TOP
                          |
                          S
              ____________|______
             |
             |             ______|___________
             |            VP                 |
             |     _______|______            |
             |    |            S::VP         |
             |    |        ______|_____      |
             NP                        NP
             |    |       |            |     |
             _    _       _            _     _
             |    |       |            |     |
            She enjoys playing       tennis  .

            >>> TetraTaggingTree.tree2action(tree)
            (['l/NP', 'l/', 'l/', 'r/NP', 'r/'], ['L/S', 'L/VP', 'R/S::VP', 'R/'])
        """

        def traverse(tree: nltk.Tree, left: bool = True) -> List:
            if len(tree) == 1 and not isinstance(tree[0], nltk.Tree):
                return ['l' if left else 'r'], []
            if len(tree) == 1 and not isinstance(tree[0][0], nltk.Tree):
                return [f"{'l' if left else 'r'}/{tree.label()}"], []
            return tuple(sum(i, []) for i in zip(*[traverse(tree[0]),
                                                   ([], [f'{("L" if left else "R")}/{tree.label()}']),
                                                   traverse(tree[1], False)]))
        return traverse(tree[0])

    @classmethod
    def action2tree(
        cls,
        tree: nltk.Tree,
        actions: Tuple[Sequence, Sequence],
        mark: Union[str, Tuple[str]] = ('*', '|<>'),
        join: str = '::',
    ) -> nltk.Tree:
        r"""
        Recovers a constituency tree from tetra-tagging actions.

        Args:
            tree (nltk.tree.Tree):
                An empty tree that provides a base for building a result tree.
            actions (Tuple[Sequence, Sequence]):
                Tetra-tagging actions.
            mark (Union[str, List[str]]):
                A string used to mark newly inserted nodes. Non-terminals containing this will be removed.
                Default: ``('*', '|<>')``.
            join (str):
                A string used to connect collapsed node labels. Non-terminals containing this will be expanded to unary chains.
                Default: ``'::'``.

        Returns:
            A result constituency tree.

        Examples:
            >>> from supar.models.const.tt.transform import TetraTaggingTree
            >>> tree = TetraTaggingTree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], 'TOP')
            >>> actions = (['l/NP', 'l/', 'l/', 'r/NP', 'r/'], ['L/S', 'L/VP', 'R/S::VP', 'R/'])
            >>> TetraTaggingTree.action2tree(tree, actions).pretty_print()
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

        stack = []
        leaves = [nltk.Tree(pos, [token]) for token, pos in tree.pos()]
        for i, (al, an) in enumerate(zip(*actions)):
            leaf = nltk.Tree(al.split('/', 1)[1], [leaves[i]])
            if al.startswith('l'):
                stack.append([leaf, None])
            else:
                slot = stack[-1][1]
                slot.append(leaf)
            if an.startswith('L'):
                node = nltk.Tree(an.split('/', 1)[1], [stack[-1][0]])
                stack[-1][0] = node
            else:
                node = nltk.Tree(an.split('/', 1)[1], [stack.pop()[0]])
                slot = stack[-1][1]
                slot.append(node)
            stack[-1][1] = node
        # the last leaf must be leftward
        leaf = nltk.Tree(actions[0][-1].split('/', 1)[1], [leaves[-1]])
        if len(stack) > 0:
            stack[-1][1].append(leaf)
        else:
            stack.append([leaf, None])

        def debinarize(tree):
            if len(tree) == 1 and not isinstance(tree[0], nltk.Tree):
                return [tree]
            label, children = tree.label(), []
            for child in tree:
                children.extend(debinarize(child))
            if not label or label.endswith(mark):
                return children
            labels = label.split(join) if join in label else [label]
            tree = nltk.Tree(labels[-1], children)
            for label in reversed(labels[:-1]):
                tree = nltk.Tree(label, [tree])
            return [tree]
        return debinarize(nltk.Tree(tree.label(), [stack[0][0]]))[0]

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        **kwargs
    ) -> List[TetraTaggingTreeSentence]:
        r"""
        Args:
            data (Union[str, Iterable]):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.

        Returns:
            A list of :class:`TetraTaggingTreeSentence` instances.
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
                sentence = TetraTaggingTreeSentence(self, tree, index)
            except ValueError:
                logger.warning(f"Error found while converting Sentence {index} to a tree:\n{s}\nDiscarding it!")
                continue
            else:
                yield sentence
                index += 1
        self.root = tree.label()


class TetraTaggingTreeSentence(Sentence):
    r"""
    Args:
        transform (TetraTaggingTree):
            A :class:`TetraTaggingTree` object.
        tree (nltk.tree.Tree):
            A :class:`nltk.tree.Tree` object.
        index (Optional[int]):
            Index of the sentence in the corpus. Default: ``None``.
    """

    def __init__(
        self,
        transform: TetraTaggingTree,
        tree: nltk.Tree,
        index: Optional[int] = None
    ) -> TetraTaggingTreeSentence:
        super().__init__(transform, index)

        words, tags = zip(*tree.pos())
        leaves, nodes = None, None
        if transform.training:
            oracle_tree = tree.copy(True)
            # the root node must have a unary chain
            if len(oracle_tree) > 1:
                oracle_tree[:] = [nltk.Tree('*', oracle_tree)]
            oracle_tree = TetraTaggingTree.binarize(oracle_tree, left=False, implicit=True)
            if len(oracle_tree) == 1 and not isinstance(oracle_tree[0][0], nltk.Tree):
                oracle_tree[0] = nltk.Tree('*', [oracle_tree[0]])
            leaves, nodes = transform.tree2action(oracle_tree)
        self.values = [words, tags, tree, leaves, nodes]

    def __repr__(self):
        return self.values[-3].pformat(1000000)

    def pretty_print(self):
        self.values[-3].pretty_print()
