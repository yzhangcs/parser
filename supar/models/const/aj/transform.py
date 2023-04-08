# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

import nltk
import torch

from supar.models.const.crf.transform import Tree
from supar.utils.common import NUL
from supar.utils.logging import get_logger
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import Sentence

if TYPE_CHECKING:
    from supar.utils import Field

logger = get_logger(__name__)


class AttachJuxtaposeTree(Tree):
    r"""
    :class:`AttachJuxtaposeTree` is derived from the :class:`Tree` class,
    supporting back-and-forth transformations between trees and AttachJuxtapose actions :cite:`yang-deng-2020-aj`.

    Attributes:
        WORD:
            Words in the sentence.
        POS:
            Part-of-speech tags, or underscores if not available.
        TREE:
            The raw constituency tree in :class:`nltk.tree.Tree` format.
        NODE:
            The target node on each rightmost chain.
        PARENT:
            The label of the parent node of each terminal.
        NEW:
            The label of each newly inserted non-terminal with a target node and a terminal as juxtaposed children.
            ``NUL`` represents the `Attach` action.
    """

    fields = ['WORD', 'POS', 'TREE', 'NODE', 'PARENT', 'NEW']

    def __init__(
        self,
        WORD: Optional[Union[Field, Iterable[Field]]] = None,
        POS: Optional[Union[Field, Iterable[Field]]] = None,
        TREE: Optional[Union[Field, Iterable[Field]]] = None,
        NODE: Optional[Union[Field, Iterable[Field]]] = None,
        PARENT: Optional[Union[Field, Iterable[Field]]] = None,
        NEW: Optional[Union[Field, Iterable[Field]]] = None
    ) -> Tree:
        super().__init__()

        self.WORD = WORD
        self.POS = POS
        self.TREE = TREE
        self.NODE = NODE
        self.PARENT = PARENT
        self.NEW = NEW

    @property
    def tgt(self):
        return self.NODE, self.PARENT, self.NEW

    @classmethod
    def tree2action(cls, tree: nltk.Tree):
        r"""
        Converts a constituency tree into AttachJuxtapose actions.

        Args:
            tree (nltk.tree.Tree):
                A constituency tree in :class:`nltk.tree.Tree` format.

        Returns:
            A sequence of AttachJuxtapose actions.

        Examples:
            >>> from supar.models.const.aj.transform import AttachJuxtaposeTree
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ Arthur))
                                                (VP
                                                  (_ is)
                                                  (NP (NP (_ King)) (PP (_ of) (NP (_ the) (_ Britons)))))
                                                (_ .)))
                                            ''')
            >>> tree.pretty_print()
                            TOP
                             |
                             S
               ______________|_______________________
              |              VP                      |
              |      ________|___                    |
              |     |            NP                  |
              |     |    ________|___                |
              |     |   |            PP              |
              |     |   |     _______|___            |
              NP    |   NP   |           NP          |
              |     |   |    |        ___|_____      |
              _     _   _    _       _         _     _
              |     |   |    |       |         |     |
            Arthur  is King  of     the     Britons  .
            >>> AttachJuxtaposeTree.tree2action(tree)
            [(0, 'NP', '<nul>'), (0, 'VP', 'S'), (1, 'NP', '<nul>'),
             (2, 'PP', 'NP'), (3, 'NP', '<nul>'), (4, '<nul>', '<nul>'),
             (0, '<nul>', '<nul>')]
        """

        def isroot(node):
            return node == tree[0]

        def isterminal(node):
            return len(node) == 1 and not isinstance(node[0], nltk.Tree)

        def last_leaf(node):
            pos = ()
            while True:
                pos += (len(node) - 1,)
                node = node[-1]
                if isterminal(node):
                    return node, pos

        def parent(position):
            return tree[position[:-1]]

        def grand(position):
            return tree[position[:-2]]

        def detach(tree):
            last, last_pos = last_leaf(tree)
            siblings = parent(last_pos)[:-1]

            if len(siblings) > 0:
                last_subtree = last
                last_subtree_siblings = siblings
                parent_label = NUL
            else:
                last_subtree, last_pos = parent(last_pos), last_pos[:-1]
                last_subtree_siblings = [] if isroot(last_subtree) else parent(last_pos)[:-1]
                parent_label = last_subtree.label()

            target_pos, new_label, last_tree = 0, NUL, tree
            if isroot(last_subtree):
                last_tree = None
            elif len(last_subtree_siblings) == 1 and not isterminal(last_subtree_siblings[0]):
                new_label = parent(last_pos).label()
                target = last_subtree_siblings[0]
                last_grand = grand(last_pos)
                if last_grand is None:
                    last_tree = target
                else:
                    last_grand[-1] = target
                target_pos = len(last_pos) - 2
            else:
                target = parent(last_pos)
                target.pop()
                target_pos = len(last_pos) - 2
            action = target_pos, parent_label, new_label
            return action, last_tree
        if tree is None:
            return []
        action, last_tree = detach(tree)
        return cls.tree2action(last_tree) + [action]

    @classmethod
    def action2tree(
        cls,
        tree: nltk.Tree,
        actions: List[Tuple[int, str, str]],
        join: str = '::',
    ) -> nltk.Tree:
        r"""
        Recovers a constituency tree from a sequence of AttachJuxtapose actions.

        Args:
            tree (nltk.tree.Tree):
                An empty tree that provides a base for building a result tree.
            actions (List[Tuple[int, str, str]]):
                A sequence of AttachJuxtapose actions.
            join (str):
                A string used to connect collapsed node labels. Non-terminals containing this will be expanded to unary chains.
                Default: ``'::'``.

        Returns:
            A result constituency tree.

        Examples:
            >>> from supar.models.const.aj.transform import AttachJuxtaposeTree
            >>> tree = AttachJuxtaposeTree.totree(['Arthur', 'is', 'King', 'of', 'the', 'Britons', '.'], 'TOP')
            >>> AttachJuxtaposeTree.action2tree(tree,
                                                [(0, 'NP', '<nul>'), (0, 'VP', 'S'), (1, 'NP', '<nul>'),
                                                 (2, 'PP', 'NP'), (3, 'NP', '<nul>'), (4, '<nul>', '<nul>'),
                                                 (0, '<nul>', '<nul>')]).pretty_print()
                            TOP
                             |
                             S
               ______________|_______________________
              |              VP                      |
              |      ________|___                    |
              |     |            NP                  |
              |     |    ________|___                |
              |     |   |            PP              |
              |     |   |     _______|___            |
              NP    |   NP   |           NP          |
              |     |   |    |        ___|_____      |
              _     _   _    _       _         _     _
              |     |   |    |       |         |     |
            Arthur  is King  of     the     Britons  .
        """

        def target(node, depth):
            node_pos = ()
            for _ in range(depth):
                node_pos += (len(node) - 1,)
                node = node[-1]
            return node, node_pos

        def parent(tree, position):
            return tree[position[:-1]]

        def execute(tree: nltk.Tree, terminal: Tuple(str, str), action: Tuple[int, str, str]) -> nltk.Tree:
            new_leaf = nltk.Tree(terminal[1], [terminal[0]])
            target_pos, parent_label, new_label = action
            # create the subtree to be inserted
            new_subtree = new_leaf if parent_label == NUL else nltk.Tree(parent_label, [new_leaf])
            # find the target position at which to insert the new subtree
            target_node = tree
            if target_node is not None:
                target_node, target_pos = target(target_node, target_pos)

            # Attach
            if new_label == NUL:
                # attach the first token
                if target_node is None:
                    return new_subtree
                target_node.append(new_subtree)
            # Juxtapose
            else:
                new_subtree = nltk.Tree(new_label, [target_node, new_subtree])
                if len(target_pos) > 0:
                    parent_node = parent(tree, target_pos)
                    parent_node[-1] = new_subtree
                else:
                    tree = new_subtree
            return tree

        tree, root, terminals = None, tree.label(), tree.pos()
        for terminal, action in zip(terminals, actions):
            tree = execute(tree, terminal, action)
        # recover unary chains
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, nltk.Tree):
                nodes.extend(node)
                if join in node.label():
                    labels = node.label().split(join)
                    node.set_label(labels[0])
                    subtree = nltk.Tree(labels[-1], node)
                    for label in reversed(labels[1:-1]):
                        subtree = nltk.Tree(label, [subtree])
                    node[:] = [subtree]
        return nltk.Tree(root, [tree])

    @classmethod
    def action2span(
        cls,
        action: torch.Tensor,
        spans: torch.Tensor = None,
        nul_index: int = -1,
        mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        r"""
        Converts a batch of the tensorized action at a given step into spans.

        Args:
            action (~torch.Tensor): ``[3, batch_size]``.
                A batch of the tensorized action at a given step, containing indices of target nodes, parent and new labels.
            spans (~torch.Tensor):
                Spans generated at previous steps, ``None`` at the first step. Default: ``None``.
            nul_index (int):
                The index for the obj:`NUL` token, representing the Attach action. Default: -1.
            mask (~torch.BoolTensor): ``[batch_size]``.
                The mask for covering the unpadded tokens.

        Returns:
            A tensor representing a batch of spans for the given step.

        Examples:
            >>> from collections import Counter
            >>> from supar.models.const.aj.transform import AttachJuxtaposeTree, Vocab
            >>> from supar.utils.common import NUL
            >>> nodes, parents, news = zip(*[(0, 'NP', NUL), (0, 'VP', 'S'), (1, 'NP', NUL),
                                             (2, 'PP', 'NP'), (3, 'NP', NUL), (4, NUL, NUL),
                                             (0, NUL, NUL)])
            >>> vocab = Vocab(Counter(sorted(set([*parents, *news]))))
            >>> actions = torch.tensor([nodes, vocab[parents], vocab[news]]).unsqueeze(1)
            >>> spans = None
            >>> for action in actions.unbind(-1):
            ...     spans = AttachJuxtaposeTree.action2span(action, spans, vocab[NUL])
            ...
            >>> spans
            tensor([[[-1,  1, -1, -1, -1, -1, -1,  3],
                     [-1, -1, -1, -1, -1, -1,  4, -1],
                     [-1, -1, -1,  1, -1, -1,  1, -1],
                     [-1, -1, -1, -1, -1, -1,  2, -1],
                     [-1, -1, -1, -1, -1, -1,  1, -1],
                     [-1, -1, -1, -1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1, -1, -1, -1]]])
            >>> sequence = torch.where(spans.ge(0))
            >>> sequence = list(zip(sequence[1].tolist(), sequence[2].tolist(), vocab[spans[sequence]]))
            >>> sequence
            [(0, 1, 'NP'), (0, 7, 'S'), (1, 6, 'VP'), (2, 3, 'NP'), (2, 6, 'NP'), (3, 6, 'PP'), (4, 6, 'NP')]
            >>> tree = AttachJuxtaposeTree.totree(['Arthur', 'is', 'King', 'of', 'the', 'Britons', '.'], 'TOP')
            >>> AttachJuxtaposeTree.build(tree, sequence).pretty_print()
                            TOP
                             |
                             S
               ______________|_______________________
              |              VP                      |
              |      ________|___                    |
              |     |            NP                  |
              |     |    ________|___                |
              |     |   |            PP              |
              |     |   |     _______|___            |
              NP    |   NP   |           NP          |
              |     |   |    |        ___|_____      |
              _     _   _    _       _         _     _
              |     |   |    |       |         |     |
            Arthur  is King  of     the     Britons  .

        """

        # [batch_size]
        target, parent, new = action
        if spans is None:
            spans = action.new_full((action.shape[1], 2, 2), -1)
            spans[:, 0, 1] = parent
            return spans
        if mask is None:
            mask = torch.ones_like(target, dtype=bool)
        juxtapose_mask = new.ne(nul_index) & mask
        # ancestor nodes are those on the rightmost chain and higher than the target node
        # [batch_size, seq_len]
        rightmost_mask = spans[..., -1].ge(0)
        ancestors = rightmost_mask.cumsum(-1).masked_fill_(~rightmost_mask, -1) - 1
        # should not include the target node for the Juxtapose action
        ancestor_mask = mask.unsqueeze(-1) & ancestors.ge(0) & ancestors.le((target - juxtapose_mask.long()).unsqueeze(-1))
        target_pos = torch.where(ancestors.eq(target.unsqueeze(-1))[juxtapose_mask])[-1]
        # the right boundaries of ancestor nodes should be aligned with the new generated terminals
        spans = torch.cat((spans, torch.where(ancestor_mask, spans[..., -1], -1).unsqueeze(-1)), -1)
        spans[..., -2].masked_fill_(ancestor_mask, -1)
        spans[juxtapose_mask, target_pos, -1] = new.masked_fill(new.eq(nul_index), -1)[juxtapose_mask]
        spans[mask, -1, -1] = parent.masked_fill(parent.eq(nul_index), -1)[mask]
        # [batch_size, seq_len+1, seq_len+1]
        spans = torch.cat((spans, torch.full_like(spans[:, :1], -1)), 1)
        return spans

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        **kwargs
    ) -> List[AttachJuxtaposeTreeSentence]:
        r"""
        Args:
            data (Union[str, Iterable]):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.

        Returns:
            A list of :class:`AttachJuxtaposeTreeSentence` instances.
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
                sentence = AttachJuxtaposeTreeSentence(self, tree, index)
            except ValueError:
                logger.warning(f"Error found while converting Sentence {index} to a tree:\n{s}\nDiscarding it!")
                continue
            else:
                yield sentence
                index += 1
        self.root = tree.label()


class AttachJuxtaposeTreeSentence(Sentence):
    r"""
    Args:
        transform (AttachJuxtaposeTree):
            A :class:`AttachJuxtaposeTree` object.
        tree (nltk.tree.Tree):
            A :class:`nltk.tree.Tree` object.
        index (Optional[int]):
            Index of the sentence in the corpus. Default: ``None``.
    """

    def __init__(
        self,
        transform: AttachJuxtaposeTree,
        tree: nltk.Tree,
        index: Optional[int] = None
    ) -> AttachJuxtaposeTreeSentence:
        super().__init__(transform, index)

        words, tags = zip(*tree.pos())
        nodes, parents, news = None, None, None
        if transform.training:
            oracle_tree = tree.copy(True)
            # the root node must have a unary chain
            if len(oracle_tree) > 1:
                oracle_tree[:] = [nltk.Tree('*', oracle_tree)]
            oracle_tree.collapse_unary(joinChar='::')
            if len(oracle_tree) == 1 and not isinstance(oracle_tree[0][0], nltk.Tree):
                oracle_tree[0] = nltk.Tree('*', [oracle_tree[0]])
            nodes, parents, news = zip(*transform.tree2action(oracle_tree))
        self.values = [words, tags, tree, nodes, parents, news]

    def __repr__(self):
        return self.values[-4].pformat(1000000)

    def pretty_print(self):
        self.values[-4].pretty_print()
