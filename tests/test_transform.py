# -*- coding: utf-8 -*-

import itertools

import nltk

from supar.models.const.crf.transform import Tree
from supar.models.dep.biaffine.transform import CoNLL


class TestCoNLL:

    def istree_naive(self, sequence, proj=False, multiroot=True):
        if proj and not CoNLL.isprojective(sequence):
            return False
        roots = [i for i, head in enumerate(sequence, 1) if head == 0]
        if len(roots) == 0:
            return False
        if len(roots) > 1 and not multiroot:
            return False
        sequence = [-1] + sequence

        def track(sequence, visited, i):
            if visited[i]:
                return False
            visited[i] = True
            for j, head in enumerate(sequence[1:], 1):
                if head == i:
                    track(sequence, visited, j)
            return True
        visited = [False]*len(sequence)
        for root in roots:
            if not track(sequence, visited, root):
                return False
            if any([not i for i in visited[1:]]):
                return False
        return True

    def test_isprojective(self):
        assert CoNLL.isprojective([2, 4, 2, 0, 5])
        assert CoNLL.isprojective([3, -1, 0, -1, 3])
        assert not CoNLL.isprojective([2, 4, 0, 3, 4])
        assert not CoNLL.isprojective([4, -1, 0, -1, 4])
        assert not CoNLL.isprojective([2, -1, -1, 1, 0])
        assert not CoNLL.isprojective([0, 5, -1, -1, 4])

    def test_istree(self):
        permutations = [list(sequence[:5]) for sequence in itertools.permutations(range(6))]
        for sequence in permutations:
            assert CoNLL.istree(sequence, False, False) == self.istree_naive(sequence, False, False), f"{sequence}"
            assert CoNLL.istree(sequence, False, True) == self.istree_naive(sequence, False, True), f"{sequence}"
            assert CoNLL.istree(sequence, True, False) == self.istree_naive(sequence, True, False), f"{sequence}"
            assert CoNLL.istree(sequence, True, True) == self.istree_naive(sequence, True, True), f"{sequence}"


class TestTree:

    def test_tree(self):
        tree = nltk.Tree.fromstring("""
                                    (TOP
                                      (S
                                        (NP (DT This) (NN time))
                                        (, ,)
                                        (NP (DT the) (NNS firms))
                                        (VP (VBD were) (ADJP (JJ ready)))
                                        (. .)))
                                    """)
        assert tree == Tree.build(tree, Tree.factorize(Tree.binarize(tree)[0]))
