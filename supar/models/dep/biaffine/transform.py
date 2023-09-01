# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import tempfile
from io import StringIO
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Tuple, Union

from supar.utils.logging import get_logger
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import Sentence, Transform

if TYPE_CHECKING:
    from supar.utils import Field

logger = get_logger(__name__)


class CoNLL(Transform):
    r"""
    A :class:`CoNLL` object holds ten fields required for CoNLL-X data format :cite:`buchholz-marsi-2006-conll`.
    Each field can be bound to one or more :class:`~supar.utils.field.Field` objects.
    For example, ``FORM`` can contain both :class:`~supar.utils.field.Field` and :class:`~supar.utils.field.SubwordField`
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
            for j, hj in enumerate(heads[i + 1:], i + 1):
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
        edges = [[0] * (len(sequence) + 1) for _ in range(len(sequence) + 1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edges[i][int(pair.split(':')[0])] = 1
        return edges

    @classmethod
    def get_labels(cls, sequence):
        labels = [[None] * (len(sequence) + 1) for _ in range(len(sequence) + 1)]
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
    def toconll(cls, tokens: Sequence[Union[str, Tuple]]) -> str:
        r"""
        Converts a list of tokens to a string in CoNLL-X format with missing fields filled with underscores.

        Args:
            tokens (Sequence[Union[str, Tuple]]):
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
            s = '\n'.join([f"{i}\t{word}\t" + '\t'.join(['_'] * 8)
                           for i, word in enumerate(tokens, 1)])
        elif len(tokens[0]) == 2:
            s = '\n'.join([f"{i}\t{word}\t_\t{tag}\t" + '\t'.join(['_'] * 6)
                           for i, (word, tag) in enumerate(tokens, 1)])
        elif len(tokens[0]) == 3:
            s = '\n'.join([f"{i}\t{word}\t{lemma}\t{tag}\t" + '\t'.join(['_'] * 6)
                           for i, (word, lemma, tag) in enumerate(tokens, 1)])
        else:
            raise RuntimeError(f"Invalid sequence {tokens}. Only list of str or list of word/pos/lemma tuples are support.")
        return s + '\n'

    @classmethod
    def isprojective(cls, sequence: Sequence[int]) -> bool:
        r"""
        Checks if a dependency tree is projective.
        This also works for partial annotation.

        Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
        which are hard to detect in the scenario of partial annotation.

        Args:
            sequence (Sequence[int]):
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
            for hj, dj in pairs[i + 1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    return False
                if lj <= hi <= rj and hj == di:
                    return False
                if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                    return False
        return True

    @classmethod
    def istree(cls, sequence: Sequence[int], proj: bool = False, multiroot: bool = False) -> bool:
        r"""
        Checks if the arcs form an valid dependency tree.

        Args:
            sequence (Sequence[int]):
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

    @classmethod
    def projective_order(cls, sequence: Sequence[int]) -> Sequence:
        r"""
        Returns the projective order corresponding to the tree :cite:`nivre-2009-non`.

        Args:
            sequence (Sequence[int]):
                A list of head indices.

        Returns:
            The projective order of the tree.

        Examples:
            >>> CoNLL.projective_order([2, 0, 2, 3])
            [1, 2, 3, 4]
            >>> CoNLL.projective_order([3, 0, 0, 3])
            [2, 1, 3, 4]
            >>> CoNLL.projective_order([2, 3, 0, 3, 2, 7, 5, 4, 3])
            [1, 2, 5, 6, 7, 3, 4, 8, 9]
        """

        adjs = [[] for _ in range(len(sequence) + 1)]
        for dep, head in enumerate(sequence, 1):
            adjs[head].append(dep)

        def order(adjs, head):
            i = 0
            for dep in adjs[head]:
                if head < dep:
                    break
                i += 1
            left = [j for dep in adjs[head][:i] for j in order(adjs, dep)]
            right = [j for dep in adjs[head][i:] for j in order(adjs, dep)]
            return left + [head] + right
        return [i for head in adjs[0] for i in order(adjs, head)]

    @classmethod
    def projectivize(cls, file: str, fproj: str, malt: str) -> str:
        r"""
        Projectivizes the non-projective input trees to pseudo-projective ones with MaltParser.

        Args:
            file (str):
                Path to the input file containing non-projective trees that need to be handled.
            fproj (str):
                Path to the output file containing produced pseudo-projective trees.
            malt (str):
                Path to the MaltParser, which requires the Java execution environment.

        Returns:
            The name of the output file.
        """

        import hashlib
        import subprocess
        file, fproj, malt = os.path.abspath(file), os.path.abspath(fproj), os.path.abspath(malt)
        path, parser = os.path.dirname(malt), os.path.basename(malt)
        cfg = hashlib.sha256(file.encode('ascii')).hexdigest()[:8]
        subprocess.check_output([f"cd {path}; java -jar {parser} -c {cfg} -m proj -i {file} -o {fproj} -pp head"],
                                stderr=subprocess.STDOUT,
                                shell=True)
        return fproj

    @classmethod
    def deprojectivize(
        cls,
        sentences: Iterable[Sentence],
        arcs: Iterable,
        rels: Iterable,
        data: str,
        malt: str
    ) -> Tuple[Iterable, Iterable]:
        r"""
        Recover the projectivized sentences to the orginal format with MaltParser.

        Args:
            sentences (Iterable[Sentence]):
                Sentences in CoNLL-like format.
            arcs (Iterable):
                Sequences of arcs for pseudo projective trees.
            rels (Iterable):
                Sequences of dependency relations for pseudo projective trees.
            data (str):
                The data file used for projectivization, typically the training file.
            malt (str):
                Path to the MaltParser, which requires the Java execution environment.

        Returns:
            Recovered arcs and dependency relations.
        """

        import hashlib
        import subprocess
        data, malt = os.path.abspath(data), os.path.abspath(malt)
        path, parser = os.path.dirname(malt), os.path.basename(malt)
        cfg = hashlib.sha256(data.encode('ascii')).hexdigest()[:8]
        with tempfile.TemporaryDirectory() as tdir:
            fproj, file = os.path.join(tdir, 'proj.conll'), os.path.join(tdir, 'nonproj.conll')
            with open(fproj, 'w') as f:
                f.write('\n'.join([s.conll_format(arcs[i], rels[i]) for i, s in enumerate(sentences)]))
            # in cases when cfg files are deleted by new java executions
            subprocess.check_output([f"cd {path}; if [ ! -f {cfg}.mco ]; then sleep 30; fi;"
                                     f"java -jar {parser} -c {cfg} -m deproj -i {fproj} -o {file}"],
                                    stderr=subprocess.STDOUT,
                                    shell=True)
            arcs, rels, sent = [], [], []
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        sent = [line for line in sent if line[0].isdigit()]
                        arcs.append([int(line[6]) for line in sent])
                        rels.append([line[7] for line in sent])
                        sent = []
                    else:
                        sent.append(line.split('\t'))
            return arcs, rels

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        proj: bool = False,
        malt: str = None,
        **kwargs
    ) -> Iterable[CoNLLSentence]:
        r"""
        Loads the data in CoNLL-X format.
        Also supports for loading data from CoNLL-U file with comments and non-integer IDs.

        Args:
            data (Union[str, Iterable]):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            proj (bool):
                If ``True``, discards all non-projective sentences.
                Default: ``False``.
            malt (bool):
                If specified, projectivizes all the non-projective trees to pseudo-projective ones.
                Default: ``None``.

        Returns:
            A list of :class:`CoNLLSentence` instances.
        """

        isconll = False
        if lang is not None:
            tokenizer = Tokenizer(lang)
        with tempfile.TemporaryDirectory() as tdir:
            if isinstance(data, str) and os.path.exists(data):
                f = open(data)
                if data.endswith('.txt'):
                    lines = (i
                             for s in f
                             if len(s) > 1
                             for i in StringIO(self.toconll(s.split() if lang is None else tokenizer(s)) + '\n'))
                else:
                    if malt is not None:
                        f = open(CoNLL.projectivize(data, os.path.join(tdir, f"{os.path.basename(data)}.proj"), malt))
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
                    if isconll and self.training and proj and not sentence.projective:
                        logger.warning(f"Sentence {index} is not projective. Discarding it!")
                    else:
                        yield sentence
                        index += 1
                    sentence = []
                else:
                    sentence.append(line)


class CoNLLSentence(Sentence):
    r"""
    Sencence in CoNLL-X format.

    Args:
        transform (CoNLL):
            A :class:`~supar.utils.transform.CoNLL` object.
        lines (Sequence[str]):
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

    def __init__(self, transform: CoNLL, lines: Sequence[str], index: Optional[int] = None) -> CoNLLSentence:
        super().__init__(transform, index)

        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i - 1] = line
            else:
                self.annotations[len(self.values)] = line
                self.values.append(value)
        self.values = list(zip(*self.values))

    def __repr__(self):
        return self.conll_format()

    @property
    def projective(self):
        return CoNLL.isprojective(CoNLL.get_arcs(self.values[6]))

    def conll_format(self, arcs: Iterable[int] = None, rels: Iterable[str] = None):
        if arcs is None:
            arcs = self.values[6]
        if rels is None:
            rels = self.values[7]
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values[:6], arcs, rels, *self.values[8:]))}}
        return '\n'.join(merged.values()) + '\n'
