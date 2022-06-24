# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, List, Optional

import torch
from supar.utils.data import Dataset
from supar.utils.embed import Embedding
from supar.utils.fn import pad
from supar.utils.logging import progress_bar
from supar.utils.vocab import Vocab


class RawField(object):
    r"""
    Defines a general datatype.

    A :class:`RawField` object does not assume any property of the datatype and
    it holds parameters relating to how a datatype should be processed.

    Args:
        name (str):
            The name of the field.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(self, name: str, fn: Optional[Callable] = None) -> RawField:
        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence: Iterable) -> Iterable:
        return self.fn(sequence) if self.fn is not None else sequence

    def transform(self, sequences: Iterable[List]) -> Iterable[List]:
        return (self.preprocess(seq) for seq in sequences)

    def compose(self, sequences: Iterable[List]) -> Iterable[List]:
        return sequences


class Field(RawField):
    r"""
    Defines a datatype together with instructions for converting to :class:`~torch.Tensor`.
    :class:`Field` models common text processing datatypes that can be represented by tensors.
    It holds a :class:`~supar.utils.vocab.Vocab` object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The :class:`Field` object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method.

    Args:
        name (str):
            The name of the field.
        pad_token (str):
            The string token used as padding. Default: ``None``.
        unk_token (str):
            The string token used to represent OOV words. Default: ``None``.
        bos_token (str):
            A token that will be prepended to every example using this field, or ``None`` for no `bos_token`.
            Default: ``None``.
        eos_token (str):
            A token that will be appended to every example using this field, or ``None`` for no `eos_token`.
        lower (bool):
            Whether to lowercase the text in this field. Default: ``False``.
        use_vocab (bool):
            Whether to use a :class:`~supar.utils.vocab.Vocab` object.
            If ``False``, the data in this field should already be numerical.
            Default: ``True``.
        tokenize (function):
            The function used to tokenize strings using this field into sequential examples. Default: ``None``.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(
        self,
        name: str,
        pad: Optional[str] = None,
        unk: Optional[str] = None,
        bos: Optional[str] = None,
        eos: Optional[str] = None,
        lower: bool = False,
        use_vocab: bool = True,
        tokenize: Optional[Callable] = None,
        fn: Optional[Callable] = None
    ) -> Field:
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn

        self.specials = [token for token in [pad, unk, bos, eos] if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if hasattr(self, 'vocab'):
            params.append(f"vocab_size={len(self.vocab)}")
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        return s + ', '.join(params) + ')'

    @property
    def pad_index(self):
        if self.pad is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)

    @property
    def unk_index(self):
        if self.unk is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.unk]
        return self.specials.index(self.unk)

    @property
    def bos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.bos]
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.eos]
        return self.specials.index(self.eos)

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess(self, sequence: Iterable) -> Iterable:
        r"""
        Loads a single example using this field, tokenizing if necessary.
        The sequence will be first passed to ``fn`` if available.
        If ``tokenize`` is not None, the input will be tokenized.
        Then the input will be lowercased optionally.

        Args:
            sequence (Iterable):
                The sequence to be preprocessed.

        Returns:
            A list of preprocessed sequence.
        """

        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]
        return sequence

    def build(
        self,
        dataset: Dataset,
        min_freq: int = 1,
        embed: Optional[Embedding] = None,
        norm: Callable = None
    ) -> Field:
        r"""
        Constructs a :class:`~supar.utils.vocab.Vocab` object for this field from the dataset.
        If the vocabulary has already existed, this function will have no effect.

        Args:
            dataset (Dataset):
                A :class:`~supar.utils.data.Dataset` object.
                One of the attributes should be named after the name of this field.
            min_freq (int):
                The minimum frequency needed to include a token in the vocabulary. Default: 1.
            embed (Embedding):
                An Embedding object, words in which will be extended to the vocabulary. Default: ``None``.
            norm (Callable):
                Callable function used for normalizing embedding weights. Default: ``None``.
        """

        if hasattr(self, 'vocab'):
            return
        counter = Counter(token
                          for seq in progress_bar(getattr(dataset, self.name))
                          for token in self.preprocess(seq))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.update(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors
            if norm is not None:
                self.embed = norm(self.embed)
        return self

    def transform(self, sequences: Iterable[List[str]]) -> Iterable[torch.Tensor]:
        r"""
        Turns a list of sequences that use this field into tensors.

        Each sequence is first preprocessed and then numericalized if needed.

        Args:
            sequences (Iterable[list[str]]):
                A list of sequences.

        Returns:
            A list of tensors transformed from the input sequences.
        """

        for seq in sequences:
            seq = self.preprocess(seq)
            if self.use_vocab:
                seq = self.vocab[seq]
            if self.bos:
                seq = [self.bos_index] + seq
            if self.eos:
                seq = seq + [self.eos_index]
            yield torch.tensor(seq)

    def compose(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        r"""
        Composes a batch of sequences into a padded tensor.

        Args:
            batch (Iterable[~torch.Tensor]):
                A list of tensors.

        Returns:
            A padded tensor converted to proper device.
        """

        return pad(batch, self.pad_index).to(self.device)


class SubwordField(Field):
    r"""
    A field that conducts tokenization and numericalization over each token rather the sequence.

    This is customized for models requiring character/subword-level inputs, e.g., CharLSTM and BERT.

    Args:
        fix_len (int):
            A fixed length that all subword pieces will be padded to.
            This is used for truncating the subword pieces that exceed the length.
            To save the memory, the final length will be the smaller value
            between the max length of subword pieces in a batch and `fix_len`.

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        >>> field = SubwordField('bert',
                                 pad=tokenizer.pad_token,
                                 unk=tokenizer.unk_token,
                                 bos=tokenizer.cls_token,
                                 eos=tokenizer.sep_token,
                                 fix_len=20,
                                 tokenize=tokenizer.tokenize)
        >>> field.vocab = tokenizer.get_vocab()  # no need to re-build the vocab
        >>> next(field.transform([['This', 'field', 'performs', 'token-level', 'tokenization']]))
        tensor([[  101,     0,     0],
                [ 1188,     0,     0],
                [ 1768,     0,     0],
                [10383,     0,     0],
                [22559,   118,  1634],
                [22559,  2734,     0],
                [  102,     0,     0]])
    """

    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super().__init__(*args, **kwargs)

    def build(
        self,
        dataset: Dataset,
        min_freq: int = 1,
        embed: Optional[Embedding] = None,
        norm: Callable = None
    ) -> SubwordField:
        if hasattr(self, 'vocab'):
            return
        counter = Counter(piece
                          for seq in progress_bar(getattr(dataset, self.name))
                          for token in seq
                          for piece in self.preprocess(token))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.update(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors
            if norm is not None:
                self.embed = norm(self.embed)
        return self

    def transform(self, sequences: Iterable[List[str]]) -> Iterable[torch.Tensor]:
        for seq in sequences:
            seq = [self.preprocess(token) for token in seq]
            if self.use_vocab:
                seq = [[self.vocab[i] if i in self.vocab else self.unk_index for i in token] if token else [self.unk_index]
                       for token in seq]
            if self.bos:
                seq = [[self.bos_index]] + seq
            if self.eos:
                seq = seq + [[self.eos_index]]
            if self.fix_len > 0:
                seq = [ids[:self.fix_len] for ids in seq]
            yield pad([torch.tensor(ids) for ids in seq], self.pad_index)


class ChartField(Field):
    r"""
    Field dealing with chart inputs.

    Examples:
        >>> chart = [[    None,    'NP',    None,    None,  'S|<>',     'S'],
                     [    None,    None, 'VP|<>',    None,    'VP',    None],
                     [    None,    None,    None, 'VP|<>', 'S::VP',    None],
                     [    None,    None,    None,    None,    'NP',    None],
                     [    None,    None,    None,    None,    None,  'S|<>'],
                     [    None,    None,    None,    None,    None,    None]]
        >>> next(field.transform([chart]))
        tensor([[ -1,  37,  -1,  -1, 107,  79],
                [ -1,  -1, 120,  -1, 112,  -1],
                [ -1,  -1,  -1, 120,  86,  -1],
                [ -1,  -1,  -1,  -1,  37,  -1],
                [ -1,  -1,  -1,  -1,  -1, 107],
                [ -1,  -1,  -1,  -1,  -1,  -1]])
    """

    def build(
        self,
        dataset: Dataset,
        min_freq: int = 1
    ) -> ChartField:
        counter = Counter(i
                          for chart in progress_bar(getattr(dataset, self.name))
                          for row in self.preprocess(chart)
                          for i in row if i is not None)
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)
        return self

    def transform(self, charts: Iterable[List[List]]) -> Iterable[torch.Tensor]:
        for chart in charts:
            chart = self.preprocess(chart)
            if self.use_vocab:
                chart = [[self.vocab[i] if i is not None else -1 for i in row] for row in chart]
            if self.bos:
                chart = [[self.bos_index]*len(chart[0])] + chart
            if self.eos:
                chart = chart + [[self.eos_index]*len(chart[0])]
            yield torch.tensor(chart)
