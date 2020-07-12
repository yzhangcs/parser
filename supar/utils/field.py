# -*- coding: utf-8 -*-

from collections import Counter

import torch
from supar.utils.fn import pad
from supar.utils.vocab import Vocab


class RawField(object):
    """
    Defines a general datatype.

    A RawField instance does not assume any property of the datatype and
    it holds parameters relating to how a datatype should be processed.

    Args:
        name (str):
            The name of the field.
        fn (function, default: None):
            The function used for preprocessing the examples.
    """

    def __init__(self, name, fn=None):
        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        return self.fn(sequence) if self.fn is not None else sequence

    def transform(self, sequences):
        return [self.preprocess(seq) for seq in sequences]

    def compose(self, sequences):
        return sequences


class Field(RawField):
    """
    Defines a datatype together with instructions for converting to Tensor.
    Field class models common text processing datatypes that can be represented by tensors.
    It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.

    Args:
        name (str):
            The name of the field.
        pad_token (str, default: None):
            The string token used as padding.
        unk_token (str, default: None):
            The string token used to represent OOV words.
        bos_token (str, default: None):
            A token that will be prepended to every example using this
            field, or None for no bos_token.
        eos_token (str, default: None)::
            A token that will be appended to every example using this
            field, or None for no eos_token.
        lower (bool, default: False):
            Whether to lowercase the text in this field.
        use_vocab (bool, default: True):
            Whether to use a Vocab object.
            If False, the data in this field should already be numerical.
        tokenize (function, default: None):
            The function used to tokenize strings using this field into sequential examples.
        fn (function, default: None):
            The function used for preprocessing the examples.
    """

    def __init__(self, name, pad=None, unk=None, bos=None, eos=None,
                 lower=False, use_vocab=True, tokenize=None, fn=None):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn

        self.specials = [token for token in [pad, unk, bos, eos]
                         if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
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
        s += ", ".join(params)
        s += ")"

        return s

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

    def preprocess(self, sequence):
        """
        Load a single example using this field, tokenizing if necessary.
        The sequence will be first passed to `self.fn` if available.
        If `self.tokenize` is not None, the input will be tokenized.
        Then the input will be optionally lowercased.

        Args (List):
            The sequence to be preprocessed.

        Returns:
            sequence (List):
                the preprocessed sequence.
        """

        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]

        return sequence

    def build(self, dataset, min_freq=1, embed=None):
        """
        Construct the Vocab object for this field from the dataset.
        If the Vocab has already existed, this function will have no effect.

        Args:
            dataset (Dataset):
                A Dataset instance. One of the attributes should be named after the name of this field.
            min_freq (int, default: 1):
                The minimum frequency needed to include a token in the vocabulary.
            embed (Embedding, default: None):
                An Embedding instance, words in which will be extended to the vocabulary.
        """

        if hasattr(self, 'vocab'):
            return
        sequences = getattr(dataset, self.name)
        counter = Counter(token
                          for seq in sequences
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

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors
            self.embed /= torch.std(self.embed)

    def transform(self, sequences):
        """
        Turns a list of sequences that use this field into tensors.

        Each sequence is first preprocessed and then numericalized if needed.

        Args:
            sequences (List[List[str]]):
                A List of sequences.

        Returns:
            sequences (List[Tensor]):
                A list of tensors transformed from the input sequences.
        """

        sequences = [self.preprocess(seq) for seq in sequences]
        if self.use_vocab:
            sequences = [self.vocab[seq] for seq in sequences]
        if self.bos:
            sequences = [[self.bos_index] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [self.eos_index] for seq in sequences]
        sequences = [torch.tensor(seq) for seq in sequences]

        return sequences

    def compose(self, sequences):
        """
        Compose a batch of sequences into a padded tensor.

        Args:
            sequences (List[Tensor]):
                A List of tensors.

        Returns:
            A padded tensor converted to proper device.
        """

        return pad(sequences, self.pad_index).to(self.device)


class SubwordField(Field):

    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super().__init__(*args, **kwargs)

    def build(self, dataset, min_freq=1, embed=None):
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(dataset, self.name)
        counter = Counter(piece
                          for seq in sequences
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

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors

    def transform(self, sequences):
        sequences = [[self.preprocess(token) for token in seq]
                     for seq in sequences]
        if self.fix_len <= 0:
            self.fix_len = max(len(token) for seq in sequences for token in seq)
        if self.use_vocab:
            sequences = [[[self.vocab[i] for i in token] if token else [self.unk] for token in seq]
                         for seq in sequences]
        if self.bos:
            sequences = [[[self.bos_index]] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [[self.eos_index]] for seq in sequences]
        lens = [min(self.fix_len, max(len(ids) for ids in seq)) for seq in sequences]
        sequences = [pad([torch.tensor(ids[:i]) for ids in seq], self.pad_index, i)
                     for i, seq in zip(lens, sequences)]

        return sequences


class ChartField(Field):

    def build(self, dataset, min_freq=1):
        counter = Counter(label
                          for seq in getattr(dataset, self.name)
                          for i, j, label in self.preprocess(seq))

        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

    def transform(self, sequences):
        sequences = [self.preprocess(seq) for seq in sequences]
        spans, labels = [], []

        for sequence in sequences:
            seq_len = sequence[0][1] + 1
            span_chart = torch.full((seq_len, seq_len), self.pad_index).bool()
            label_chart = torch.full((seq_len, seq_len), self.pad_index).long()
            for i, j, label in sequence:
                span_chart[i, j] = 1
                label_chart[i, j] = self.vocab[label]
            spans.append(span_chart)
            labels.append(label_chart)

        return list(zip(spans, labels))

    def compose(self, sequences):
        return [pad(i).to(self.device) for i in zip(*sequences)]
