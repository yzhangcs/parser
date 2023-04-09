# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import torch
from torch.distributions.utils import lazy_property

from supar.utils.fn import debinarize
from supar.utils.logging import get_logger, progress_bar

logger = get_logger(__name__)


class Transform(object):
    r"""
    A :class:`Transform` object corresponds to a specific data format, which holds several instances of data fields
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


class Batch(object):

    def __init__(self, sentences: Iterable[Sentence]) -> Batch:
        self.sentences = sentences
        self.names, self.fields = [], {}

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([f"{name}" for name in self.names])})'

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.fields[self.names[index]]

    def __getattr__(self, name):
        return [s.fields[name] for s in self.sentences]

    def __setattr__(self, name: str, value: Iterable[Any]):
        if name not in ('sentences', 'fields', 'names'):
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
    def lens(self):
        return torch.tensor([len(i) for i in self.sentences]).to(self.device, non_blocking=True)

    @lazy_property
    def mask(self):
        return self.lens.unsqueeze(-1).gt(self.lens.new_tensor(range(self.lens.max())))

    def compose(self, transform: Transform) -> Batch:
        for f in transform.flattened_fields:
            self.names.append(f.name)
            self.fields[f.name] = f.compose([s.fields[f.name] for s in self.sentences])
        return self

    def shrink(self, batch_size: Optional[int] = None) -> Batch:
        if batch_size is None:
            batch_size = len(self) // 2
        if batch_size <= 0:
            raise RuntimeError(f"The batch has only {len(self)} sentences and can't be shrinked!")
        return Batch([self.sentences[i] for i in torch.randperm(len(self))[:batch_size].tolist()])

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
        state = vars(self)
        if 'fields' in state:
            state['fields'] = {
                name: ((value.dtype, value.tolist())
                       if isinstance(value, torch.Tensor)
                       else value)
                for name, value in state['fields'].items()
            }
        return state

    def __setstate__(self, state):
        if 'fields' in state:
            state['fields'] = {
                name: (torch.tensor(value[1], dtype=value[0])
                       if isinstance(value, tuple) and isinstance(value[0], torch.dtype)
                       else value)
                for name, value in state['fields'].items()
            }
            self.__dict__.update(state)

    def __len__(self):
        try:
            return len(next(iter(self.fields.values())))
        except Exception:
            raise AttributeError("Cannot get size of a sentence with no fields")

    @lazy_property
    def size(self):
        return len(self)

    def numericalize(self, fields):
        for f in fields:
            self.fields[f.name] = next(f.transform([getattr(self, f.name)]))
        self.pad_index = fields[0].pad_index
        return self

    @classmethod
    def from_cache(cls, fbin: str, pos: Tuple[int, int]) -> Sentence:
        return debinarize(fbin, pos)
