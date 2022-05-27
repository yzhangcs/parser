# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional

import torch


class Embedding(object):

    def __init__(self, tokens: List[str], vectors: List[List[float]], unk: Optional[str] = None) -> Embedding:
        self.tokens = tokens
        self.vectors = torch.tensor(vectors)
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.pretrained

    @property
    def dim(self):
        return self.vectors.size(1)

    @property
    def unk_index(self):
        if self.unk is not None:
            return self.tokens.index(self.unk)
        else:
            raise AttributeError

    @classmethod
    def load(cls, path: str, unk: Optional[str] = None) -> Embedding:
        with open(path, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        tokens, vectors = zip(*[(s[0], list(map(float, s[1:])))
                                for s in splits])

        return cls(tokens, vectors, unk=unk)
