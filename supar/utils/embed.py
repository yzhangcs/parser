# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import torch
from supar.utils.common import CACHE
from supar.utils.fn import download
from supar.utils.logging import progress_bar
from supar.utils.vocab import Vocab
from torch.distributions.utils import lazy_property


class Embedding(object):

    CACHE = os.path.join(CACHE, 'data/embeds')

    def __init__(
        self,
        path: str,
        unk: Optional[str] = None,
        skip_first: bool = False,
        split: str = ' ',
        cache: bool = False,
        **kwargs
    ) -> Embedding:
        super().__init__()

        self.path = path
        self.unk = unk
        self.skip_first = skip_first
        self.split = split
        self.cache = cache
        self.kwargs = kwargs

        self.vocab = Vocab(Counter(self.tokens), unk_index=self.tokens.index(unk) if unk is not None else 0)

    def __len__(self):
        return len(self.vocab)

    def __contains__(self, token):
        return token in self.vocab

    def __getitem__(self, key):
        return self.vectors[self.vocab[key]]

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_tokens={len(self)}, dim={self.dim}"
        if self.unk is not None:
            s += f", unk={self.unk}"
        if self.skip_first:
            s += f", skip_first={self.skip_first}"
        s += ")"
        return s

    @property
    def dim(self):
        return len(self[self.vocab[0]])

    @property
    def unk_index(self):
        if self.unk is not None:
            return self.vocab[self.unk]
        raise AttributeError

    @lazy_property
    def tokens(self):
        with open(self.path, 'r') as f:
            if self.skip_first:
                f.readline()
            return [line.split(self.split)[0] for line in progress_bar(f)]

    @lazy_property
    def vectors(self):
        with open(self.path, 'r') as f:
            if self.skip_first:
                f.readline()
            return torch.tensor([list(map(float, line.strip().split(self.split)[1:])) for line in progress_bar(f)])

    @classmethod
    def load(cls, path: str, unk: Optional[str] = None, **kwargs) -> Embedding:
        if path in PRETRAINED:
            cfg = dict(**PRETRAINED[path])
            embed = cfg.pop('_target_')
            return embed(**cfg, **kwargs)
        return cls(path, unk, **kwargs)


class GloveEmbedding(Embedding):

    def __init__(self, src: str = '6B', dim: int = 100, reload=False, *args, **kwargs) -> GloveEmbedding:
        if src == '6B' or src == 'twitter.27B':
            url = f'https://nlp.stanford.edu/data/glove.{src}.zip'
        else:
            url = f'https://nlp.stanford.edu/data/glove.{src}.{dim}d.zip'
        path = os.path.join(os.path.join(self.CACHE, 'glove'), f'glove.{src}.{dim}d.txt')
        if not os.path.exists(path) or reload:
            download(url, os.path.join(self.CACHE, 'glove'), clean=True)

        super().__init__(path=path, unk='unk', *args, **kwargs, )


class FasttextEmbedding(Embedding):

    def __init__(self, lang: str = 'en', reload=False, *args, **kwargs) -> FasttextEmbedding:
        url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz'
        path = os.path.join(self.CACHE, 'fasttext', f'cc.{lang}.300.vec')
        if not os.path.exists(path) or reload:
            download(url, os.path.join(self.CACHE, 'fasttext'), clean=True)

        super().__init__(path=path, skip_first=True, *args, **kwargs)


class GigaEmbedding(Embedding):

    def __init__(self, reload=False, *args, **kwargs) -> GigaEmbedding:
        url = 'https://github.com/yzhangcs/parser/releases/download/v1.1.0/giga.100.zip'
        path = os.path.join(self.CACHE, 'giga', 'giga.100.txt')
        if not os.path.exists(path) or reload:
            download(url, os.path.join(self.CACHE, 'giga'), clean=True)

        super().__init__(path=path, *args, **kwargs)


class TencentEmbedding(Embedding):

    def __init__(self, dim: int = 100, big: bool = False, reload=False, *args, **kwargs) -> TencentEmbedding:
        url = f'https://ai.tencent.com/ailab/nlp/zh/data/tencent-ailab-embedding-zh-d{dim}-v0.2.0{"" if big else "-s"}.tar.gz'  # noqa
        name = f'tencent-ailab-embedding-zh-d{dim}-v0.2.0{"" if big else "-s"}'
        path = os.path.join(os.path.join(self.CACHE, 'tencent'), name, f'{name}.txt')
        if not os.path.exists(path) or reload:
            download(url, os.path.join(self.CACHE, 'tencent'), clean=True)

        super().__init__(path=path, skip_first=True, *args, **kwargs)


PRETRAINED = {
    'glove-6b-50': {'_target_': GloveEmbedding, 'src': '6B', 'dim': 50},
    'glove-6b-100': {'_target_': GloveEmbedding, 'src': '6B', 'dim': 100},
    'glove-6b-200': {'_target_': GloveEmbedding, 'src': '6B', 'dim': 200},
    'glove-6b-300': {'_target_': GloveEmbedding, 'src': '6B', 'dim': 300},
    'glove-42b-300': {'_target_': GloveEmbedding, 'src': '42B', 'dim': 300},
    'glove-840b-300': {'_target_': GloveEmbedding, 'src': '84B', 'dim': 300},
    'glove-twitter-27b-25': {'_target_': GloveEmbedding, 'src': 'twitter.27B', 'dim': 25},
    'glove-twitter-27b-50': {'_target_': GloveEmbedding, 'src': 'twitter.27B', 'dim': 50},
    'glove-twitter-27b-100': {'_target_': GloveEmbedding, 'src': 'twitter.27B', 'dim': 100},
    'glove-twitter-27b-200': {'_target_': GloveEmbedding, 'src': 'twitter.27B', 'dim': 200},
    'fasttext-bg': {'_target_': FasttextEmbedding, 'lang': 'bg'},
    'fasttext-ca': {'_target_': FasttextEmbedding, 'lang': 'ca'},
    'fasttext-cs': {'_target_': FasttextEmbedding, 'lang': 'cs'},
    'fasttext-de': {'_target_': FasttextEmbedding, 'lang': 'de'},
    'fasttext-en': {'_target_': FasttextEmbedding, 'lang': 'en'},
    'fasttext-es': {'_target_': FasttextEmbedding, 'lang': 'es'},
    'fasttext-fr': {'_target_': FasttextEmbedding, 'lang': 'fr'},
    'fasttext-it': {'_target_': FasttextEmbedding, 'lang': 'it'},
    'fasttext-nl': {'_target_': FasttextEmbedding, 'lang': 'nl'},
    'fasttext-no': {'_target_': FasttextEmbedding, 'lang': 'no'},
    'fasttext-ro': {'_target_': FasttextEmbedding, 'lang': 'ro'},
    'fasttext-ru': {'_target_': FasttextEmbedding, 'lang': 'ru'},
    'giga-100': {'_target_': GigaEmbedding},
    'tencent-100': {'_target_': TencentEmbedding, 'dim': 100},
    'tencent-100-b': {'_target_': TencentEmbedding, 'dim': 100, 'big': True},
    'tencent-200': {'_target_': TencentEmbedding, 'dim': 200},
    'tencent-200-b': {'_target_': TencentEmbedding, 'dim': 200, 'big': True},
}
