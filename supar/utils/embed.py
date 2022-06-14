# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Iterable, Optional, Union

import torch
from supar.utils.common import CACHE
from supar.utils.fn import download
from supar.utils.logging import progress_bar
from torch.distributions.utils import lazy_property


class Embedding(object):
    r"""
    Defines a container object for holding pretrained embeddings.
    This object is callable and behaves like :class:`torch.nn.Embedding`.
    For huge files, this object supports lazy loading, seeking to retrieve vectors from the disk on the fly if necessary.

    Currently available embeddings:
        - `GloVe`_
        - `Fasttext`_
        - `Giga`_
        - `Tencent`_

    Args:
        path (str):
            Path to the embedding file or short name registered in ``supar.utils.embed.PRETRAINED``.
        unk (Optional[str]):
            The string token used to represent OOV tokens. Default: ``None``.
        skip_first (bool)
            If ``True``, skips the first line of the embedding file. Default: ``False``.
        cache (bool):
            If ``True``, instead of loading entire embeddings into memory, seeks to load vectors from the disk once called.
            Default: ``True``.
        sep (str):
            Separator used by embedding file. Default: ``' '``.

    Examples:
        >>> import torch.nn as nn
        >>> from supar.utils.embed import Embedding
        >>> glove = Embedding.load('glove-6b-100')
        >>> glove
        GloVeEmbedding(n_tokens=400000, dim=100, unk=unk, cache=True)
        >>> fasttext = Embedding.load('fasttext-en')
        >>> fasttext
        FasttextEmbedding(n_tokens=2000000, dim=300, skip_first=True, cache=True)
        >>> giga = Embedding.load('giga-100')
        >>> giga
        GigaEmbedding(n_tokens=372846, dim=100, cache=True)
        >>> indices = torch.tensor([glove.vocab[i.lower()] for i in ['She', 'enjoys', 'playing', 'tennis', '.']])
        >>> indices
        tensor([  67, 8371,  697, 2140,    2])
        >>> glove(indices).shape
        torch.Size([5, 100])
        >>> glove(indices).equal(nn.Embedding.from_pretrained(glove.vectors)(indices))
        True

    .. _GloVe:
        https://nlp.stanford.edu/projects/glove/
    .. _Fasttext:
        https://fasttext.cc/docs/en/crawl-vectors.html
    .. _Giga:
        https://github.com/yzhangcs/parser/releases/download/v1.1.0/giga.100.zip
    .. _Tencent:
        https://ai.tencent.com/ailab/nlp/zh/download.html
    """

    CACHE = os.path.join(CACHE, 'data/embeds')

    def __init__(
        self,
        path: str,
        unk: Optional[str] = None,
        skip_first: bool = False,
        cache: bool = True,
        sep: str = ' ',
        **kwargs
    ) -> Embedding:
        super().__init__()

        self.path = path
        self.unk = unk
        self.skip_first = skip_first
        self.cache = cache
        self.sep = sep
        self.kwargs = kwargs

        self.vocab = {token: i for i, token in enumerate(self.tokens)}

    def __len__(self):
        return len(self.vocab)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_tokens={len(self)}, dim={self.dim}"
        if self.unk is not None:
            s += f", unk={self.unk}"
        if self.skip_first:
            s += f", skip_first={self.skip_first}"
        if self.cache:
            s += f", cache={self.cache}"
        s += ")"
        return s

    def __contains__(self, token):
        return token in self.vocab

    def __getitem__(self, key: Union[int, Iterable[int], torch.Tensor]) -> torch.Tensor:
        indices = key
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(key)
        if self.cache:
            elems, indices = indices.unique(return_inverse=True)
            with open(self.path) as f:
                vectors = []
                for index in elems.tolist():
                    f.seek(self.positions[index])
                    vectors.append(list(map(float, f.readline().strip().split(self.sep)[1:])))
                vectors = torch.tensor(vectors)
        else:
            vectors = self.vectors
        return torch.embedding(vectors, indices)

    def __call__(self, key: Union[int, Iterable[int], torch.Tensor]) -> torch.Tensor:
        return self[key]

    @property
    def dim(self):
        return len(self[0])

    @property
    def unk_index(self):
        if self.unk is not None:
            return self.vocab[self.unk]
        raise AttributeError

    @lazy_property
    def tokens(self):
        with open(self.path) as f:
            if self.skip_first:
                f.readline()
            return [line.strip().split(self.sep)[0] for line in progress_bar(f)]

    @lazy_property
    def vectors(self):
        with open(self.path) as f:
            if self.skip_first:
                f.readline()
            return torch.tensor([list(map(float, line.strip().split(self.sep)[1:])) for line in progress_bar(f)])

    @lazy_property
    def positions(self):
        with open(self.path) as f:
            if self.skip_first:
                f.readline()
            positions = [f.tell()]
            while True:
                line = f.readline()
                if line:
                    positions.append(f.tell())
                else:
                    break
            return positions

    @classmethod
    def load(cls, path: str, unk: Optional[str] = None, **kwargs) -> Embedding:
        if path in PRETRAINED:
            cfg = dict(**PRETRAINED[path])
            embed = cfg.pop('_target_')
            return embed(**cfg, **kwargs)
        return cls(path, unk, **kwargs)


class GloVeEmbedding(Embedding):

    r"""
    `GloVe`_: Global Vectors for Word Representation.
    Training is performed on aggregated global word-word co-occurrence statistics from a corpus,
    and the resulting representations showcase interesting linear substructures of the word vector space.

    Args:
        lang (str):
            Language code. Default: ``en``.
        reload (bool):
                If ``True``, forces a fresh download. Default: ``False``.

    Examples:
        >>> from supar.utils.embed import Embedding
        >>> Embedding.load('glove-6b-100')
        GloVeEmbedding(n_tokens=400000, dim=100, unk=unk, cache=True)

    .. _GloVe:
        https://nlp.stanford.edu/projects/glove/
    """

    def __init__(self, src: str = '6B', dim: int = 100, reload=False, *args, **kwargs) -> GloVeEmbedding:
        if src == '6B' or src == 'twitter.27B':
            url = f'https://nlp.stanford.edu/data/glove.{src}.zip'
        else:
            url = f'https://nlp.stanford.edu/data/glove.{src}.{dim}d.zip'
        path = os.path.join(os.path.join(self.CACHE, 'glove'), f'glove.{src}.{dim}d.txt')
        if not os.path.exists(path) or reload:
            download(url, os.path.join(self.CACHE, 'glove'), clean=True)

        super().__init__(path=path, unk='unk', *args, **kwargs, )


class FasttextEmbedding(Embedding):

    r"""
    `Fasttext`_ word embeddings for 157 languages, trained using CBOW, in dimension 300,
    with character n-grams of length 5, a window of size 5 and 10 negatives.

    Args:
        lang (str):
            Language code. Default: ``en``.
        reload (bool):
                If ``True``, forces a fresh download. Default: ``False``.

    Examples:
        >>> from supar.utils.embed import Embedding
        >>> Embedding.load('fasttext-en')
        FasttextEmbedding(n_tokens=2000000, dim=300, skip_first=True, cache=True)

    .. _Fasttext:
        https://fasttext.cc/docs/en/crawl-vectors.html
    """

    def __init__(self, lang: str = 'en', reload=False, *args, **kwargs) -> FasttextEmbedding:
        url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz'
        path = os.path.join(self.CACHE, 'fasttext', f'cc.{lang}.300.vec')
        if not os.path.exists(path) or reload:
            download(url, os.path.join(self.CACHE, 'fasttext'), clean=True)

        super().__init__(path=path, skip_first=True, *args, **kwargs)


class GigaEmbedding(Embedding):

    r"""
    `Giga`_ word embeddings, trained on Chinese Gigaword Third Edition for Chinese using word2vec,
    used by :cite:`zhang-etal-2020-efficient` and :cite:`zhang-etal-2020-fast`.

    Args:
        reload (bool):
            If ``True``, forces a fresh download. Default: ``False``.

    Examples:
        >>> from supar.utils.embed import Embedding
        >>> Embedding.load('giga-100')
        GigaEmbedding(n_tokens=372846, dim=100, cache=True)

    .. _Giga:
        https://github.com/yzhangcs/parser/releases/download/v1.1.0/giga.100.zip
    """

    def __init__(self, reload=False, *args, **kwargs) -> GigaEmbedding:
        url = 'https://github.com/yzhangcs/parser/releases/download/v1.1.0/giga.100.zip'
        path = os.path.join(self.CACHE, 'giga', 'giga.100.txt')
        if not os.path.exists(path) or reload:
            download(url, os.path.join(self.CACHE, 'giga'), clean=True)

        super().__init__(path=path, *args, **kwargs)


class TencentEmbedding(Embedding):

    r"""
    `Tencent`_ word embeddings.
    The embeddings are trained on large-scale text collected from news, webpages, and novels with Directional Skip-Gram.
    100-dimension and 200-dimension embeddings for over 12 million Chinese words are provided.

    Args:
        dim (int):
            Which dimension of the embeddings to use. Currently 100 and 200 are available. Default: 100.
        large (bool):
            If ``True``, uses large version with larger vocab size (12,287,936); 2,000,000 otherwise. Default: ``False``.
        reload (bool):
            If ``True``, forces a fresh download. Default: ``False``.

    .. _Tencent:
        https://ai.tencent.com/ailab/nlp/zh/download.html
    """

    def __init__(self, dim: int = 100, large: bool = False, reload=False, *args, **kwargs) -> TencentEmbedding:
        url = f'https://ai.tencent.com/ailab/nlp/zh/data/tencent-ailab-embedding-zh-d{dim}-v0.2.0{"" if large else "-s"}.tar.gz'  # noqa
        name = f'tencent-ailab-embedding-zh-d{dim}-v0.2.0{"" if large else "-s"}'
        path = os.path.join(os.path.join(self.CACHE, 'tencent'), name, f'{name}.txt')
        if not os.path.exists(path) or reload:
            download(url, os.path.join(self.CACHE, 'tencent'), clean=True)

        super().__init__(path=path, skip_first=True, *args, **kwargs)


PRETRAINED = {
    'glove-6b-50': {'_target_': GloVeEmbedding, 'src': '6B', 'dim': 50},
    'glove-6b-100': {'_target_': GloVeEmbedding, 'src': '6B', 'dim': 100},
    'glove-6b-200': {'_target_': GloVeEmbedding, 'src': '6B', 'dim': 200},
    'glove-6b-300': {'_target_': GloVeEmbedding, 'src': '6B', 'dim': 300},
    'glove-42b-300': {'_target_': GloVeEmbedding, 'src': '42B', 'dim': 300},
    'glove-840b-300': {'_target_': GloVeEmbedding, 'src': '84B', 'dim': 300},
    'glove-twitter-27b-25': {'_target_': GloVeEmbedding, 'src': 'twitter.27B', 'dim': 25},
    'glove-twitter-27b-50': {'_target_': GloVeEmbedding, 'src': 'twitter.27B', 'dim': 50},
    'glove-twitter-27b-100': {'_target_': GloVeEmbedding, 'src': 'twitter.27B', 'dim': 100},
    'glove-twitter-27b-200': {'_target_': GloVeEmbedding, 'src': 'twitter.27B', 'dim': 200},
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
    'tencent-100-b': {'_target_': TencentEmbedding, 'dim': 100, 'large': True},
    'tencent-200': {'_target_': TencentEmbedding, 'dim': 200},
    'tencent-200-b': {'_target_': TencentEmbedding, 'dim': 200, 'large': True},
}
