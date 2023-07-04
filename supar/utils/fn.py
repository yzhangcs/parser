# -*- coding: utf-8 -*-

import gzip
import mmap
import os
import pickle
import shutil
import struct
import sys
import tarfile
import unicodedata
import urllib
import zipfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from supar.utils.common import CACHE
from supar.utils.parallel import wait


def ispunct(token: str, pos: str = None, puncts: Set = {'``', "''", ':', ',', '.', 'PU'}) -> bool:
    return all(unicodedata.category(char).startswith('P') for char in token) if pos is None else pos in puncts


def isfullwidth(token: str) -> bool:
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A'] for char in token)


def islatin(token: str) -> bool:
    return all('LATIN' in unicodedata.name(char) for char in token)


def isdigit(token: str) -> bool:
    return all('DIGIT' in unicodedata.name(char) for char in token)


def tohalfwidth(token: str) -> str:
    return unicodedata.normalize('NFKC', token)


def kmeans(x: List[int], k: int, max_it: int = 32) -> Tuple[List[float], List[List[int]]]:
    r"""
    KMeans algorithm for clustering the sentences by length.

    Args:
        x (List[int]):
            The list of sentence lengths.
        k (int):
            The number of clusters, which is an approximate value.
            The final number of clusters can be less or equal to `k`.
        max_it (int):
            Maximum number of iterations.
            If centroids does not converge after several iterations, the algorithm will be early stopped.

    Returns:
        List[float], List[List[int]]:
            The first list contains average lengths of sentences in each cluster.
            The second is the list of clusters holding indices of data points.

    Examples:
        >>> x = torch.randint(10, 20, (10,)).tolist()
        >>> x
        [15, 10, 17, 11, 18, 13, 17, 19, 18, 14]
        >>> centroids, clusters = kmeans(x, 3)
        >>> centroids
        [10.5, 14.0, 17.799999237060547]
        >>> clusters
        [[1, 3], [0, 5, 9], [2, 4, 6, 7, 8]]
    """

    x = torch.tensor(x, dtype=torch.float)
    # collect unique datapoints
    datapoints, indices, freqs = x.unique(return_inverse=True, return_counts=True)
    # the number of clusters must not be greater than the number of datapoints
    k = min(len(datapoints), k)
    # initialize k centroids randomly
    centroids = datapoints[torch.randperm(len(datapoints))[:k]]
    # assign each datapoint to the cluster with the closest centroid
    dists, y = torch.abs_(datapoints.unsqueeze(-1) - centroids).min(-1)

    for _ in range(max_it):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster and move that the empty one
        mask = torch.arange(k).unsqueeze(-1).eq(y)
        none = torch.where(~mask.any(-1))[0].tolist()
        for i in none:
            # the biggest cluster
            biggest = torch.where(mask[mask.sum(-1).argmax()])[0]
            # the datapoint farthest from the centroid of the biggest cluster
            farthest = dists[biggest].argmax()
            # update the assigned cluster of the farthest datapoint
            y[biggest[farthest]] = i
            # re-calculate the mask
            mask = torch.arange(k).unsqueeze(-1).eq(y)
        # update the centroids
        centroids, old = (datapoints * freqs * mask).sum(-1) / (freqs * mask).sum(-1), centroids
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(datapoints.unsqueeze(-1) - centroids).min(-1)
        # stop iteration early if the centroids converge
        if centroids.equal(old):
            break
    # assign all datapoints to the new-generated clusters
    # the empty ones are discarded
    assigned = y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = centroids[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(indices.unsqueeze(-1).eq(torch.where(y.eq(i))[0]).any(-1))[0].tolist() for i in assigned]

    return centroids, clusters


def stripe(x: torch.Tensor, n: int, w: int, offset: Tuple = (0, 0), horizontal: bool = True) -> torch.Tensor:
    r"""
    Returns a parallelogram stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        horizontal (bool): `True` if returns a horizontal stripe; `False` otherwise.

    Returns:
        A parallelogram stripe of the tensor.

    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> stripe(x, 2, 3)
        tensor([[0, 1, 2],
                [6, 7, 8]])
        >>> stripe(x, 2, 3, (1, 1))
        tensor([[ 6,  7,  8],
                [12, 13, 14]])
        >>> stripe(x, 2, 3, (1, 1), 0)
        tensor([[ 6, 11, 16],
                [12, 17, 22]])
    """

    x = x.contiguous()
    seq_len, stride = x.size(1), list(x.stride())
    numel = stride[1]
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=[(seq_len + 1) * numel, (1 if horizontal else seq_len) * numel] + stride[2:],
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)


def diagonal_stripe(x: torch.Tensor, offset: int = 1) -> torch.Tensor:
    r"""
    Returns a diagonal parallelogram stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 3 or more dims.
        offset (int): which diagonal to consider. Default: 1.

    Returns:
        A diagonal parallelogram stripe of the tensor.

    Examples:
        >>> x = torch.arange(125).view(5, 5, 5)
        >>> diagonal_stripe(x)
        tensor([[ 5],
                [36],
                [67],
                [98]])
        >>> diagonal_stripe(x, 2)
        tensor([[10, 11],
                [41, 42],
                [72, 73]])
        >>> diagonal_stripe(x, -2)
        tensor([[ 50,  51],
                [ 81,  82],
                [112, 113]])
    """

    x = x.contiguous()
    seq_len, stride = x.size(1), list(x.stride())
    n, w, numel = seq_len - abs(offset), abs(offset), stride[2]
    return x.as_strided(size=(n, w, *x.shape[3:]),
                        stride=[((seq_len + 1) * x.size(2) + 1) * numel] + stride[2:],
                        storage_offset=offset*stride[1] if offset > 0 else abs(offset)*stride[0])


def expanded_stripe(x: torch.Tensor, n: int, w: int, offset: Tuple = (0, 0)) -> torch.Tensor:
    r"""
    Returns an expanded parallelogram stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.

    Returns:
        An expanded parallelogram stripe of the tensor.

    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> expanded_stripe(x, 2, 3)
        tensor([[[ 0,  1,  2,  3,  4],
                 [ 5,  6,  7,  8,  9],
                 [10, 11, 12, 13, 14]],

                [[ 5,  6,  7,  8,  9],
                 [10, 11, 12, 13, 14],
                 [15, 16, 17, 18, 19]]])
        >>> expanded_stripe(x, 2, 3, (1, 1))
        tensor([[[ 5,  6,  7,  8,  9],
                 [10, 11, 12, 13, 14],
                 [15, 16, 17, 18, 19]],

                [[10, 11, 12, 13, 14],
                 [15, 16, 17, 18, 19],
                 [20, 21, 22, 23, 24]]])

    """
    x = x.contiguous()
    stride = list(x.stride())
    return x.as_strided(size=(n, w, *list(x.shape[1:])),
                        stride=stride[:1] + [stride[0]] + stride[1:],
                        storage_offset=(offset[1])*stride[0])


def binarize(
    data: Union[List[str], Dict[str, Iterable]],
    fbin: str = None,
    merge: bool = False
) -> Tuple[str, torch.Tensor]:
    start, meta = 0, defaultdict(list)
    # the binarized file is organized as:
    # `data`: pickled objects
    # `meta`: a dict containing the pointers of each kind of data
    # `index`: fixed size integers representing the storage positions of the meta data
    with open(fbin, 'wb') as f:
        # in this case, data should be a list of binarized files
        if merge:
            for file in data:
                if not os.path.exists(file):
                    raise RuntimeError("Some files are missing. Please check the paths")
                mi = debinarize(file, meta=True)
                for key, val in mi.items():
                    val[:, 0] += start
                    meta[key].append(val)
                with open(file, 'rb') as fi:
                    length = int(sum(val[:, 1].sum() for val in mi.values()))
                    f.write(fi.read(length))
                start = start + length
            meta = {key: torch.cat(val) for key, val in meta.items()}
        else:
            for key, val in data.items():
                for i in val:
                    buf = i if isinstance(i, (bytes, bytearray)) else pickle.dumps(i)
                    f.write(buf)
                    meta[key].append((start, len(buf)))
                    start = start + len(buf)
            meta = {key: torch.tensor(val) for key, val in meta.items()}
        pickled = pickle.dumps(meta)
        # append the meta data to the end of the bin file
        f.write(pickled)
        # record the positions of the meta data
        f.write(struct.pack('LL', start, len(pickled)))
    return fbin, meta


def debinarize(
    fbin: str,
    pos_or_key: Optional[Union[Tuple[int, int], str]] = (0, 0),
    meta: bool = False,
    unpickle: bool = False
) -> Union[Any, Iterable[Any]]:
    with open(fbin, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if meta or isinstance(pos_or_key, str):
            length = len(struct.pack('LL', 0, 0))
            mm.seek(-length, os.SEEK_END)
            offset, length = struct.unpack('LL', mm.read(length))
            mm.seek(offset)
            if meta:
                return pickle.loads(mm.read(length))
            # fetch by key
            objs, meta = [], pickle.loads(mm.read(length))[pos_or_key]
            for offset, length in meta.tolist():
                mm.seek(offset)
                objs.append(mm.read(length) if unpickle else pickle.loads(mm.read(length)))
            return objs
        # fetch by positions
        offset, length = pos_or_key
        mm.seek(offset)
        return mm.read(length) if unpickle else pickle.loads(mm.read(length))


def pad(
    tensors: List[torch.Tensor],
    padding_value: int = 0,
    total_length: int = None,
    padding_side: str = 'right'
) -> torch.Tensor:
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == 'left' else slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


@wait
def download(url: str, path: Optional[str] = None, reload: bool = False, clean: bool = False) -> str:
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    if path is None:
        path = CACHE
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, filename)
    if reload and os.path.exists(path):
        os.remove(path)
    if not os.path.exists(path):
        sys.stderr.write(f"Downloading {url} to {path}\n")
        try:
            torch.hub.download_url_to_file(url, path, progress=True)
        except (ValueError, urllib.error.URLError):
            raise RuntimeError(f"File {url} unavailable. Please try other sources.")
    return extract(path, reload, clean)


def extract(path: str, reload: bool = False, clean: bool = False) -> str:
    extracted = path
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as f:
            extracted = os.path.join(os.path.dirname(path), f.infolist()[0].filename)
            if reload or not os.path.exists(extracted):
                f.extractall(os.path.dirname(path))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as f:
            extracted = os.path.join(os.path.dirname(path), f.getnames()[0])
            if reload or not os.path.exists(extracted):
                f.extractall(os.path.dirname(path))
    elif path.endswith('.gz'):
        extracted = path[:-3]
        with gzip.open(path) as fgz:
            with open(extracted, 'wb') as f:
                shutil.copyfileobj(fgz, f)
    if clean:
        os.remove(path)
    return extracted


def resolve_config(args: Union[Dict, DictConfig]) -> DictConfig:
    OmegaConf.register_new_resolver("eval", eval)
    return DictConfig(OmegaConf.to_container(args, resolve=True))


def collect_args(args: Union[Dict, DictConfig]) -> DictConfig:
    for key in ('self', 'cls', '__class__'):
        args.pop(key, None)
    args.update(args.pop('kwargs', dict()))
    return DictConfig(args)


def get_rng_state() -> Dict[str, torch.Tensor]:
    state = {'rng_state': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: Dict) -> None:
    torch.set_rng_state(state['rng_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda_rng_state'])
