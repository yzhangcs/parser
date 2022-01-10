# -*- coding: utf-8 -*-

import os
import sys
import unicodedata
import urllib
import zipfile

import torch


def ispunct(token):
    return all(unicodedata.category(char).startswith('P') for char in token)


def isfullwidth(token):
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A'] for char in token)


def islatin(token):
    return all('LATIN' in unicodedata.name(char) for char in token)


def isdigit(token):
    return all('DIGIT' in unicodedata.name(char) for char in token)


def tohalfwidth(token):
    return unicodedata.normalize('NFKC', token)


def kmeans(x, k, max_it=32):
    r"""
    KMeans algorithm for clustering the sentences by length.

    Args:
        x (list[int]):
            The list of sentence lengths.
        k (int):
            The number of clusters.
            This is an approximate value. The final number of clusters can be less or equal to `k`.
        max_it (int):
            Maximum number of iterations.
            If centroids does not converge after several iterations, the algorithm will be early stopped.

    Returns:
        list[float], list[list[int]]:
            The first list contains average lengths of sentences in each cluster.
            The second is the list of clusters holding indices of data points.

    Examples:
        >>> x = torch.randint(10,20,(10,)).tolist()
        >>> x
        [15, 10, 17, 11, 18, 13, 17, 19, 18, 14]
        >>> centroids, clusters = kmeans(x, 3)
        >>> centroids
        [10.5, 14.0, 17.799999237060547]
        >>> clusters
        [[1, 3], [0, 5, 9], [2, 4, 6, 7, 8]]
    """

    # the number of clusters must not be greater than the number of datapoints
    x, k = torch.tensor(x, dtype=torch.float), min(len(x), k)
    # collect unique datapoints
    d = x.unique()
    # initialize k centroids randomly
    c = d[torch.randperm(len(d))[:k]]
    # assign each datapoint to the cluster with the closest centroid
    dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)

    for _ in range(max_it):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster and move that the empty one
        mask = torch.arange(k).unsqueeze(-1).eq(y)
        none = torch.where(~mask.any(-1))[0].tolist()
        while len(none) > 0:
            for i in none:
                # the biggest cluster
                b = torch.where(mask[mask.sum(-1).argmax()])[0]
                # the datapoint farthest from the centroid of cluster b
                f = dists[b].argmax()
                # update the assigned cluster of f
                y[b[f]] = i
                # re-calculate the mask
                mask = torch.arange(k).unsqueeze(-1).eq(y)
            none = torch.where(~mask.any(-1))[0].tolist()
        # update the centroids
        c, old = (x * mask).sum(-1) / mask.sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)
        # stop iteration early if the centroids converge
        if c.equal(old):
            break
    # assign all datapoints to the new-generated clusters
    # the empty ones are discarded
    assigned = y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


def stripe(x, n, w, offset=(0, 0), dim=1):
    r"""
    Returns a parallelogram stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 1 if returns a horizontal stripe; 0 otherwise.

    Returns:
        a parallelogram stripe of the tensor.

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

    x, seq_len = x.contiguous(), x.size(1)
    stride = list(x.stride())
    numel = stride[1]
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)


def pad(tensors, padding_value=0, total_length=None, padding_side='right'):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == 'left' else slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def download(url, reload=False):
    path = os.path.join(os.path.expanduser('~/.cache/supar'), os.path.basename(urllib.parse.urlparse(url).path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if reload:
        os.remove(path) if os.path.exists(path) else None
    if not os.path.exists(path):
        sys.stderr.write(f"Downloading: {url} to {path}\n")
        try:
            torch.hub.download_url_to_file(url, path, progress=True)
        except urllib.error.URLError:
            raise RuntimeError(f"File {url} unavailable. Please try other sources.")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as f:
            members = f.infolist()
            path = os.path.join(os.path.dirname(path), members[0].filename)
            if len(members) != 1:
                raise RuntimeError('Only one file (not dir) is allowed in the zipfile.')
            if reload or not os.path.exists(path):
                f.extractall(os.path.dirname(path))
    return path


def get_rng_state():
    state = {'rng_state': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state['rng_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda_rng_state'])
