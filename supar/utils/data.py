# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.distributed as dist
from supar.utils.fn import kmeans
from supar.utils.transform import Batch, Transform
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    r"""
    Dataset that is compatible with :class:`torch.utils.data.Dataset`, serving as a wrapper for manipulating all data fields
    with the operating behaviours defined in :class:`~supar.utils.transform.Transform`.
    The data fields of all the instantiated sentences can be accessed as an attribute of the dataset.

    Args:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform` or its derivations.
            The instance holds a series of loading and processing behaviours with regard to the specific data format.
        data (list[list] or str):
            A list of instances or a filename that will be passed into :meth:`transform.load`.
        kwargs (dict):
            Together with `data`, kwargs will be passed into :meth:`transform.load` to control the loading behaviour.

    Attributes:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform`.
        sentences (list[Sentence]):
            A list of sentences loaded from the data.
            Each sentence includes fields obeying the data format defined in ``transform``.
    """

    def __init__(self, transform: Transform, data: Union[str, List[List]], **kwargs) -> Dataset:
        super(Dataset, self).__init__()

        self.transform = transform
        self.sentences = transform.load(data, **kwargs)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"
        if hasattr(self, 'loader'):
            s += f", n_batches={len(self.loader)}"
        if hasattr(self, 'buckets'):
            s += f", n_buckets={len(self.buckets)}"
        s += ")"

        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return [getattr(sentence, name) for sentence in self.sentences]

    def __setattr__(self, name, value):
        if 'sentences' in self.__dict__ and name in self.sentences[0]:
            # restore the order of sequences in the buckets
            indices = torch.tensor([i for bucket in self.buckets.values() for i in bucket]).argsort()
            for index, sentence in zip(indices, self.sentences):
                setattr(sentence, name, value[index])
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def build(
        self,
        batch_size: int,
        n_buckets: int = 1,
        shuffle: bool = False,
        distributed: bool = False,
        n_workers: int = 0,
        pin_memory: bool = True
    ) -> Dataset:
        # numericalize all fields
        fields = self.transform(self.sentences)
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.buckets = dict(zip(*kmeans([len(s.fields[fields[0].name]) for s in self], n_buckets)))
        self.loader = DataLoader(dataset=self,
                                 batch_sampler=Sampler(self.buckets, batch_size, shuffle, distributed),
                                 num_workers=n_workers,
                                 collate_fn=lambda x: Batch(x),
                                 pin_memory=pin_memory)
        return self


class Sampler(torch.utils.data.Sampler):
    r"""
    Sampler that supports for bucketization and token-level batchification.

    Args:
        buckets (dict):
            A dict that maps each centroid to indices of clustered sentences.
            The centroid corresponds to the average length of all sentences in the bucket.
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as ``batch_size``.
        shuffle (bool):
            If ``True``, the sampler will shuffle both buckets and samples in each bucket. Default: ``False``.
        distributed (bool):
            If ``True``, the sampler will be used in conjunction with :class:`torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset.
            Default: ``False``.
    """

    def __init__(
        self,
        buckets: Dict[float, List],
        batch_size: int,
        shuffle: bool = False,
        distributed: bool = False
    ) -> Sampler:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket) for size, bucket in buckets.items()])
        # number of batches in each bucket, clipped by range [1, len(bucket)]
        self.n_batches = [min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
                          for size, bucket in zip(self.sizes, self.buckets)]
        self.rank, self.n_replicas, self.n_samples = 0, 1, sum(self.n_batches)
        if distributed:
            self.rank = dist.get_rank()
            self.n_replicas = dist.get_world_size()
            self.n_samples = sum(self.n_batches) // self.n_replicas + int(self.rank < sum(self.n_batches) % self.n_replicas)
        self.epoch = 1

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        total, count = 0, 0
        # if `shuffle=True`, shuffle both the buckets and samples in each bucket
        # for distributed training, make sure each process generates the same random sequence at each epoch
        range_fn = torch.arange if not self.shuffle else lambda x: torch.randperm(x, generator=g)
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.n_batches[i] + 1 for j in range(self.n_batches[i])]
            # DON'T use `torch.chunk` which may return wrong number of batches
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                if count == self.n_samples:
                    break
                if total % self.n_replicas == self.rank:
                    count += 1
                    yield [self.buckets[i][j] for j in batch.tolist()]
                total += 1
        self.epoch += 1

    def __len__(self):
        return self.n_samples
