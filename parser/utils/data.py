# -*- coding: utf-8 -*-

from collections.abc import Iterable
from itertools import chain
from parser.utils.alg import kmeans

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


class TextDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(TextDataLoader, self).__init__(*args, **kwargs)

        self.fields = self.dataset.fields

    def __iter__(self):
        for raw_batch in super(TextDataLoader, self).__iter__():
            batch, device = [], 'cuda' if torch.cuda.is_available() else 'cpu'
            for data, field in zip(raw_batch, self.fields):
                if isinstance(data[0], torch.Tensor):
                    data = pad_sequence(data, True, field.pad_index).to(device)
                elif isinstance(data[0], Iterable):
                    data = [pad_sequence(f, True, field.pad_index).to(device)
                            for f in zip(*data)]
                batch.append(data)
            yield batch


class TextDataset(Dataset):

    def __init__(self, corpus, fields, n_buckets=1):
        super(TextDataset, self).__init__()

        self.corpus = corpus
        self.fields = list(chain(*[
            field if isinstance(field, Iterable) else [field]
            for field in fields if field is not None
        ]))
        for field in self.fields:
            value = field.numericalize(getattr(corpus, field.name))
            setattr(self, field.name, value)
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.centroids, self.clusters = kmeans(x=[len(i) for i in corpus],
                                               k=n_buckets)
        self.buckets = dict(zip(self.centroids, self.clusters))

    def __getitem__(self, index):
        for field in self.fields:
            yield getattr(self, field.name)[index]

    def __len__(self):
        return len(self.corpus)

    @property
    def loader(self):
        if hasattr(self, 'data_loader'):
            return self.data_loader
        else:
            raise AttributeError

    @loader.setter
    def loader(self, data_loader):
        self.data_loader = data_loader

    @classmethod
    def collate_fn(cls, batch):
        return (field for field in zip(*batch))


class TextSampler(Sampler):

    def __init__(self, buckets, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[
            (size, bucket) for size, bucket in buckets.items()
        ])
        # the number of chunks in each bucket, which is clipped by
        # range [1, len(bucket)]
        self.chunks = [
            min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
            for size, bucket in zip(self.sizes, self.buckets)
        ]

    def __iter__(self):
        # if shuffle, shffule both the buckets and samples in each bucket
        range_fn = torch.randperm if self.shuffle else torch.arange
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                           for j in range(self.chunks[i])]
            # DON'T use `torch.chunk` which may return wrong number of chunks
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        return sum(self.chunks)


def batchify(dataset, batch_size, shuffle=False):
    batch_sampler = TextSampler(buckets=dataset.buckets,
                                batch_size=batch_size,
                                shuffle=shuffle)
    loader = TextDataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            collate_fn=dataset.collate_fn)

    return loader
