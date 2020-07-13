# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist
from supar.utils.alg import kmeans


class Dataset(torch.utils.data.Dataset):
    """
    Dataset compatible with `torch.utils.data.Dataset`.
    This serves as a wrapper for manipulating all data fields
    with the operating behaviours defined in the `Transform` class.
    The data fields in all the instantiated sentences can be accessed as an attribute of the dataset.

    Args:
        transform (Transform):
            An instance of a subclass of the `Transform` class and its derivations.
            The instance holds a series of loading and processing behaviours with regard to the specfic data format.
        data (List[List] or str):
            A list of instances or a filename.
            This will be passed into `transform.load` to load the input data.
        kwargs (Dict):
            Keyword arguments that will be passed into `transform.load` together with `data`
            to control the loading behaviour.

    Attributes:
        transform (Transform):
            The passed instance of `Transform`.
        sentences (List[Sentence]):
            A list of sentences loaded from the data.
            Each sentence includes fields obeying the data format defined in transform.
    """

    def __init__(self, transform, data, **kwargs):
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
        if not hasattr(self, 'fields'):
            raise RuntimeError("The fields are not numericalized. Please build the dataset first.")
        for d in self.fields.values():
            yield d[index]

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return [getattr(sentence, name) for sentence in self.sentences]

    def __setattr__(self, name, value):
        if 'sentences' in self.__dict__ and name in self.sentences[0]:
            # restore the order of sequences in the buckets
            indices = torch.tensor([i
                                    for bucket in self.buckets.values()
                                    for i in bucket]).argsort()
            for index, sentence in zip(indices, self.sentences):
                setattr(sentence, name, value[index])
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        # only pickles the Transform object and sentences
        return {'transform': self.transform, 'sentences': self.sentences}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def collate_fn(self, batch):
        return {f: d for f, d in zip(self.fields.keys(), zip(*batch))}

    def build(self, batch_size, n_buckets=1, shuffle=False, distributed=False):
        # numericalize all fields
        self.fields = self.transform(self.sentences)
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.lengths = [len(i) for i in self.fields[next(iter(self.fields))]]
        self.buckets = dict(zip(*kmeans(self.lengths, n_buckets)))
        self.loader = DataLoader(dataset=self,
                                 batch_sampler=Sampler(buckets=self.buckets,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle,
                                                       distributed=distributed),
                                 collate_fn=self.collate_fn)


class DataLoader(torch.utils.data.DataLoader):
    """
    DataLoader, matching with the `Dataset` class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            yield [f.compose(d) for f, d in batch.items()]


class Sampler(torch.utils.data.Sampler):
    """
    Sampler supporting for bucketizing and token-level batchification.

    Args:
        buckets (Dict):
            The dict that maps each centroid to the indices of the clustering sentences.
            The centroid corresponds to the average length of all sentences in the bucket (required).
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as batch_size (required).
        shuffle (bool, default: False):
            If True, the sampler will shuffle both the buckets and samples in each bucket (optional).
        distributed (bool, default: False):
            If True, the sample will be used be used in conjunction with `torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset (optional).
    """

    def __init__(self, buckets, batch_size, shuffle=False, distributed=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket) for size, bucket in buckets.items()])
        # number of chunks in each bucket, clipped by range [1, len(bucket)]
        self.chunks = [min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
                       for size, bucket in zip(self.sizes, self.buckets)]

        self.rank = dist.get_rank() if distributed else 0
        self.replicas = dist.get_world_size() if distributed else 1
        self.samples = sum(self.chunks) // self.replicas
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        range_fn = torch.arange
        # if shuffle, shuffle both the buckets and samples in each bucket
        # for distributed training, make sure each process generte the same random sequence at each epoch
        if self.shuffle:
            def range_fn(x):
                return torch.randperm(x, generator=g)
        total, count = 0, 0
        # TODO: more elegant way to deal with uneven data, which we directly discard right now
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                           for j in range(self.chunks[i])]
            # DON'T use `torch.chunk` which may return wrong number of chunks
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                if count == self.samples:
                    break
                if total % self.replicas == self.rank:
                    count += 1
                    yield [self.buckets[i][j] for j in batch.tolist()]
                total += 1
        self.epoch += 1

    def __len__(self):
        return self.samples
