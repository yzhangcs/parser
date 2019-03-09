# -*- coding: utf-8 -*-

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn(data):
    reprs = zip(
        *sorted(data, key=lambda x: len(x[0]), reverse=True)
    )
    reprs = (pad_sequence(i, True) for i in reprs)

    if torch.cuda.is_available():
        reprs = (i.cuda() for i in reprs)

    return reprs


class TextDataset(Dataset):

    def __init__(self, items):
        super(TextDataset, self).__init__()

        self.items = items

    def __getitem__(self, index):
        return tuple(item[index] for item in self.items)

    def __len__(self):
        return len(self.items[0])
