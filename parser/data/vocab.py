# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Vocab(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, words, labels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.PAD, self.UNK] + sorted(words)
        self.chars = [self.PAD, self.UNK] + sorted(set(''.join(words)))
        self.labels = sorted(labels)

        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.cdict = {c: i for i, c in enumerate(self.chars)}
        self.ldict = {l: i for i, l in enumerate(self.labels)}

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_labels = len(self.labels)
        self.n_train_words = self.n_words

    def __repr__(self):
        info = f"{self.__class__.__name__}(\n"
        info += f"  num of words: {self.n_words}\n"
        info += f"  num of chars: {self.n_chars}\n"
        info += f"  num of labels: {self.n_labels}\n"
        info += f")"

        return info

    def word_to_id(self, sequence):
        ids = [self.wdict[w] if w in self.wdict
               else self.wdict.get(w.lower(), self.unk_index)
               for w in sequence]
        ids = torch.tensor(ids, dtype=torch.long)

        return ids

    def char_to_id(self, sequence, fix_length=20):
        char_ids = torch.zeros(len(sequence), fix_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.cdict.get(c, self.unk_index)
                                for c in word[:fix_length]], dtype=torch.long)
            char_ids[i, :len(ids)] = ids

        return char_ids

    def label_to_id(self, sequence):
        ids = [self.ldict.get(l, 0) for l in sequence]
        ids = torch.tensor(ids, dtype=torch.long)

        return ids

    def id_to_label(self, ids):
        labels = (self.labels[i] for i in ids)

        return labels

    def read_embeddings(self, embed, unk=None, smooth=True):
        words = embed.words
        # if the UNK token has existed in pretrained vocab,
        # then replace it with a self-defined one
        if unk:
            words[words.index(unk)] = self.UNK

        self.extend(words)
        self.embeddings = torch.Tensor(self.n_words, embed.dim)

        for i, word in enumerate(self.words):
            if word in embed:
                self.embeddings[i] = embed[word]
            elif word.lower() in embed:
                self.embeddings[i] = embed[word.lower()]
            else:
                self.embeddings[i].zero_()
        if smooth:
            self.embeddings /= torch.std(self.embeddings)

    def extend(self, words):
        self.words.extend({w for w in words if w not in self.wdict})
        self.chars.extend({c for c in ''.join(words) if c not in self.cdict})
        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.cdict = {c: i for i, c in enumerate(self.chars)}
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = list(w for w, f in corpus.words.items() if f >= min_freq)
        labels = list(corpus.labels)
        words.remove(corpus.ROOT)
        labels.remove(corpus.ROOT)
        vocab = cls(words, labels)

        return vocab

    @classmethod
    def load(cls, fname):
        return torch.load(fname)

    def save(fname):
        torch.save(self, fname)
