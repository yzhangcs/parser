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

        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}
        self.label_dict = {l: i for i, l in enumerate(self.labels)}

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
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def char_to_id(self, sequence, fix_length=20):
        char_ids = torch.zeros(len(sequence), fix_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.char_dict.get(c, self.unk_index)
                                for c in word[:fix_length]])
            char_ids[i, :len(ids)] = ids

        return char_ids

    def label_to_id(self, sequence):
        return torch.tensor([self.label_dict.get(label, 0)
                             for label in sequence])

    def id_to_label(self, ids):
        return [self.labels[i] for i in ids]

    def read_embeddings(self, embed, unk=None):
        words = embed.words
        # if the UNK token has existed in pretrained vocab,
        # then replace it with a self-defined one
        if unk:
            words[words.index(unk)] = self.UNK

        self.extend(words)
        self.embeddings = torch.zeros(self.n_words, embed.dim)

        for i, word in enumerate(self.words):
            if word in embed:
                self.embeddings[i] = embed[word]
        self.embeddings /= torch.std(self.embeddings)

    def extend(self, words):
        self.words.extend(set(words).difference(self.word_dict))
        self.chars.extend(set(''.join(words)).difference(self.char_dict))
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    def numericalize(self, corpus):
        words, chars, heads, labels = [], [], [], []
        for sentence in corpus:
            words.append(self.word_to_id(sentence.words))
            chars.append(self.char_to_id(sentence.words))
            heads.append(torch.tensor(sentence.heads))
            labels.append(self.label_to_id(sentence.labels))

        return words, chars, heads, labels

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = list(w for w, f in corpus.words.items() if f >= min_freq)
        labels = list(corpus.labels)
        vocab = cls(words, labels)

        return vocab
