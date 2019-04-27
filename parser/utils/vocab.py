# -*- coding: utf-8 -*-

from collections import Counter

import regex
import torch


class Vocab(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, words, tags, rels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.PAD, self.UNK] + sorted(words)
        self.tags = [self.PAD, self.UNK] + sorted(tags)
        self.rels = sorted(rels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.tag_dict = {tag: i for i, tag in enumerate(self.tags)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}

        # ids of punctuation that appear in words
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))

        self.n_words = len(self.words)
        self.n_tags = len(self.tags)
        self.n_rels = len(self.rels)
        self.n_train_words = self.n_words

    def __repr__(self):
        info = f"{self.__class__.__name__}(\n"
        info += f"  num of words: {self.n_words}\n"
        info += f"  num of tags: {self.n_tags}\n"
        info += f"  num of rels: {self.n_rels}\n"
        info += f")"

        return info

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def tag2id(self, sequence):
        return torch.tensor([self.tag_dict.get(tag, self.unk_index)
                             for tag in sequence])

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 0)
                             for rel in sequence])

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def read_embeddings(self, embed, unk=None):
        words = embed.words
        # if the UNK token has existed in pretrained vocab,
        # then replace it with a self-defined one
        if unk in embed:
            words[words.index(unk)] = self.UNK

        self.extend(words)
        self.embeddings = torch.zeros(self.n_words, embed.dim)

        for i, word in enumerate(self.words):
            if word in embed:
                self.embeddings[i] = embed[word]
        self.embeddings /= torch.std(self.embeddings)

    def extend(self, words):
        self.words.extend(sorted(set(words).difference(self.word_dict)))
        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))
        self.n_words = len(self.words)

    def numericalize(self, corpus):
        words = [self.word2id(seq) for seq in corpus.words]
        tags = [self.tag2id(seq) for seq in corpus.tags]
        arcs = [torch.tensor(seq) for seq in corpus.heads]
        rels = [self.rel2id(seq) for seq in corpus.rels]

        return words, tags, arcs, rels

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word.lower() for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        tags = list({tag for seq in corpus.tags for tag in seq})
        rels = list({rel for seq in corpus.rels for rel in seq})
        vocab = cls(words, tags, rels)

        return vocab
