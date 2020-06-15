# -*- coding: utf-8 -*-

from collections import defaultdict
from collections.abc import Iterable


class Vocab(object):

    def __init__(self, counter, min_freq=1, specials=[], unk_index=0):
        self.itos = list(specials)
        self.stoi = defaultdict(lambda: unk_index)
        self.stoi.update({token: i for i, token in enumerate(self.itos)})
        self.extend([token for token, freq in counter.items()
                     if freq >= min_freq])
        self.unk_index = unk_index
        self.n_init = len(self)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.stoi[key]
        elif not isinstance(key, Iterable):
            return self.itos[key]
        elif isinstance(key[0], str):
            return [self.stoi[i] for i in key]
        else:
            return [self.itos[i] for i in key]

    def __contains__(self, token):
        return token in self.stoi

    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        stoi = defaultdict(lambda: self.unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def extend(self, tokens):
        self.itos.extend(sorted(set(tokens).difference(self.stoi)))
        self.stoi.update({token: i for i, token in enumerate(self.itos)})


class FieldVocab(dict):
   """Surrogate for missing vocab in certain Transformers tokenizers."""
   def __init__(self, unk_token_id, items):
       super(FieldVocab, self).__init__(items)
       self.unk_token_id = unk_token_id

   def __getitem__(self, tok):
       return super(FieldVocab, self).get(tok, self.unk_token_id)

