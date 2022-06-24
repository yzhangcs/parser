# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List


class Tokenizer:

    def __init__(self, lang: str = 'en') -> Tokenizer:
        import stanza
        try:
            self.pipeline = stanza.Pipeline(lang=lang, processors='tokenize', verbose=False, tokenize_no_ssplit=True)
        except Exception:
            stanza.download(lang=lang, resources_url='stanford')
            self.pipeline = stanza.Pipeline(lang=lang, processors='tokenize', verbose=False, tokenize_no_ssplit=True)

    def __call__(self, text: str) -> List[str]:
        return [i.text for i in self.pipeline(text).sentences[0].tokens]


class TransformerTokenizer:

    def __init__(self, name) -> TransformerTokenizer:
        from transformers import AutoTokenizer
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __call__(self, text: str) -> List[str]:
        from transformers import GPT2Tokenizer, GPT2TokenizerFast
        if isinstance(self.tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
            text = ' ' + text
        return self.tokenizer.tokenize(text)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def pad(self):
        return self.tokenizer.pad_token

    @property
    def unk(self):
        return self.tokenizer.unk_token

    @property
    def bos(self):
        return self.tokenizer.bos_token or self.tokenizer.cls_token

    @property
    def eos(self):
        return self.tokenizer.eos_token or self.tokenizer.sep_token
