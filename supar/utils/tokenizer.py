# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import torch.distributed as dist
from supar.utils.parallel import is_master


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

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str) -> List[str]:
        from transformers import GPT2Tokenizer, GPT2TokenizerFast
        if isinstance(self.tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
            text = ' ' + text
        return self.tokenizer.tokenize(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __getstate__(self) -> Dict:
        return self.__dict__

    def __setstate__(self, state: Dict):
        self.__dict__.update(state)

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def vocab_size(self):
        return len(self.vocab)

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


class BPETokenizer:

    def __init__(
        self,
        path: str = None,
        files: Optional[List[str]] = None,
        vocab_size: Optional[int] = 32000,
        pad: Optional[str] = None,
        unk: Optional[str] = None,
        bos: Optional[str] = None,
        eos: Optional[str] = None
    ) -> BPETokenizer:

        from tokenizers import Tokenizer
        from tokenizers.decoders import BPEDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from tokenizers.trainers import BpeTrainer

        self.path = path
        self.files = files
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.special_tokens = [i for i in [pad, unk, bos, eos] if i is not None]

        if not os.path.exists(path) and is_master():
            # start to train a tokenizer from scratch
            self.tokenizer = Tokenizer(BPE(unk_token=unk))
            self.tokenizer.pre_tokenizer = WhitespaceSplit()
            self.tokenizer.decoder = BPEDecoder()
            self.tokenizer.train(files=files,
                                 trainer=BpeTrainer(vocab_size=vocab_size,
                                                    special_tokens=self.special_tokens,
                                                    end_of_word_suffix='</w>'))
            self.tokenizer.save(path)
        if dist.is_initialized():
            dist.barrier()
        self.tokenizer = Tokenizer.from_file(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.vocab_size})"

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: Union[str, List]) -> List[str]:
        is_pretokenized = isinstance(text, list)
        return self.tokenizer.encode(text, is_pretokenized=is_pretokenized).tokens

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __getstate__(self) -> Dict:
        return self.__dict__

    def __setstate__(self, state: Dict):
        self.__dict__.update(state)

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
