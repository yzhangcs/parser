# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Union

import torch.distributed as dist
from supar.utils.parallel import is_dist, is_master
from supar.utils.vocab import Vocab


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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str) -> List[str]:
        from tokenizers.pre_tokenizers import ByteLevel
        if isinstance(self.tokenizer.backend_tokenizer.pre_tokenizer, ByteLevel):
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
        return defaultdict(lambda: self.tokenizer.vocab[self.unk], self.tokenizer.get_vocab())

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

    def decode(self, text: List) -> str:
        return self.tokenizer.decode(text, skip_special_tokens=True, clean_up_tokenization_spaces=False)


class BPETokenizer:

    def __init__(
        self,
        path: str = None,
        files: Optional[List[str]] = None,
        vocab_size: Optional[int] = 32000,
        min_freq: Optional[int] = 2,
        dropout: float = None,
        backend: str = 'huggingface',
        pad: Optional[str] = None,
        unk: Optional[str] = None,
        bos: Optional[str] = None,
        eos: Optional[str] = None,
    ) -> BPETokenizer:

        self.path = path
        self.files = files
        self.min_freq = min_freq
        self.dropout = dropout or .0
        self.backend = backend
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.special_tokens = [i for i in [pad, unk, bos, eos] if i is not None]

        if backend == 'huggingface':
            from tokenizers import Tokenizer
            from tokenizers.decoders import BPEDecoder
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import WhitespaceSplit
            from tokenizers.trainers import BpeTrainer
            path = os.path.join(path, 'tokenizer.json')
            if is_master() and not os.path.exists(path):
                # start to train a tokenizer from scratch
                self.tokenizer = Tokenizer(BPE(dropout=dropout, unk_token=unk))
                self.tokenizer.pre_tokenizer = WhitespaceSplit()
                self.tokenizer.decoder = BPEDecoder()
                self.tokenizer.train(files=files,
                                     trainer=BpeTrainer(vocab_size=vocab_size,
                                                        min_frequency=min_freq,
                                                        special_tokens=self.special_tokens,
                                                        end_of_word_suffix='</w>'))
                self.tokenizer.save(path)
            if is_dist():
                dist.barrier()
            self.tokenizer = Tokenizer.from_file(path)
            self.vocab = self.tokenizer.get_vocab()

        elif backend == 'subword-nmt':
            import argparse
            from argparse import Namespace

            from subword_nmt.apply_bpe import BPE, read_vocabulary
            from subword_nmt.learn_joint_bpe_and_vocab import learn_joint_bpe_and_vocab
            fmerge = os.path.join(path, 'merge.txt')
            fvocab = os.path.join(path, 'vocab.txt')
            separator = '@@'
            if is_master() and (not os.path.exists(fmerge) or not os.path.exists(fvocab)):
                with tempfile.TemporaryDirectory() as ftemp:
                    fall = os.path.join(ftemp, 'fall')
                    with open(fall, 'w') as f:
                        for file in files:
                            with open(file) as fi:
                                f.write(fi.read())
                    learn_joint_bpe_and_vocab(Namespace(input=[argparse.FileType()(fall)],
                                                        output=argparse.FileType('w')(fmerge),
                                                        symbols=vocab_size,
                                                        separator=separator,
                                                        vocab=[argparse.FileType('w')(fvocab)],
                                                        min_frequency=min_freq,
                                                        total_symbols=False,
                                                        verbose=False,
                                                        num_workers=32))
            if is_dist():
                dist.barrier()
            self.tokenizer = BPE(codes=open(fmerge), separator=separator, vocab=read_vocabulary(open(fvocab), None))
            self.vocab = Vocab(counter=Counter(self.tokenizer.vocab),
                               specials=self.special_tokens,
                               unk_index=self.special_tokens.index(unk))
        else:
            raise ValueError(f'Unsupported backend: {backend} not in (huggingface, subword-nmt)')

    def __repr__(self) -> str:
        s = self.__class__.__name__ + f'({self.vocab_size}, min_freq={self.min_freq}'
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        s += f", backend={self.backend}"
        if self.pad is not None:
            s += f", pad={self.pad}"
        if self.unk is not None:
            s += f", unk={self.unk}"
        if self.bos is not None:
            s += f", bos={self.bos}"
        if self.eos is not None:
            s += f", eos={self.eos}"
        s += ')'
        return s

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: Union[str, List]) -> List[str]:
        is_pretokenized = isinstance(text, list)
        if self.backend == 'huggingface':
            return self.tokenizer.encode(text, is_pretokenized=is_pretokenized).tokens
        else:
            if not is_pretokenized:
                text = text.split()
            return self.tokenizer.segment_tokens(text, dropout=self.dropout)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def decode(self, text: List) -> str:
        if self.backend == 'huggingface':
            return self.tokenizer.decode(text)
        else:
            text = self.vocab[text]
            text = ' '.join([i for i in text if i not in self.special_tokens])
            return re.sub(f'({self.tokenizer.separator} )|({self.tokenizer.separator} ?$)', '', text)
