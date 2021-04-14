# -*- coding: utf-8 -*-

import supar
from supar import Parser


def test_parse():
    sentence = ['The', 'dog', 'chases', 'the', 'cat', '.']
    for name in supar.PRETRAINED:
        parser = Parser.load(name)
        parser.predict(sentence, prob=True)
        parser.predict(' '.join(sentence), prob=True, lang=('zh' if name.endswith('zh') else 'en'))
