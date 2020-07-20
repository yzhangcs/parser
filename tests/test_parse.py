# -*- coding: utf-8 -*-

from supar import Parser
import supar


def test_parse():
    sentence = ['The', 'dog', 'chases', 'the', 'cat', '.']
    for name in supar.PRETRAINED:
        parser = Parser.load(name)
        parser.predict([sentence], prob=True)
