# -*- coding: utf-8 -*-

import os

import supar
from supar import Parser


def test_parse():
    sents = {'en': 'She enjoys playing tennis.',
             'zh': '她喜欢打网球.',
             'de': 'Sie spielt gerne Tennis.',
             'fr': 'Elle aime jouer au tennis.',
             'ru': 'Она любит играть в теннис.',
             'he': 'היא נהנית לשחק טניס.'}
    tokenized_sents = {'en': ['She', 'enjoys', 'playing', 'tennis', '.'],
                       'zh': ['她', '喜欢', '打', '网球', '.'],
                       'de': ['Sie', 'spielt', 'gerne', 'Tennis', '.'],
                       'fr': ['Elle', 'aime', 'jouer', 'au', 'tennis', '.'],
                       'ru': ['Она', 'любит', 'играть', 'в', 'теннис', '.'],
                       'he': ['היא', 'נהנית', 'לשחק', 'טניס', '.']}
    for name, model in supar.NAME.items():
        if 'xlmr' in name or 'roberta' in name or 'electra' in name:
            continue
        parser = Parser.load(name, reload=True)
        if name.endswith(('en', 'zh')):
            lang = name[-2:]
            parser.predict(sents[lang], prob=True, lang=lang)
            parser.predict(tokenized_sents[lang], prob=True, lang=None)
        else:
            for lang in sents:
                parser.predict(sents[lang], prob=True, lang=lang)
            parser.predict(list(tokenized_sents.values()), prob=True, lang=None)
        os.remove(os.path.join(os.path.expanduser('~/.cache/supar'), model))
