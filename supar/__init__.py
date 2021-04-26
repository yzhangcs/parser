# -*- coding: utf-8 -*-

from .parsers import (BiaffineDependencyParser,
                      BiaffineSemanticDependencyParser, CRF2oDependencyParser,
                      CRFConstituencyParser, CRFDependencyParser, Parser,
                      VIConstituencyParser, VIDependencyParser,
                      VISemanticDependencyParser)

__all__ = ['BiaffineDependencyParser',
           'CRFDependencyParser',
           'CRF2oDependencyParser',
           'VIDependencyParser',
           'CRFConstituencyParser',
           'VIConstituencyParser',
           'BiaffineSemanticDependencyParser',
           'VISemanticDependencyParser',
           'Parser']

__version__ = '1.1.0'

PARSER = {parser.NAME: parser for parser in [BiaffineDependencyParser,
                                             CRFDependencyParser,
                                             CRF2oDependencyParser,
                                             VIDependencyParser,
                                             CRFConstituencyParser,
                                             VIConstituencyParser,
                                             BiaffineSemanticDependencyParser,
                                             VISemanticDependencyParser]}

SRC = 'https://github.com/yzhangcs/parser/releases/download'
NAME = {
    'biaffine-dep-en': 'ptb.biaffine.dep.lstm.char',
    'biaffine-dep-zh': 'ctb7.biaffine.dep.lstm.char',
    'crf2o-dep-en': 'ptb.crf2o.dep.lstm.char',
    'crf2o-dep-zh': 'ctb7.crf2o.dep.lstm.char',
    'biaffine-dep-roberta-en': 'ptb.biaffine.dep.roberta',
    'biaffine-dep-electra-zh': 'ctb7.biaffine.dep.electra',
    'biaffine-dep-xlmr': 'ud.biaffine.dep.xlmr',
    'crf-con-en': 'ptb.crf.con.lstm.char',
    'crf-con-zh': 'ctb7.crf.con.lstm.char',
    'crf-con-roberta-en': 'ptb.crf.con.roberta',
    'crf-con-electra-zh': 'ctb7.crf.con.electra',
    'crf-con-xlmr': 'spmrl.crf.con.xlmr',
    'biaffine-sdp-en': 'dm.biaffine.sdp.lstm.tag-char-lemma',
    'biaffine-sdp-zh': 'semeval16.biaffine.sdp.lstm.tag-char-lemma',
    'vi-sdp-en': 'dm.vi.sdp.lstm.tag-char-lemma',
    'vi-sdp-zh': 'semeval16.vi.sdp.lstm.tag-char-lemma',
    'biaffine-sdp-roberta-en': 'dm.biaffine.sdp.roberta',
    'biaffine-sdp-electra-zh': 'semeval16.biaffine.sdp.electra'
}
MODEL = {n: f'{SRC}/v{__version__}/{m}.zip' for n, m in NAME.items()}
CONFIG = {n: f'{SRC}/v{__version__}/{m}.ini' for n, m in NAME.items()}
