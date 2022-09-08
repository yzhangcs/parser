# -*- coding: utf-8 -*-

from .models import (AttachJuxtaposeConstituencyParser,
                     BiaffineDependencyParser,
                     BiaffineSemanticDependencyParser, CRF2oDependencyParser,
                     CRFConstituencyParser, CRFDependencyParser,
                     VIConstituencyParser, VIDependencyParser,
                     VISemanticDependencyParser)
from .parser import Parser
from .structs import (BiLexicalizedConstituencyCRF, ConstituencyCRF,
                      ConstituencyLBP, ConstituencyMFVI, Dependency2oCRF,
                      DependencyCRF, DependencyLBP, DependencyMFVI,
                      LinearChainCRF, MatrixTree, SemanticDependencyLBP,
                      SemanticDependencyMFVI)

__all__ = ['Parser',
           'BiaffineDependencyParser',
           'CRFDependencyParser',
           'CRF2oDependencyParser',
           'VIDependencyParser',
           'CRFConstituencyParser',
           'AttachJuxtaposeConstituencyParser',
           'VIConstituencyParser',
           'BiaffineSemanticDependencyParser',
           'VISemanticDependencyParser',
           'LinearChainCRF',
           'MatrixTree',
           'DependencyCRF',
           'Dependency2oCRF',
           'ConstituencyCRF',
           'BiLexicalizedConstituencyCRF',
           'DependencyLBP',
           'DependencyMFVI',
           'ConstituencyLBP',
           'ConstituencyMFVI',
           'SemanticDependencyLBP',
           'SemanticDependencyMFVI']

__version__ = '1.1.4'

PARSER = {parser.NAME: parser for parser in [BiaffineDependencyParser,
                                             CRFDependencyParser,
                                             CRF2oDependencyParser,
                                             VIDependencyParser,
                                             CRFConstituencyParser,
                                             AttachJuxtaposeConstituencyParser,
                                             VIConstituencyParser,
                                             BiaffineSemanticDependencyParser,
                                             VISemanticDependencyParser]}

SRC = {'github': 'https://github.com/yzhangcs/parser/releases/download',
       'hlt': 'http://hlt.suda.edu.cn/~yzhang/supar'}
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
    'vi-sdp-roberta-en': 'dm.vi.sdp.roberta',
    'vi-sdp-electra-zh': 'semeval16.vi.sdp.electra'
}
MODEL = {src: {n: f"{link}/v1.1.0/{m}.zip" for n, m in NAME.items()} for src, link in SRC.items()}
CONFIG = {src: {n: f"{link}/v1.1.0/{m}.ini" for n, m in NAME.items()} for src, link in SRC.items()}
