# -*- coding: utf-8 -*-

from .models import (AttachJuxtaposeConstituencyParser,
                     BiaffineDependencyParser,
                     BiaffineSemanticDependencyParser, CRF2oDependencyParser,
                     CRFConstituencyParser, CRFDependencyParser,
                     TetraTaggingConstituencyParser, VIConstituencyParser,
                     VIDependencyParser, VISemanticDependencyParser)
from .parser import Parser
from .structs import (BiLexicalizedConstituencyCRF, ConstituencyCRF,
                      ConstituencyLBP, ConstituencyMFVI, Dependency2oCRF,
                      DependencyCRF, DependencyLBP, DependencyMFVI,
                      LinearChainCRF, MatrixTree, SemanticDependencyLBP,
                      SemanticDependencyMFVI, SemiMarkovCRF)

__all__ = [
    'Parser',
    'BiaffineDependencyParser',
    'CRFDependencyParser',
    'CRF2oDependencyParser',
    'VIDependencyParser',
    'AttachJuxtaposeConstituencyParser',
    'CRFConstituencyParser',
    'TetraTaggingConstituencyParser',
    'VIConstituencyParser',
    'BiaffineSemanticDependencyParser',
    'VISemanticDependencyParser',
    'LinearChainCRF',
    'SemiMarkovCRF',
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
    'SemanticDependencyMFVI'
]

__version__ = '1.1.4'

PARSER = {
    parser.NAME: parser for parser in [
        BiaffineDependencyParser,
        CRFDependencyParser,
        CRF2oDependencyParser,
        VIDependencyParser,
        AttachJuxtaposeConstituencyParser,
        CRFConstituencyParser,
        TetraTaggingConstituencyParser,
        VIConstituencyParser,
        BiaffineSemanticDependencyParser,
        VISemanticDependencyParser
    ]
}

SRC = {
    'github': 'https://github.com/yzhangcs/parser/releases/download',
    'hlt': 'http://hlt.suda.edu.cn/~yzhang/supar'
}
NAME = {
    'dep-biaffine-en': 'ptb.biaffine.dep.lstm.char',
    'dep-biaffine-zh': 'ctb7.biaffine.dep.lstm.char',
    'dep-crf2o-en': 'ptb.crf2o.dep.lstm.char',
    'dep-crf2o-zh': 'ctb7.crf2o.dep.lstm.char',
    'dep-biaffine-roberta-en': 'ptb.biaffine.dep.roberta',
    'dep-biaffine-electra-zh': 'ctb7.biaffine.dep.electra',
    'dep-biaffine-xlmr': 'ud.biaffine.dep.xlmr',
    'con-crf-en': 'ptb.crf.con.lstm.char',
    'con-crf-zh': 'ctb7.crf.con.lstm.char',
    'con-crf-roberta-en': 'ptb.crf.con.roberta',
    'con-crf-electra-zh': 'ctb7.crf.con.electra',
    'con-crf-xlmr': 'spmrl.crf.con.xlmr',
    'sdp-biaffine-en': 'dm.biaffine.sdp.lstm.tag-char-lemma',
    'sdp-biaffine-zh': 'semeval16.biaffine.sdp.lstm.tag-char-lemma',
    'sdp-vi-en': 'dm.vi.sdp.lstm.tag-char-lemma',
    'sdp-vi-zh': 'semeval16.vi.sdp.lstm.tag-char-lemma',
    'sdp-vi-roberta-en': 'dm.vi.sdp.roberta',
    'sdp-vi-electra-zh': 'semeval16.vi.sdp.electra'
}
MODEL = {src: {n: f"{link}/v1.1.0/{m}.zip" for n, m in NAME.items()} for src, link in SRC.items()}
CONFIG = {src: {n: f"{link}/v1.1.0/{m}.ini" for n, m in NAME.items()} for src, link in SRC.items()}


def compatible():
    import sys
    supar = sys.modules[__name__]
    if supar.__version__ < '1.2':
        sys.modules['supar.utils.transform'].CoNLL = supar.models.dep.biaffine.transform.CoNLL
        sys.modules['supar.utils.transform'].Tree = supar.models.const.crf.transform.Tree
        sys.modules['supar.parsers'] = supar.models
        sys.modules['supar.parsers.con'] = supar.models.const


compatible()
