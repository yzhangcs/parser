# -*- coding: utf-8 -*-

from .parsers import (BiaffineDependencyParser, CRF2oDependencyParser,
                      CRFConstituencyParser, CRFDependencyParser,
                      CRFNPDependencyParser, Parser)

from .utils import Config

__all__ = ['Parser',
           'BiaffineDependencyParser',
           'CRFNPDependencyParser',
           'CRFDependencyParser',
           'CRF2oDependencyParser',
           'CRFConstituencyParser']

PARSER = {parser.NAME: parser for parser in [BiaffineDependencyParser,
                                             CRFNPDependencyParser,
                                             CRFDependencyParser,
                                             CRF2oDependencyParser,
                                             CRFConstituencyParser]}

PRETRAINED = {
    'biaffine-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dependency.char.zip',
    'biaffine-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dependency.char.zip',
    'crfnp-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crfnp.dependency.char.zip',
    'crfnp-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crfnp.dependency.char.zip',
    'crf-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.dependency.char.zip',
    'crf-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.dependency.char.zip',
    'crf2o-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf2o.dependency.char.zip',
    'crf2o-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf2o.dependency.char.zip',
    'crf-con-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.constituency.char.zip',
    'crf-con-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.constituency.char.zip'
}
