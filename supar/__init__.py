# -*- coding: utf-8 -*-

from .parsers import (BiaffineDependencyParser, CRF2oDependencyParser,
                      CRFConstituencyParser, CRFDependencyParser,
                      MSTDependencyParser, Parser)

__all__ = ['Parser',
           'BiaffineDependencyParser',
           'MSTDependencyParser',
           'CRFDependencyParser',
           'CRF2oDependencyParser',
           'CRFConstituencyParser']

PARSER = {parser.NAME: parser for parser in [BiaffineDependencyParser,
                                             MSTDependencyParser,
                                             CRFDependencyParser,
                                             CRF2oDependencyParser,
                                             CRFConstituencyParser]}

PRETRAINED = {
    'loc-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dependency.char.zip',
    'loc-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dependency.char.zip',
    'mtree-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.mst.dependency.char.zip',
    'mtree-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.mst.dependency.char.zip',
    'crf-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.dependency.char.zip',
    'crf-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.dependency.char.zip',
    'crf2o-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf2o.dependency.char.zip',
    'crf2o-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf2o.dependency.char.zip',
    'crf-con-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.constituency.char.zip',
    'crf-con-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.constituency.char.zip'
}
