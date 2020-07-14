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
    'biff-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dependency.zip',
    'biff-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dependency.zip',
    'mst-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.mst.dependency.zip',
    'mst-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.mst.dependency.zip',
    'crf-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.dependency.zip',
    'crf-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.dependency.zip',
    'crf2o-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf2o.dependency.zip',
    'crf2o-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf2o.dependency.zip',
    'crf-con-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.constituency.zip',
    'crf-con-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.constituency.zip'
}
