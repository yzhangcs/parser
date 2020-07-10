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
    'biff-dep-en': 'https://github.com/yzhangcs/supar/release/downloads/ptb.biaffine.dependency.zip',
    'biff-dep-zh': 'https://github.com/yzhangcs/supar/release/downloads/ctb7.biaffine.dependency.zip',
    'mst-dep-en': 'https://github.com/yzhangcs/parser/release/downloads/ptb.mst.dependency.zip',
    'mst-dep-zh': 'https://github.com/yzhangcs/parser/release/downloads/ctb7.mst.dependency.zip',
    'crf-dep-en': 'https://github.com/yzhangcs/parser/release/downloads/ptb.crf.dependency.zip',
    'crf-dep-zh': 'https://github.com/yzhangcs/parser/release/downloads/ctb7.crf.dependency.zip',
    'crf2o-dep-en': 'https://github.com/yzhangcs/parser/release/downloads/ptb.crf2o.dependency.zip',
    'crf2o-dep-zh': 'https://github.com/yzhangcs/parser/release/downloads/ctb7.crf2o.dependency.zip',
    'crf-con-en': 'https://github.com/yzhangcs/parser/release/downloads/ptb.crf.constituency.zip',
    'crf-con-zh': 'https://github.com/yzhangcs/parser/release/downloads/ctb7.crf.constituency.zip'
}
