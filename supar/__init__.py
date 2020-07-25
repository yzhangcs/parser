# -*- coding: utf-8 -*-

from .parsers import (BiaffineDependencyParser, CRF2oDependencyParser,
                      CRFConstituencyParser, CRFDependencyParser,
                      CRFNPDependencyParser, Parser)

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
    'biaffine-dep-en': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ptb.biaffine.dependency.char.zip',
    'biaffine-dep-zh': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ctb7.biaffine.dependency.char.zip',
    'crfnp-dep-en': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ptb.crfnp.dependency.char.zip',
    'crfnp-dep-zh': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ctb7.crfnp.dependency.char.zip',
    'crf-dep-en': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ptb.crf.dependency.char.zip',
    'crf-dep-zh': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ctb7.crf.dependency.char.zip',
    'crf2o-dep-en': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ptb.crf2o.dependency.char.zip',
    'crf2o-dep-zh': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ctb7.crf2o.dependency.char.zip',
    'crf-con-en': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ptb.crf.constituency.char.zip',
    'crf-con-zh': 'https://github.com/yzhangcs/supar/releases/download/v0.1.0/ctb7.crf.constituency.char.zip'
}
