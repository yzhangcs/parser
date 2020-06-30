# -*- coding: utf-8 -*-

from .parsers import (BiaffineParser, CRF2oDependencyParser,
                      CRFConstituencyParser, CRFDependencyParser,
                      MSTDependencyParser)

__all__ = ['BiaffineParser',
           'CRF2oDependencyParser',
           'CRFConstituencyParser',
           'CRFDependencyParser',
           'MSTDependencyParser']
