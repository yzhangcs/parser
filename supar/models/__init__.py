# -*- coding: utf-8 -*-

from .constituency import CRFConstituencyModel
from .dependency import (BiaffineParserModel, CRF2oDependencyModel,
                         CRFDependencyModel, MSTDependencyModel)

MODELS = {'biaffine-parser': BiaffineParserModel,
          'crf-dependency': CRFDependencyModel,
          'crf2o-dependency': CRF2oDependencyModel,
          'mst-dependency': MSTDependencyModel,
          'crf-constituency': CRFConstituencyModel}

__all__ = MODELS.keys()
