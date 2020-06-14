# -*- coding: utf-8 -*-

from supar import (biaffine_parser, crf2o_dependency, crf_constituency,
                   crf_dependency, mst_dependency)

MODELS = {
    'biaffine-parser': biaffine_parser,
    'crf-dependency': crf_dependency,
    'crf2o-dependency': crf2o_dependency,
    'crf_constituncy': crf_constituency,
    'mst_dependency': mst_dependency
}

if __name__ == '__main__':
    biaffine_parser.run()
