# -*- coding: utf-8 -*-

from supar import crf2o_dependency, crf_constituency, crf_dependency, parser

MODELS = {
    'biaffine-parser': parser,
    'crf-dependency': crf_dependency,
    'crf2o-dependency': crf2o_dependency,
    'crf_constituncy': crf_constituency
}

if __name__ == '__main__':
    crf2o_dependency.run()
