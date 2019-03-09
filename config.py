# -*- coding: utf-8 -*-


class Config(object):
    ftrain = 'data/cdt-train-10k.conll'
    fdev = 'data/cdt-dev.conll'
    ftest = 'data/cdt-test.conll'
    fembed = 'data/giga.100.txt'
    n_embed = 100
    n_char_embed = 50
    n_char_out = 100
    n_lstm_hidden = 150
    n_lstm_layers = 2
    n_mlp_arc = 200
    n_mlp_lab = 100
    betas = (0.9, 0.9)
    epsilon = 1e-12
