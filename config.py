# -*- coding: utf-8 -*-


class Config(object):
    n_embed = 100
    n_char_embed = 50
    n_char_out = 100
    n_lstm_hidden = 400
    n_lstm_layers = 3
    n_mlp_arc = 500
    n_mlp_lab = 100
    betas = (0.9, 0.9)
    epsilon = 1e-12
