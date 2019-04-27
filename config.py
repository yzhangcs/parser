# -*- coding: utf-8 -*-


class Config(object):

    # [Network]
    n_embed = 100
    n_tag_embed = 100
    embed_dropout = 0.33
    n_lstm_hidden = 400
    n_lstm_layers = 3
    lstm_dropout = 0.33
    n_mlp_arc = 500
    n_mlp_rel = 100
    mlp_dropout = 0.33

    # [Optimizer]
    lr = 2e-3
    beta_1 = 0.9
    beta_2 = 0.9
    epsilon = 1e-12
    decay = .75
    decay_steps = 5000

    # [Run]
    batch_size = 200
    epochs = 1000
    patience = 100
