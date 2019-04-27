# -*- coding: utf-8 -*-

from parser.modules.dropout import SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            input_size = hidden_size * 2

        self.reset_parameters()

    def reset_parameters(self):
        for i in self.parameters():
            # apply orthogonal_ to weight
            if len(i.shape) > 1:
                nn.init.orthogonal_(i)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(i)

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        h, c = hx
        init_h, init_c = h, c
        output, seq_len = [], len(x)
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        if self.training:
            hid_mask = SharedDropout.get_mask(h, self.dropout)

        for t in steps:
            batch_size = batch_sizes[t]
            if len(h) < batch_size:
                h = torch.cat((h, init_h[last_batch_size:batch_size]))
                c = torch.cat((c, init_c[last_batch_size:batch_size]))
            else:
                h = h[:batch_size]
                c = c[:batch_size]
            h, c = cell(input=x[t], hx=(h, c))
            output.append(h)
            if self.training:
                h = h * hid_mask[:batch_size]
            last_batch_size = batch_size
        if reverse:
            output.reverse()
        output = torch.cat(output)

        return output

    def forward(self, x, hx=None):
        x, batch_sizes = x
        batch_size = batch_sizes[0]

        if hx is None:
            init = x.new_zeros(batch_size, self.hidden_size)
            hx = (init, init)

        for layer in range(self.num_layers):
            if self.training:
                mask = SharedDropout.get_mask(x[:batch_size], self.dropout)
                mask = torch.cat([mask[:batch_size]
                                  for batch_size in batch_sizes])
                x *= mask
            x = torch.split(x, batch_sizes.tolist())
            f_output = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.f_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=False)
            b_output = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.b_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=True)
            x = torch.cat([f_output, b_output], -1)
        x = PackedSequence(x, batch_sizes)

        return x
