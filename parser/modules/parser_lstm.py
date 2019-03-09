# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ParserLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=False, dropout=0, bidirectional=False):
        super(ParserLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                                hidden_size=hidden_size))
            input_size = hidden_size * self.num_directions

        self.reset_parameters()

    def reset_parameters(self):
        for i in self.parameters():
            # apply orthogonal_ to weight
            if len(i.shape) > 1:
                nn.init.orthogonal_(i)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(i)

    def _lstm_forward(self, x, hx, mask, cell, in_mask,
                      hid_mask, reverse):
        output = []
        seq_len = x.size(0)
        if in_mask is not None:
            x = x * in_mask
        steps = reversed(range(seq_len)) if reverse else range(seq_len)

        for t in steps:
            h_next, c_next = cell(input=x[t], hx=hx)
            h_next = h_next * mask[t]
            c_next = c_next * mask[t]
            output.append(h_next)
            if hid_mask is not None:
                h_next = h_next * hid_mask
            hx = (h_next, c_next)
        if reverse:
            output.reverse()
        output = torch.stack(output, 0)

        return output

    def forward(self, x, mask, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)
            mask = mask.transpose(0, 1)
        mask = torch.unsqueeze(mask, dim=2).float()
        seq_len, batch_size, input_size = x.shape

        if hx is None:
            initial = x.new_zeros(batch_size, self.hidden_size)
            hx = (initial, initial)

        for layer in range(self.num_layers):
            in_mask, hid_mask, b_hid_mask = None, None, None
            if self.training:
                in_mask = torch.bernoulli(
                    x.new_full((batch_size, x.size(2)), 1 - self.dropout)
                ) / (1 - self.dropout)
                hid_mask = torch.bernoulli(
                    x.new_full((batch_size, self.hidden_size),
                               1 - self.dropout)
                ) / (1 - self.dropout)
                if self.bidirectional:
                    b_hid_mask = torch.bernoulli(
                        x.new_full((batch_size, self.hidden_size),
                                   1 - self.dropout)
                    ) / (1 - self.dropout)

            layer_output = self._lstm_forward(x=x,
                                              hx=hx,
                                              mask=mask,
                                              cell=self.f_cells[layer],
                                              in_mask=in_mask,
                                              hid_mask=hid_mask,
                                              reverse=False)

            if self.bidirectional:
                b_layer_output = self._lstm_forward(x=x,
                                                    hx=hx,
                                                    mask=mask,
                                                    cell=self.b_cells[layer],
                                                    in_mask=in_mask,
                                                    hid_mask=b_hid_mask,
                                                    reverse=True)
            if self.bidirectional:
                x = torch.cat([layer_output, b_layer_output], 2)
            else:
                x = layer_output

        if self.batch_first:
            x = x.transpose(0, 1)

        return x
