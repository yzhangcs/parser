# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from supar.modules.dropout import SharedDropout
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence


class CharLSTM(nn.Module):
    r"""
    CharLSTM aims to generate character-level embeddings for tokens.
    It summarizes the information of characters in each token to an embedding using a LSTM layer.

    Args:
        n_char (int):
            The number of characters.
        n_embed (int):
            The size of each embedding vector as input to LSTM.
        n_hidden (int):
            The size of each LSTM hidden state.
        n_out (int):
            The size of each output vector. Default: 0.
            If 0, equals to the size of hidden states.
        pad_index (int):
            The index of the padding token in the vocabulary. Default: 0.
        dropout (float):
            The dropout ratio of CharLSTM hidden states. Default: 0.
    """

    def __init__(
        self,
        n_chars: int,
        n_embed: int,
        n_hidden: int,
        n_out: int = 0,
        pad_index: int = 0,
        dropout: float = 0
    ) -> CharLSTM:
        super().__init__()

        self.n_chars = n_chars
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_out = n_out or n_hidden
        self.pad_index = pad_index

        self.embed = nn.Embedding(num_embeddings=n_chars, embedding_dim=n_embed)
        self.lstm = nn.LSTM(input_size=n_embed, hidden_size=n_hidden//2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(in_features=n_hidden, out_features=self.n_out) if n_hidden != self.n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.n_chars}, {self.n_embed}"
        if self.n_hidden != self.n_out:
            s += f", n_hidden={self.n_hidden}"
        s += f", n_out={self.n_out}, pad_index={self.pad_index}"
        if self.dropout.p != 0:
            s += f", dropout={self.dropout.p}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
                Characters of all tokens.
                Each token holds no more than `fix_len` characters, and the excess is cut off directly.
        Returns:
            ~torch.Tensor:
                The embeddings of shape ``[batch_size, seq_len, n_out]`` derived from the characters.
        """

        # [batch_size, seq_len, fix_len]
        mask = x.ne(self.pad_index)
        # [batch_size, seq_len]
        lens = mask.sum(-1)
        char_mask = lens.gt(0)

        # [n, fix_len, n_embed]
        x = self.embed(x[char_mask])
        x = pack_padded_sequence(x, lens[char_mask].tolist(), True, False)
        x, (h, _) = self.lstm(x)
        # [n, fix_len, n_hidden]
        h = self.dropout(torch.cat(torch.unbind(h), -1))
        # [n, fix_len, n_out]
        h = self.projection(h)
        # [batch_size, seq_len, n_out]
        return h.new_zeros(*lens.shape, self.n_out).masked_scatter_(char_mask.unsqueeze(-1), h)


class VariationalLSTM(nn.Module):
    r"""
    VariationalLSTM :cite:`yarin-etal-2016-dropout` is an variant of the vanilla bidirectional LSTM
    adopted by Biaffine Parser with the only difference of the dropout strategy.
    It drops nodes in the LSTM layers (input and recurrent connections)
    and applies the same dropout mask at every recurrent timesteps.

    APIs are roughly the same as :class:`~torch.nn.LSTM` except that we only allows
    :class:`~torch.nn.utils.rnn.PackedSequence` as input.

    Args:
        input_size (int):
            The number of expected features in the input.
        hidden_size (int):
            The number of features in the hidden state `h`.
        num_layers (int):
            The number of recurrent layers. Default: 1.
        bidirectional (bool):
            If ``True``, becomes a bidirectional LSTM. Default: ``False``
        dropout (float):
            If non-zero, introduces a :class:`SharedDropout` layer on the outputs of each LSTM layer except the last layer.
            Default: 0.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = .0
    ) -> VariationalLSTM:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_directions = 1 + self.bidirectional

        self.f_cells = nn.ModuleList()
        if bidirectional:
            self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            input_size = hidden_size * self.num_directions

        self.reset_parameters()

    def __repr__(self):
        s = f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.bidirectional:
            s += f", bidirectional={self.bidirectional}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)

    def permute_hidden(
        self,
        hx: Tuple[torch.Tensor, torch.Tensor],
        permutation: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def layer_forward(
        self,
        x: List[torch.Tensor],
        hx: Tuple[torch.Tensor, torch.Tensor],
        cell: nn.LSTMCell,
        batch_sizes: List[int],
        reverse: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size])) for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(
        self,
        sequence: PackedSequence,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Args:
            sequence (~torch.nn.utils.rnn.PackedSequence):
                A packed variable length sequence.
            hx (~torch.Tensor, ~torch.Tensor):
                A tuple composed of two tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial hidden state
                for each element in the batch.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial cell state
                for each element in the batch.
                If `hx` is not provided, both `h` and `c` default to zero.
                Default: ``None``.

        Returns:
            ~torch.nn.utils.rnn.PackedSequence, (~torch.Tensor, ~torch.Tensor):
                The first is a packed variable length sequence.
                The second is a tuple of tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the hidden state for `t=seq_len`.
                Like output, the layers can be separated using ``h.view(num_layers, num_directions, batch_size, hidden_size)``
                and similarly for c.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the cell state for `t=seq_len`.
        """
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = x.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        c = c.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            x_i, (h_i, c_i) = self.layer_forward(x, (h[i, 0], c[i, 0]), self.f_cells[i], batch_sizes)
            if self.bidirectional:
                x_b, (h_b, c_b) = self.layer_forward(x, (h[i, 1], c[i, 1]), self.b_cells[i], batch_sizes, True)
                x_i = torch.cat((x_i, x_b), -1)
                h_i = torch.stack((h_i, h_b))
                c_i = torch.stack((c_i, c_b))
            x = x_i
            h_n.append(h_i)
            c_n.append(c_i)

        x = PackedSequence(x, sequence.batch_sizes, sequence.sorted_indices, sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)

        return x, hx
