# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CharLSTM(nn.Module):
    r"""
    CharLSTM aims to generate character-level embeddings for tokens.
    It summerizes the information of characters in each token to an embedding using a LSTM layer.

    Args:
        n_char (int):
            The number of characters.
        n_embed (int):
            The size of each embedding vector as input to LSTM.
        n_out (int):
            The size of each output vector.
        pad_index (int):
            The index of the padding token in the vocabulary. Default: 0.
    """

    def __init__(self, n_chars, n_embed, n_out, pad_index=0):
        super().__init__()

        self.n_chars = n_chars
        self.n_embed = n_embed
        self.n_out = n_out
        self.pad_index = pad_index

        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=n_chars,
                                  embedding_dim=n_embed)
        # the lstm layer
        self.lstm = nn.LSTM(input_size=n_embed,
                            hidden_size=n_out//2,
                            batch_first=True,
                            bidirectional=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_chars}, {self.n_embed}, n_out={self.n_out}, pad_index={self.pad_index})"

    def forward(self, x):
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
        lens = mask.sum(-1).cpu()
        char_mask = lens.gt(0)

        # [n, fix_len, n_embed]
        x = self.embed(x[char_mask])
        x = pack_padded_sequence(x, lens[char_mask], True, False)
        x, (h, _) = self.lstm(x)
        # [n, fix_len, n_out]
        h = torch.cat(torch.unbind(h), -1)
        # [batch_size, seq_len, n_out]
        embed = h.new_zeros(*lens.shape, self.n_out)
        embed = embed.masked_scatter_(char_mask.unsqueeze(-1), h)

        return embed
