# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import ElectraForMaskedLM

from .scalar_mix import ScalarMix


class ElectraEmbedding(nn.Module):

    def __init__(self, model, n_layers, n_out, requires_grad=False):
        super(ElectraEmbedding, self).__init__()

        self.electra = ElectraForMaskedLM.from_pretrained(model, output_hidden_states=True)
        self.electra = self.electra.requires_grad_(requires_grad)
        self.n_layers = n_layers
        self.n_out = n_out
        self.requires_grad = requires_grad
        self.hidden_size = self.electra.config.hidden_size

        self.scalar_mix = ScalarMix(n_layers)
        self.projection = nn.Linear(self.hidden_size, n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        s += ')'

        return s

    def forward(self, subwords, electra_lens, electra_mask):
        batch_size, seq_len = electra_lens.shape
        mask = electra_lens.gt(0)

        if not self.requires_grad:
            self.electra.eval()
        _, electra = self.electra(subwords, attention_mask=electra_mask)
        electra = electra[-self.n_layers:]
        electra = self.scalar_mix(electra)
        electra = electra[electra_mask].split(electra_lens[mask].tolist())
        electra = torch.stack([i.mean(0) for i in electra])
        electra_embed = electra.new_zeros(batch_size, seq_len, self.hidden_size)
        electra_embed = electra_embed.masked_scatter_(mask.unsqueeze(-1), electra)
        electra_embed = self.projection(electra_embed)

        return electra_embed
