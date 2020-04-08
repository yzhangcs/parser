# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertModel

from .scalar_mix import ScalarMix


class BertEmbedding(nn.Module):

    def __init__(self, model, n_layers, n_out, requires_grad=False):
        super(BertEmbedding, self).__init__()

        self.bert = BertModel.from_pretrained(model, output_hidden_states=True)
        self.bert = self.bert.requires_grad_(requires_grad)
        self.n_layers = n_layers
        self.n_out = n_out
        self.requires_grad = requires_grad
        self.hidden_size = self.bert.config.hidden_size

        self.scalar_mix = ScalarMix(n_layers)
        if self.hidden_size != n_out:
            self.projection = nn.Linear(self.hidden_size, n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        s += ')'

        return s

    def forward(self, subwords, bert_lens, bert_mask):
        batch_size, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)

        if not self.requires_grad:
            self.bert.eval()
        _, _, bert = self.bert(subwords, attention_mask=bert_mask)
        bert = bert[-self.n_layers:]
        bert = self.scalar_mix(bert)
        bert = bert[bert_mask].split(bert_lens[mask].tolist())
        bert = torch.stack([i.mean(0) for i in bert])
        bert_embed = bert.new_zeros(batch_size, seq_len, self.hidden_size)
        bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(-1), bert)
        if hasattr(self, 'projection'):
            bert_embed = self.projection(bert_embed)

        return bert_embed
