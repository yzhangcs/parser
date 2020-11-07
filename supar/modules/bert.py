# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules.scalar_mix import ScalarMix
from torch.nn.utils.rnn import pad_sequence


class BertEmbedding(nn.Module):
    r"""
    A module that directly utilizes the pretrained models in `transformers`_ to produce BERT representations.
    While mainly tailored to provide input preparation and post-processing for the BERT model,
    it is also compatiable with other pretrained language models like XLNet, RoBERTa and ELECTRA, etc.
    Args:
        model (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of layers from the model to use.
            If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings.
            If 0, uses the size of the pretrained embedding model.
        stride (int):
            A sequence longer than the limited max length will be splitted into several small pieces
            with a window size of ``stride``. Default: 5.
        pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        dropout (float):
            The dropout ratio of BERT layers. Default: 0.
            This value will be passed into the :class:`ScalarMix` layer.
        requires_grad (bool):
            If ``True``, the model parameters will be updated together with the downstream task.
            Default: ``False``.
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, model, n_layers, n_out, stride=5, pad_index=0, dropout=0, requires_grad=False):
        super().__init__()

        from transformers import AutoConfig, AutoModel
        self.bert = AutoModel.from_pretrained(model, config=AutoConfig.from_pretrained(model, output_hidden_states=True))
        self.bert = self.bert.requires_grad_(requires_grad)

        self.model = model
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.stride = stride
        self.pad_index = pad_index
        self.dropout = dropout
        self.requires_grad = requires_grad
        self.max_len = self.bert.config.max_position_embeddings

        self.scalar_mix = ScalarMix(self.n_layers, dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_layers={self.n_layers}, n_out={self.n_out}, pad_index={self.pad_index}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, subwords):
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """
        batch_size, seq_len, fix_len = subwords.shape
        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_subwords]
        subwords = pad_sequence(subwords[mask].split(lens.tolist()), True)
        bert_mask = pad_sequence(mask[mask].split(lens.tolist()), True)

        # return the hidden states of all layers
        bert = self.bert(subwords[:, :self.max_len], attention_mask=bert_mask[:, :self.max_len].float())[-1]
        # [n_layers, batch_size, max_len, hidden_size]
        bert = bert[-self.n_layers:]
        # [batch_size, max_len, hidden_size]
        bert = self.scalar_mix(bert)
        # [batch_size, n_subwords, hidden_size]
        for i in range(self.stride, (subwords.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part = self.bert(subwords[:, i:i+self.max_len], attention_mask=bert_mask[:, i:i+self.max_len].float())[-1]
            bert = torch.cat((bert, self.scalar_mix(part[-self.n_layers:])[:, self.max_len-self.stride:]), 1)

        # [batch_size, n_subwords]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size)
        embed = embed.masked_scatter_(mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        embed = embed.sum(2) / bert_lens.unsqueeze(-1)
        embed = self.projection(embed)

        return embed
