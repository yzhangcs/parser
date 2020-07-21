# -*- coding: utf-8 -*-

import torch.nn as nn
from supar.modules.scalar_mix import ScalarMix
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel


class BertEmbedding(nn.Module):
    """
    A module used for outputting BERT representations.
    This module directly utilizes the pretrained models like `bert-base-cased` in `transformers`
    to produce the representations.

    While mainly tailored to provide input preparation and post-processing for the BERT model,
    this module is not only limited to BERT, but also compatiable with other pretrained language models
    like XLNet, RoBERTa and ELECTRA, etc.

    Args:
        model (str):
            Path or name of the pretrained model.
        n_layers (int):
            Number of layers from the model to use.
            If 0, use all layers.
        n_out (int):
            The requested size of the embeddings.
            If 0, use the size of the pretrained embedding model.
        pad_index (int, default: 0):
            The index of the padding token in the BERT vocabulary.
        dropout (float, default: 0):
            Dropout ratio of BERT layers.
            This value will be passed into the ScalarMix layer.
        requires_grad (bool, default: False):
            If True, the parameters of the pretrained model won't be freezed,
            and will be updated together with the downstream task.
    """

    def __init__(self, model, n_layers, n_out, pad_index=0, dropout=0,
                 requires_grad=False,
                 use_hidden_states=True, output_attentions=False, attention_layer=8):
        super().__init__()

        config = AutoConfig.from_pretrained(model, output_hidden_states=True,
                                            output_attentions=output_attentions)
        self.model = model
        self.bert = AutoModel.from_pretrained(model, config=config)
        self.bert = self.bert.requires_grad_(requires_grad)
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.pad_index = pad_index
        self.requires_grad = requires_grad
        self.output_attentions = output_attentions
        self.attention_layer = attention_layer
        self.head = 0

        self.scalar_mix = ScalarMix(self.n_layers, dropout)
        self.projection = None
        if self.hidden_size != self.n_out:
            self.projection = nn.Linear(self.hidden_size, self.n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + f"({self.model}, "
        s += f"n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"pad_index={self.pad_index}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        s += ')'

        return s

    def forward(self, subwords):
        """
        Args:
            subwords (Tensor): [batch_size, seq_len, fix_len]

        Returns:
            embed (Tensor): [batch_size, seq_len, n_out]
        """
        batch_size, seq_len, fix_len = subwords.shape
        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))

        # [batch_size, n_subwords]
        subwords = pad_sequence(subwords[mask].split(lens.tolist()), True)
        bert_mask = pad_sequence(mask[mask].split(lens.tolist()), True)
        # return the hidden states of all layers
        outputs = self.bert(subwords, attention_mask=bert_mask.float())
        bert = outputs[-2] if self.output_attentions else outputs[-1]
        # [n_layers, batch_size, n_subwords, hidden_size]
        bert = bert[-self.n_layers:]
        # [batch_size, n_subwords, hidden_size]
        bert = self.scalar_mix(bert)
        # [batch_size, n_subwords]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size)
        embed = embed.masked_scatter_(mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        embed = embed.sum(2) / bert_lens.unsqueeze(-1) # sum wordpieces
        seq_attn = None
        if self.output_attentions:
            # (a list of layers) = [ [batch, num_heads, sent_len, sent_len] ]
            attns = outputs[-1]
            # [batch, n_subwords, n_subwords]
            attn = attns[self.attention_layer][:,self.head,:,:] # layer 9 represents syntax
            # squeeze out multiword tokens
            mask2 = ~mask
            mask2[:,:,0] = True # keep first column
            sub_masks = pad_sequence(mask2[mask].split(lens.tolist()), True)
            seq_mask = torch.einsum('bi,bj->bij', sub_masks, sub_masks) # outer product
            seq_lens = seq_mask.sum((1,2))
            # [batch_size, seq_len, seq_len]
            sub_attn = attn[seq_mask].split(seq_lens.tolist())
            # fill a tensor [batch_size, seq_len, seq_len]
            seq_attn = attn.new_zeros(batch_size, seq_len, seq_len)
            for i, attn_i in enumerate(sub_attn):
                size = sub_masks[i].sum(0)
                attn_i = attn_i.view(size, size)
                size = min(size, self.output_attentions)
                seq_attn[i,:size,:size] = attn_i
        if self.projection:
            embed = self.projection(embed)

        return embed, seq_attn
