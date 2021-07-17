# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules.scalar_mix import ScalarMix
from supar.utils.fn import pad


class TransformerEmbedding(nn.Module):
    r"""
    Bidirectional transformer embeddings of words from various transformer architectures :cite:`devlin-etal-2019-bert`.

    Args:
        model (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of BERT layers to use. If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        stride (int):
            A sequence longer than max length will be splitted into several small pieces
            with a window size of ``stride``. Default: 10.
        pooling (str):
            Pooling way to get from token piece embeddings to token embedding.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        pad_index (int):
            The index of the padding token in BERT vocabulary. Default: 0.
        dropout (float):
            The dropout ratio of BERT layers. Default: 0. This value will be passed into the :class:`ScalarMix` layer.
        requires_grad (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, model, n_layers, n_out=0, stride=256, pooling='mean', pad_index=0, dropout=0, requires_grad=False):
        super().__init__()

        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.bert = AutoModel.from_pretrained(model, config=AutoConfig.from_pretrained(model, output_hidden_states=True))
        self.bert = self.bert.requires_grad_(requires_grad)

        self.model = model
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.stride = stride
        self.pooling = pooling
        self.pad_index = pad_index
        self.dropout = dropout
        self.requires_grad = requires_grad
        self.max_len = int(max(0, self.bert.config.max_position_embeddings) or 1e12) - 2

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.scalar_mix = ScalarMix(self.n_layers, dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"stride={self.stride}, pooling={self.pooling}, pad_index={self.pad_index}"
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

        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_subwords]
        subwords = pad(subwords[mask].split(lens.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        bert_mask = pad(mask[mask].split(lens.tolist()), 0, padding_side=self.tokenizer.padding_side)

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

        # [batch_size, seq_len]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size).masked_scatter_(mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            embed = embed[:, :, 0]
        elif self.pooling == 'last':
            embed = embed.gather(2, (bert_lens-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        else:
            embed = embed.sum(2) / bert_lens.unsqueeze(-1)
        embed = self.projection(embed)

        return embed


class ELMoEmbedding(nn.Module):
    r"""
    Contextual word embeddings using word-level bidirectional LM :cite:`peters-etal-2018-deep`.

    Args:
        model (str):
            The name of the pretrained ELMo registered in `OPTION` and `WEIGHT`. Default: ``'original_5b'``.
        bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of sentence outputs.
            Default: ``(True, True)``.
        n_out (int):
            The requested size of the embeddings. If 0, uses the default size of ELMo outputs. Default: 0.
        dropout (float):
            The dropout to be applied to the ELMo representations. Default: 0.5.
        requires_grad (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.
    """

    OPTION = {
        'small': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json',  # noqa
        'medium': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json',  # noqa
        'original': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',  # noqa
        'original_5b': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',  # noqa
    }
    WEIGHT = {
        'small': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',  # noqa
        'medium': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5',  # noqa
        'original': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',  # noqa
        'original_5b': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',  # noqa
    }

    def __init__(self, model='original_5b', bos_eos=(True, True), n_out=0, dropout=0.5, requires_grad=False):
        super().__init__()

        from allennlp.modules import Elmo

        self.elmo = Elmo(options_file=self.OPTION[model],
                         weight_file=self.WEIGHT[model],
                         num_output_representations=1,
                         dropout=dropout,
                         requires_grad=requires_grad,
                         keep_sentence_boundaries=True)

        self.model = model
        self.bos_eos = bos_eos
        self.hidden_size = self.elmo.get_output_dim()
        self.n_out = n_out or self.hidden_size
        self.dropout = dropout
        self.requires_grad = requires_grad

        self.scalar_mix = ScalarMix(self.elmo._elmo_lstm.num_layers)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_out={self.n_out}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, chars):
        r"""
        Args:
            chars (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.

        Returns:
            ~torch.Tensor:
                ELMo embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        x = self.elmo._elmo_lstm(chars)['activations']
        x = self.scalar_mix(x)
        x = self.projection(x)
        if not self.bos_eos[0]:
            x = x[:, 1:]
        if not self.bos_eos[1]:
            x = x[:, :-1]
        return x
