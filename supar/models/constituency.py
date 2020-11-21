# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import LSTM, MLP, BertEmbedding, Biaffine, CharLSTM
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.treecrf import CRFConstituency
from supar.utils import Config
from supar.utils.alg import cky
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRFConstituencyModel(nn.Module):
    r"""
    The implementation of CRF Constituency Parser,
    also called FANCY (abbr. of Fast and Accurate Neural Crf constituencY) Parser.

    References:
        - Yu Zhang, Houquan Zhou and Zhenghua Li. 2020.
          `Fast and Accurate Neural CRF Constituency Parsing`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_feats (int):
            The size of the feat vocabulary.
        n_labels (int):
            The number of labels.
        feat (str):
            Specifies which type of additional feature to use: ``'char'`` | ``'bert'`` | ``'tag'``.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            ``'tag'``: POS tag embeddings.
            Default: 'char'.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_span (int):
            Span MLP size. Default: 500.
        n_mlp_label  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Fast and Accurate Neural CRF Constituency Parsing:
        https://www.ijcai.org/Proceedings/2020/560/
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_feats,
                 n_labels,
                 feat='char',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if feat == 'char':
            self.feat_embed = CharLSTM(n_chars=n_feats,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=feat_pad_index)
        elif feat == 'bert':
            self.feat_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=feat_pad_index,
                                            dropout=mix_dropout)
            self.n_feat_embed = self.feat_embed.n_out
        elif feat == 'tag':
            self.feat_embed = nn.Embedding(num_embeddings=n_feats,
                                           embedding_dim=n_feat_embed)
        else:
            raise RuntimeError("The feat type should be in ['char', 'bert', 'tag'].")
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = LSTM(input_size=n_embed+n_feat_embed,
                         hidden_size=n_lstm_hidden,
                         num_layers=n_lstm_layers,
                         bidirectional=True,
                         dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)

        # the MLP layers
        self.mlp_span_l = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_span_r = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_label_l = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_label, dropout=mlp_dropout)
        self.mlp_label_r = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_label, dropout=mlp_dropout)

        # the Biaffine layers
        self.span_attn = Biaffine(n_in=n_mlp_span, bias_x=True, bias_y=False)
        self.label_attn = Biaffine(n_in=n_mlp_label, n_out=n_labels, bias_x=True, bias_y=True)
        self.crf = CRFConstituency()
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def forward(self, words, feats):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (~torch.LongTensor):
                Feat indices.
                If feat is ``'char'`` or ``'bert'``, the size of feats should be ``[batch_size, seq_len, fix_len]``
                if ``'tag'``, the size is ``[batch_size, seq_len]``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible spans.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each span.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        x_f, x_b = x.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        # apply MLPs to the BiLSTM output states
        span_l = self.mlp_span_l(x)
        span_r = self.mlp_span_r(x)
        label_l = self.mlp_label_l(x)
        label_r = self.mlp_label_r(x)

        # [batch_size, seq_len, seq_len]
        s_span = self.span_attn(span_l, span_r)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return s_span, s_label

    def loss(self, s_span, s_label, charts, mask, mbr=True):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all spans
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each span.
            charts (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels, in which positions without labels are filled with -1.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                original span scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """

        span_mask = charts.ge(0) & mask
        span_loss, span_probs = self.crf(s_span, mask, span_mask, mbr)
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = span_loss + label_loss

        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all spans.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each span.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            list[list[tuple]]:
                Sequences of factorized labeled trees traversed in pre-order.
        """

        span_preds = cky(s_span, mask)
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j in spans] for spans, labels in zip(span_preds, label_preds)]
