# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import LSTM, MLP, BertEmbedding, CharLSTM
from supar.modules.affine import Biaffine, Triaffine
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.variational_inference import (LBPSemanticDependency,
                                                 MFVISemanticDependency)
from supar.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiaffineSemanticDependencyModel(nn.Module):
    r"""
    The implementation of Biaffine Semantic Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2018.
          `Simpler but More Accurate Semantic Dependency Parsing`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, needed if lemma embeddings are used. Default: ``None``.
        feat (str):
            Additional features to use，separated by commas.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: ``'tag,char,lemma'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_embed_proj (int):
            The size of linearly transformed word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        char_pad_index (int):
            The index of the padding token in the character vocabulary. Default: 0.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_edge (int):
            Edge MLP size. Default: 600.
        n_mlp_label  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of edge MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Simpler but More Accurate Semantic Dependency Parsing:
        https://www.aclweb.org/anthology/P18-2077/
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat='tag,char,lemma',
                 n_embed=100,
                 n_embed_proj=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 char_pad_index=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pad_index=0,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_edge=600,
                 n_mlp_label=600,
                 edge_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        self.embed_proj = nn.Linear(n_embed, n_embed_proj)

        self.n_input = n_embed + n_embed_proj
        if 'tag' in feat:
            self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                          embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'char' in feat:
            self.char_embed = CharLSTM(n_chars=n_chars,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=char_pad_index)
            self.n_input += n_feat_embed
        if 'lemma' in feat:
            self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                            embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'bert' in feat:
            self.bert_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=bert_pad_index,
                                            dropout=mix_dropout)
            self.n_input += self.bert_embed.n_out
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = LSTM(input_size=self.n_input,
                         hidden_size=n_lstm_hidden,
                         num_layers=n_lstm_layers,
                         bidirectional=True,
                         dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)

        # the MLP layers
        self.mlp_edge_d = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_edge, dropout=edge_mlp_dropout, activation=False)
        self.mlp_edge_h = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_edge, dropout=edge_mlp_dropout, activation=False)
        self.mlp_label_d = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_label, dropout=label_mlp_dropout, activation=False)
        self.mlp_label_h = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_label, dropout=label_mlp_dropout, activation=False)

        # the Biaffine layers
        self.edge_attn = Biaffine(n_in=n_mlp_edge, n_out=2, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_mlp_label, n_out=n_labels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.interpolation = interpolation
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
        return self

    def forward(self, words, feats):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size of indices is ``[batch_size, seq_len, fix_len]`` if feat is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len, 2]`` holds scores of all possible edges.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each edge.
        """

        _, seq_len = words.shape
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
            word_embed = torch.cat((word_embed, self.embed_proj(self.pretrained(words))), -1)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        edge_d = self.mlp_edge_d(x)
        edge_h = self.mlp_edge_h(x)
        label_d = self.mlp_label_d(x)
        label_h = self.mlp_label_h(x)

        # [batch_size, seq_len, seq_len, 2]
        s_egde = self.edge_attn(edge_d, edge_h).permute(0, 2, 3, 1)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_egde, s_label

    def loss(self, s_egde, s_label, edges, labels, mask):
        r"""
        Args:
            s_egde (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            edges (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard edges.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        edge_mask = edges.gt(0) & mask
        edge_loss = self.criterion(s_egde[mask], edges[mask])
        label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
        return self.interpolation * label_loss + (1 - self.interpolation) * edge_loss

    def decode(self, s_egde, s_label):
        r"""
        Args:
            s_egde (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted edges and labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        return s_egde.argmax(-1), s_label.argmax(-1)


class VISemanticDependencyModel(BiaffineSemanticDependencyModel):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference.

    References:
        - Xinyu Wang, Jingxian Huang and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, needed if lemma embeddings are used. Default: ``None``.
        feat (str):
            Additional features to use，separated by commas.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: ``'tag,char,lemma'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_embed_proj (int):
            The size of linearly transformed word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        char_pad_index (int):
            The index of the padding token in the character vocabulary. Default: 0.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_un (int):
            Unary factor MLP size. Default: 600.
        n_mlp_bin (int):
            Binary factor MLP size. Default: 150.
        n_mlp_label  (int):
            Label MLP size. Default: 600.
        un_mlp_dropout (float):
            The dropout ratio of unary factor MLP layers. Default: .25.
        bin_mlp_dropout (float):
            The dropout ratio of binary factor MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        inference (str):
            Approximate inference methods. Default: 'mfvi'.
        max_iter (int):
            Max iteration times for Variational Inference. Default: 3.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat='tag,char,lemma',
                 n_embed=100,
                 n_embed_proj=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 char_pad_index=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pad_index=0,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_un=600,
                 n_mlp_bin=150,
                 n_mlp_label=600,
                 un_mlp_dropout=.25,
                 bin_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 inference='mfvi',
                 max_iter=3,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        self.embed_proj = nn.Linear(n_embed, n_embed_proj)

        self.n_input = n_embed + n_embed_proj
        if 'tag' in feat:
            self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                          embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'char' in feat:
            self.char_embed = CharLSTM(n_chars=n_chars,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=char_pad_index)
            self.n_input += n_feat_embed
        if 'lemma' in feat:
            self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                            embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'bert' in feat:
            self.bert_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=bert_pad_index,
                                            dropout=mix_dropout)
            self.n_input += self.bert_embed.n_out
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = LSTM(input_size=self.n_input,
                         hidden_size=n_lstm_hidden,
                         num_layers=n_lstm_layers,
                         bidirectional=True,
                         dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)

        # the MLP layers
        self.mlp_un_d = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_un, dropout=un_mlp_dropout, activation=False)
        self.mlp_un_h = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_un, dropout=un_mlp_dropout, activation=False)
        self.mlp_bin_d = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_bin, dropout=bin_mlp_dropout, activation=False)
        self.mlp_bin_h = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_bin, dropout=bin_mlp_dropout, activation=False)
        self.mlp_bin_g = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_bin, dropout=bin_mlp_dropout, activation=False)
        self.mlp_label_d = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_label, dropout=label_mlp_dropout, activation=False)
        self.mlp_label_h = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_label, dropout=label_mlp_dropout, activation=False)

        # the affine layers
        self.edge_attn = Biaffine(n_in=n_mlp_un, bias_x=True, bias_y=True)
        self.sib_attn = Triaffine(n_in=n_mlp_bin, bias_x=True, bias_y=True)
        self.cop_attn = Triaffine(n_in=n_mlp_bin, bias_x=True, bias_y=True)
        self.grd_attn = Triaffine(n_in=n_mlp_bin, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_mlp_label, n_out=n_labels, bias_x=True, bias_y=True)
        self.vi = (MFVISemanticDependency if inference == 'mfvi' else LBPSemanticDependency)(max_iter)
        self.criterion = nn.CrossEntropyLoss()
        self.interpolation = interpolation
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
        return self

    def forward(self, words, feats):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size of indices is ``[batch_size, seq_len, fix_len]`` if feat is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible edges of shape ``[batch_size, seq_len, seq_len]``,
                dependent-head-sibling triples of shape ``[batch_size, seq_len, seq_len, seq_len]`` and
                all possible labels on each edge of shape ``[batch_size, seq_len, seq_len, n_labels]``.
        """

        _, seq_len = words.shape
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
            word_embed = torch.cat((word_embed, self.embed_proj(self.pretrained(words))), -1)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        un_d = self.mlp_un_d(x)
        un_h = self.mlp_un_h(x)
        bin_d = self.mlp_bin_d(x)
        bin_h = self.mlp_bin_h(x)
        bin_g = self.mlp_bin_g(x)
        label_h = self.mlp_label_h(x)
        label_d = self.mlp_label_d(x)
        label_h = self.mlp_label_h(x)

        # [batch_size, seq_len, seq_len]
        s_egde = self.edge_attn(un_d, un_h)
        # [batch_size, seq_len, seq_len, n_labels]
        s_sib = self.sib_attn(bin_d, bin_d, bin_h).triu_()
        s_sib = (s_sib + s_sib.transpose(-1, -2)).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_labels]
        s_cop = self.cop_attn(bin_h, bin_d, bin_h).permute(0, 3, 1, 2).triu_()
        s_cop = s_cop + s_cop.transpose(-1, -2)
        # [batch_size, seq_len, seq_len, n_labels]
        s_grd = self.grd_attn(bin_g, bin_d, bin_h).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_egde, s_sib, s_cop, s_grd, s_label

    def loss(self, s_egde, s_sib, s_cop, s_grd, s_label, edges, labels, mask):
        r"""
        Args:
            s_egde (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_cop (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-coparent triples.
            s_grd (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-grandparent triples.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            edges (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard edges.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        edge_mask = edges.gt(0) & mask
        edge_loss, marginals = self.vi((s_egde, s_sib, s_cop, s_grd), mask, edges)
        label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
        loss = self.interpolation * label_loss + (1 - self.interpolation) * edge_loss
        return loss, marginals

    def decode(self, s_egde, s_label):
        r"""
        Args:
            s_egde (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted edges and labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        return s_egde.argmax(-1), s_label.argmax(-1)
