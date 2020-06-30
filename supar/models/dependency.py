# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import (MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM,
                           MatrixTree, Triaffine)
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.treecrf import CRF2oDependency, CRFDependency
from supar.utils.alg import eisner, mst
from supar.utils.transform import CoNLL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiaffineParserModel(nn.Module):

    r'''The implementation of
    "Deep Biaffine Attention for Neural Dependency Parsing":
    https://arxiv.org/abs/1611.01734.
    '''

    def __init__(self, args):
        super(BiaffineParserModel, self).__init__()

        self.args = args
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=args.n_words,
                                       embedding_dim=args.n_embed)
        if args.feat == 'char':
            self.feat_embed = CharLSTM(n_chars=args.n_feats,
                                       n_embed=args.n_char_embed,
                                       n_out=args.n_feat_embed,
                                       pad_index=args.feat_pad_index)
        elif args.feat == 'bert':
            self.feat_embed = BertEmbedding(model=args.bert,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.n_feat_embed,
                                            pad_index=args.feat_pad_index,
                                            dropout=args.mix_dropout)
            self.args.n_feat_embed = self.feat_embed.n_out
        else:
            self.feat_embed = nn.Embedding(num_embeddings=args.n_feats,
                                           embedding_dim=args.n_feat_embed)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=args.n_embed+args.n_feat_embed,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_rel,
                             dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_rel,
                             dropout=args.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
                                 n_out=args.n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def forward(self, words, feats):
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

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask):
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask):
        lens = mask.sum(1)
        # prevent self-loops
        s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[:i+1], self.args.proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if self.args.tree and any(bad):
            alg = mst if self.args.proj else eisner
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds


class MSTDependencyModel(BiaffineParserModel):

    def __init__(self, args):
        super(MSTDependencyModel, self).__init__(args)

        self.matrix_tree = MatrixTree()

    def loss(self, s_arc, s_rel, arcs, rels, mask):
        batch_size, seq_len = mask.shape
        arc_loss, arc_probs = self.matrix_tree(s_arc, mask, arcs)
        # -1 denotes un-annotated arcs
        if self.args.partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, arc_probs


class CRFDependencyModel(BiaffineParserModel):

    r'''The implementation of
    "Efficient Second-Order TreeCRF for Neural Dependency Parsing":
    https://www.aclweb.org/anthology/2020.acl-main.302/.
    '''

    def __init__(self, args):
        super(CRFDependencyModel, self).__init__(args)

        self.crf = CRFDependency()

    def loss(self, s_arc, s_rel, arcs, rels, mask):
        batch_size, seq_len = mask.shape
        arc_loss, arc_probs = self.crf(s_arc, mask, arcs,
                                       self.args.mbr, self.args.partial)
        # -1 denotes un-annotated arcs
        if self.args.partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, arc_probs


class CRF2oDependencyModel(BiaffineParserModel):

    r'''The implementation of
    "Efficient Second-Order TreeCRF for Neural Dependency Parsing":
    https://www.aclweb.org/anthology/2020.acl-main.302.
    '''

    def __init__(self, args):
        super(CRF2oDependencyModel, self).__init__(args)

        self.mlp_sib_s = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_sib,
                             dropout=args.mlp_dropout)
        self.mlp_sib_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_sib,
                             dropout=args.mlp_dropout)
        self.mlp_sib_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_sib,
                             dropout=args.mlp_dropout)

        self.sib_attn = Triaffine(n_in=args.n_mlp_sib,
                                  bias_x=True,
                                  bias_y=True)
        self.crf = CRF2oDependency()

    def forward(self, words, feats):
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

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        sib_s = self.mlp_sib_s(x)
        sib_d = self.mlp_sib_d(x)
        sib_h = self.mlp_sib_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, seq_len]
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_sib, s_rel

    def loss(self, s_arc, s_sib, s_rel, arcs, sibs, rels, mask):
        batch_size, seq_len = mask.shape
        scores, target = (s_arc, s_sib), (arcs, sibs)
        arc_loss, arc_probs = self.crf(scores, mask, target,
                                       self.args.mbr, self.args.partial)
        # -1 denotes un-annotated arcs
        if self.args.partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, arc_probs
