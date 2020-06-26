# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.treecrf import CRFConstituency
from supar.utils.alg import cky
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRFConstituencyModel(nn.Module):

    def __init__(self, args):
        super(CRFConstituencyModel, self).__init__()

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
        self.mlp_span_l = MLP(n_in=args.n_lstm_hidden*2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_span_r = MLP(n_in=args.n_lstm_hidden*2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_label_l = MLP(n_in=args.n_lstm_hidden*2,
                               n_out=args.n_mlp_label,
                               dropout=args.mlp_dropout)
        self.mlp_label_r = MLP(n_in=args.n_lstm_hidden*2,
                               n_out=args.n_mlp_label,
                               dropout=args.mlp_dropout)

        # the Biaffine layers
        self.span_attn = Biaffine(n_in=args.n_mlp_span,
                                  bias_x=True,
                                  bias_y=False)
        self.label_attn = Biaffine(n_in=args.n_mlp_label,
                                   n_out=args.n_labels,
                                   bias_x=True,
                                   bias_y=True)
        self.crf = CRFConstituency()
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

    def loss(self, s_span, s_label, spans, labels, mask):
        span_mask = spans & mask
        span_loss, span_probs = self.crf(s_span, mask, spans, self.args.mbr)
        label_loss = self.criterion(s_label[span_mask], labels[span_mask])
        loss = span_loss + label_loss

        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        span_preds = cky(s_span, mask)
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j in spans]
                for spans, labels in zip(span_preds, label_preds)]
