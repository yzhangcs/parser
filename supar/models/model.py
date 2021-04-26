# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import (CharLSTM, IndependentDropout, SharedDropout,
                           TransformerEmbedding, VariationalLSTM)
from supar.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):

    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=False,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 pad_index=0,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())

        if encoder != 'bert':
            self.word_embed = nn.Embedding(num_embeddings=n_words,
                                           embedding_dim=n_embed)

            n_input = n_embed
            if n_pretrained != n_embed:
                n_input += n_pretrained
            if 'tag' in feat:
                self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                              embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'char' in feat:
                self.char_embed = CharLSTM(n_chars=n_chars,
                                           n_embed=n_char_embed,
                                           n_hidden=n_char_hidden,
                                           n_out=n_feat_embed,
                                           pad_index=char_pad_index,
                                           dropout=char_dropout)
                n_input += n_feat_embed
            if 'lemma' in feat:
                self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                                embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'bert' in feat:
                self.bert_embed = TransformerEmbedding(model=bert,
                                                       n_layers=n_bert_layers,
                                                       n_out=n_feat_embed,
                                                       pooling=bert_pooling,
                                                       pad_index=bert_pad_index,
                                                       dropout=mix_dropout,
                                                       requires_grad=(not freeze))
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=embed_dropout)
        if encoder == 'lstm':
            self.encoder = VariationalLSTM(input_size=n_input,
                                           hidden_size=n_lstm_hidden,
                                           num_layers=n_lstm_layers,
                                           bidirectional=True,
                                           dropout=encoder_dropout)
            self.encoder_dropout = SharedDropout(p=encoder_dropout)
            self.args.n_hidden = n_lstm_hidden * 2
        else:
            self.encoder = TransformerEmbedding(model=bert,
                                                n_layers=n_bert_layers,
                                                pooling=bert_pooling,
                                                pad_index=pad_index,
                                                dropout=mix_dropout,
                                                requires_grad=True)
            self.encoder_dropout = nn.Dropout(p=encoder_dropout)
            self.args.n_hidden = self.encoder.n_out

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed.to(self.args.device))
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained).to(self.args.device)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self, words, feats):
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            if self.args.n_embed == self.args.n_pretrained:
                word_embed += pretrained
            else:
                word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)

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

        return embed

    def encode(self, words, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(words, feats), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError
