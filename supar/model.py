# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from supar.modules import (CharLSTM, ELMoEmbedding, IndependentDropout,
                           SharedDropout, TransformerEmbedding,
                           TransformerWordEmbedding, VariationalLSTM)
from supar.modules.transformer import (TransformerEncoder,
                                       TransformerEncoderLayer)
from supar.utils import Config


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
                 elmo_bos_eos=(True, True),
                 elmo_dropout=0.5,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 encoder_dropout=.33,
                 pad_index=0,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())

        if encoder == 'lstm':
            self.word_embed = nn.Embedding(num_embeddings=self.args.n_words,
                                           embedding_dim=self.args.n_embed)

            n_input = self.args.n_embed
            if self.args.n_pretrained != self.args.n_embed:
                n_input += self.args.n_pretrained
            if 'tag' in self.args.feat:
                self.tag_embed = nn.Embedding(num_embeddings=self.args.n_tags,
                                              embedding_dim=self.args.n_feat_embed)
                n_input += self.args.n_feat_embed
            if 'char' in self.args.feat:
                self.char_embed = CharLSTM(n_chars=self.args.n_chars,
                                           n_embed=self.args.n_char_embed,
                                           n_hidden=self.args.n_char_hidden,
                                           n_out=self.args.n_feat_embed,
                                           pad_index=self.args.char_pad_index,
                                           dropout=self.args.char_dropout)
                n_input += self.args.n_feat_embed
            if 'lemma' in self.args.feat:
                self.lemma_embed = nn.Embedding(num_embeddings=self.args.n_lemmas,
                                                embedding_dim=self.args.n_feat_embed)
                n_input += self.args.n_feat_embed
            if 'elmo' in self.args.feat:
                self.elmo_embed = ELMoEmbedding(n_out=self.args.n_plm_embed,
                                                bos_eos=self.args.elmo_bos_eos,
                                                dropout=self.args.elmo_dropout,
                                                finetune=self.args.finetune)
                n_input += self.elmo_embed.n_out
            if 'bert' in self.args.feat:
                self.bert_embed = TransformerEmbedding(model=self.args.bert,
                                                       n_layers=self.args.n_bert_layers,
                                                       n_out=self.args.n_plm_embed,
                                                       pooling=self.args.bert_pooling,
                                                       pad_index=self.args.bert_pad_index,
                                                       mix_dropout=self.args.mix_dropout,
                                                       finetune=self.args.finetune)
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=self.args.embed_dropout)
            self.encoder = VariationalLSTM(input_size=n_input,
                                           hidden_size=self.args.n_encoder_hidden//2,
                                           num_layers=self.args.n_encoder_layers,
                                           bidirectional=True,
                                           dropout=self.args.encoder_dropout)
            self.encoder_dropout = SharedDropout(p=self.args.encoder_dropout)
        elif encoder == 'transformer':
            self.word_embed = TransformerWordEmbedding(n_vocab=self.args.n_words,
                                                       n_embed=self.args.n_embed,
                                                       pos=self.args.pos,
                                                       pad_index=self.args.pad_index)
            self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)
            self.encoder = TransformerEncoder(layer=TransformerEncoderLayer(n_heads=self.args.n_encoder_heads,
                                                                            n_model=self.args.n_encoder_hidden,
                                                                            n_inner=self.args.n_encoder_inner,
                                                                            attn_dropout=self.args.encoder_attn_dropout,
                                                                            ffn_dropout=self.args.encoder_ffn_dropout,
                                                                            dropout=self.args.encoder_dropout),
                                              n_layers=self.args.n_encoder_layers,
                                              n_model=self.args.n_encoder_hidden)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
        elif encoder == 'bert':
            self.encoder = TransformerEmbedding(model=self.args.bert,
                                                n_layers=self.args.n_bert_layers,
                                                pooling=self.args.bert_pooling,
                                                pad_index=self.args.pad_index,
                                                mix_dropout=self.args.mix_dropout,
                                                finetune=True)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
            self.args.n_encoder_hidden = self.encoder.n_out

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self, words, feats=None):
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

        feat_embed = []
        if 'tag' in self.args.feat:
            feat_embed.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embed.append(self.char_embed(feats.pop(0)))
        if 'elmo' in self.args.feat:
            feat_embed.append(self.elmo_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embed.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embed.append(self.lemma_embed(feats.pop(0)))
        if isinstance(self.embed_dropout, IndependentDropout):
            if len(feat_embed) == 0:
                raise RuntimeError(f"`feat` is not allowed to be empty, which is {self.args.feat} now")
            embed = torch.cat(self.embed_dropout(word_embed, torch.cat(feat_embed, -1)), -1)
        else:
            embed = word_embed
            if len(feat_embed) > 0:
                embed = torch.cat((embed, torch.cat(feat_embed, -1)), -1)
            embed = self.embed_dropout(embed)
        return embed

    def encode(self, words, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(words, feats), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        elif self.args.encoder == 'transformer':
            x = self.encoder(self.embed(words, feats), words.ne(self.args.pad_index))
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError
