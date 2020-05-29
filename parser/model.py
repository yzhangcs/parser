# -*- coding: utf-8 -*-

from parser.modules import MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM
from parser.modules.dropout import IndependentDropout, SharedDropout
from parser.utils.alg import eisner
from parser.utils.fn import istree

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

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
            self.feat_embed = BertEmbedding(model=args.bert_model,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.n_feat_embed,
                                            pad_index=args.feat_pad_index)
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
        lens = mask.sum(dim=1)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        if self.args.feat == 'char':
            feat_embed = self.feat_embed(feats[mask])
            feat_embed = pad_sequence(feat_embed.split(lens.tolist()), True)
        else:
            feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), dim=-1)

        x = pack_padded_sequence(embed, lens, True, False)
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

    def get_loss(self, s_arc, s_rel, arcs, rels, mask):
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
        bad = [not istree(seq[:i+1], self.args.proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if self.args.tree and any(bad):
            arc_preds[bad] = eisner(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)
