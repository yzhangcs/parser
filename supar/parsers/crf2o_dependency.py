# -*- coding: utf-8 -*-


import os

import torch
import torch.nn as nn
from supar.models import CRF2oDependencyModel
from supar.parsers.biaffine_dependency import BiaffineDependencyParser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import Field, SubwordField
from supar.utils.logging import logger, progress_bar
from supar.utils.metric import AttachmentMetric
from supar.utils.transform import CoNLL


class CRF2oDependencyParser(BiaffineDependencyParser):

    NAME = 'crf2o-dependency'
    MODEL = CRF2oDependencyModel

    def __init__(self, *args, **kwargs):
        super(CRF2oDependencyParser, self).__init__(*args, **kwargs)

    def _train(self, loader):
        self.model.train()

        metric = AttachmentMetric()
        progress = progress_bar(loader)

        for words, feats, arcs, sibs, rels in progress:
            self.optimizer.zero_grad()

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_sib, s_rel = self.model(words, feats)
            loss, s_arc = self.model.loss(s_arc, s_sib, s_rel,
                                          arcs, sibs, rels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
            progress.set_postfix_str(f"lr: {self.scheduler.get_lr()[0]:.4e} - "
                                     f"loss: {loss:.4f} - "
                                     f"{metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, AttachmentMetric()

        for words, feats, arcs, sibs, rels in loader:
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_sib, s_rel = self.model(words, feats)
            loss, s_arc = self.model.loss(s_arc, s_sib, s_rel,
                                          arcs, sibs, rels, mask)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        arcs, rels, probs = [], [], []
        for words, feats in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_sib, s_rel = self.model(words, feats)
            if self.args.mbr:
                s_arc = self.model.crf((s_arc, s_rel), mask, mbr=True)
            arc_preds, rel_preds = self.model.decode(s_arc, s_sib, s_rel, mask)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                s_arc = s_arc if self.args.mbr else s_arc.softmax(-1)
                arc_probs = s_arc.gather(-1, arc_preds.unsqueeze(-1))
                probs.extend(arc_probs.squeeze(-1)[mask].split(lens))
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = [seq.tolist() for seq in probs]

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Build the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        else:
            FEAT = Field('tags', bos=bos)
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        SIB = Field('sibs', bos=bos, use_vocab=False, fn=CoNLL.get_sibs)
        REL = Field('rels', bos=bos)
        if args.feat in ('char', 'bert'):
            transform = CoNLL(FORM=(WORD, FEAT), HEAD=(ARC, SIB), DEPREL=REL)
        else:
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=(ARC, SIB), DEPREL=REL)

        train = Dataset(transform, args.train)
        embed = None
        embed = None
        if args.embed:
            embed = Embedding.load(args.embed, args.unk)
        WORD.build(train, args.min_freq, embed)
        FEAT.build(train)
        REL.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_rels': len(REL.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'feat_pad_index': FEAT.pad_index
        })
        model = cls.MODEL(**args)
        model = model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform)
