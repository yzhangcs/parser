# -*- coding: utf-8 -*-

import torch


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __eq__(self, other):
        return self.score == other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    def __ne__(self, other):
        return self.score != other

    @property
    def score(self):
        raise AttributeError


class AttachmentMethod(Metric):

    def __init__(self, eps=1e-5):
        super(AttachmentMethod, self).__init__()

        self.eps = eps
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_labels = 0.0

    def __call__(self, pred_heads, pred_labels, gold_heads, gold_labels):
        arc_mask = pred_heads.eq(gold_heads)
        lab_mask = pred_labels.eq(gold_labels) & arc_mask

        self.total += len(arc_mask)
        self.correct_arcs += arc_mask.sum().item()
        self.correct_labels += lab_mask.sum().item()

    def __repr__(self):
        return f"UAS: {self.uas:.2%} LAS: {self.las:.2%}"

    @property
    def score(self):
        return self.las

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_labels / (self.total + self.eps)
