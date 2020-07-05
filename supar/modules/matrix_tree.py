# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn


class MatrixTree(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        total = mask.sum()
        scores = scores.double()
        mask = mask.index_fill(1, mask.new_tensor(0).long(), 1)
        A = scores.requires_grad_().exp()
        A = A * mask.unsqueeze(1) * mask.unsqueeze(-1)
        batch_size, seq_len, _ = A.shape
        # D is the weighted degree matrix
        D = torch.zeros_like(A)
        D.diagonal(0, 1, 2).copy_(A.sum(-1))
        # Laplacian matrix
        L = nn.init.eye_(torch.empty_like(A[0])).repeat(batch_size, 1, 1)
        L[mask] = (D - A)[mask]
        # calculate the partition (a.k.a normalization) term
        logZ = L[:, 1:, 1:].slogdet()[1].sum()
        mask = mask.index_fill(1, mask.new_tensor(0).long(), 0)
        # calculate the marginal probablities
        probs, = autograd.grad(logZ, scores, retain_graph=scores.requires_grad)
        probs = probs.float()

        if target is None:
            return probs

        score = scores.gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
        loss = (logZ - score).float() / total
        return loss, probs
