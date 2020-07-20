# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
from supar.utils.fn import stripe


class MatrixTree(nn.Module):
    """
    MatrixTree for calculating partition functions and marginals in O(N^3) for directed spanning trees
    (a.k.a. non-projective trees) by an adaptation of Kirchhoff's MatrixTree Theorem.
    This module differs from the original paper in that marginals are computed via back-propagation
    rather than matrix inversion.

    References:
    - Terry Koo, Amir Globerson, Xavier Carreras and Michael Collins (ACL'07)
      Structured Prediction Models via the Matrix-Tree Theorem
      https://www.aclweb.org/anthology/D07-1015/
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False):
        """
        Args:
            scores (Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible dependent-head pairs.
            mask (BoolTensor): [batch_size, seq_len]
                Mask to avoid aggregation on padding tokens.
                The first column with pseudo words as roots should be set to False.
            target (LongTensor, default: None): [batch_size, seq_len]
                Tensor of gold-standard dependent-head pairs.
            mbr (bool, default: False):
                If True, marginals will be returned to perform minimum Bayes-risk (mbr) decoding.

        Returns:
            loss (Tensor): scalar
                Loss averaged by number of tokens. This won't be returned if target is None.
            probs (Tensor): [batch_size, seq_len, ]
                Marginals if performing mbr decoding, original scores otherwise.
        """

        training = scores.requires_grad
        # double precision to prevent overflows
        scores = scores.double()
        logZ = self.matrix_tree(scores.requires_grad_(), mask)
        probs = scores
        # calculate the marginals
        if mbr:
            probs, = autograd.grad(logZ, probs, retain_graph=training)
        probs = probs.float()
        if target is None:
            return probs

        score = scores.gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
        loss = (logZ - score).float() / mask.sum()
        return loss, probs

    def matrix_tree(self, scores, mask):
        lens = mask.sum(-1)
        batch_size, seq_len, _ = scores.shape
        mask = mask.index_fill(1, mask.new_tensor(0).long(), 1)
        scores = scores.masked_fill(~(mask.unsqueeze(-1) & mask.unsqueeze(-2)), float('-inf'))

        # the numerical stability trick is borrowed from timvieira (https://github.com/timvieira/spanning_tree)
        # log(det(exp(M))) = log(det(exp(M - m) * exp(m)))
        #                  = log(det(exp(M - m)) * exp(m)^n)
        #                  = log(det(exp(M - m))) + m*n
        m = scores.view(batch_size, -1).max(-1)[0]
        # clamp the lower bound to `torch.finfo().tiny` to prevent underflows
        A = torch.exp(scores - m.view(-1, 1, 1)).clamp(torch.finfo().tiny)
        # D is the weighted degree matrix
        # D(i, j) = sum_j(A(i, j)), if h == m
        #           0,              otherwise
        D = torch.zeros_like(A)
        D.diagonal(0, 1, 2).copy_(A.sum(-1))
        # Laplacian matrix
        L = nn.init.eye_(torch.empty_like(A[0])).repeat(batch_size, 1, 1)
        L = L.masked_scatter_(mask.unsqueeze(-1), (D - A)[mask])
        # calculate the partition (a.k.a normalization) term
        # Z = L^(0, 0), which is the minor of L w.r.t row 0 and column 0
        logZ = (L[:, 1:, 1:].slogdet()[1] + m*lens).sum()

        return logZ


class CRFDependency(nn.Module):
    """
    First-order TreeCRF for calculating partition functions and marginals in O(N^3) for projective dependency trees.
    For efficient calculation The module provides a bathcified implementation
    and relpace the outside pass with back-propagation totally.

    References:
    - Yu Zhang, Zhenghua Li and Min Zhang (ACL'20)
      Efficient Second-Order TreeCRF for Neural Dependency Parsing
      https://www.aclweb.org/anthology/2020.acl-main.302/
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False, partial=False):
        """
        Args:
            scores (Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible dependent-head pairs.
            mask (BoolTensor): [batch_size, seq_len]
                Mask to avoid aggregation on padding tokens.
                The first column with pseudo words as roots should be set to False.
            target (LongTensor, default: None): [batch_size, seq_len]
                Tensor of gold-standard dependent-head pairs.
                This should be provided for loss calculation.
                If partially annotated, the unannotated positions should be filled with -1.
            mbr (bool, default: False):
                If True, marginals will be returned to perform minimum Bayes-risk (mbr) decoding.
            partial (bool, default: False):
                True indicates that the trees are partially annotated.

        Returns:
            loss (Tensor): scalar
                Loss averaged by number of tokens. This won't be returned if target is None.
            probs (Tensor): [batch_size, seq_len, seq_len]
                Marginals if performing mbr decoding, original scores otherwise.
        """

        training = scores.requires_grad
        batch_size, seq_len, _ = scores.shape
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        probs = scores
        if mbr:
            probs, = autograd.grad(logZ, scores, retain_graph=training)

        if target is None:
            return probs
        # the second inside process is needed if use partial annotation
        if partial:
            score = self.inside(scores, mask, target)
        else:
            score = scores.gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
        loss = (logZ - score) / mask.sum()

        return loss, probs

    def inside(self, scores, mask, cands=None):
        # the end position of each sentence in a batch
        lens = mask.sum(1)
        batch_size, seq_len, _ = scores.shape
        # [seq_len, seq_len, batch_size]
        scores = scores.permute(2, 1, 0)
        s_i = torch.full_like(scores, float('-inf'))
        s_c = torch.full_like(scores, float('-inf'))
        s_c.diagonal().fill_(0)

        # set the scores of arcs excluded by cands to -inf
        if cands is not None:
            mask = mask.index_fill(1, lens.new_tensor(0), 1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
            cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
            cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
            cands = cands.permute(2, 1, 0) & mask
            scores = scores.masked_fill(~cands, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w

            # ilr = C(i->r) + C(j->r+1)
            # [n, w, batch_size]
            ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
            if ilr.requires_grad:
                ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            il = ir = ilr.permute(2, 0, 1).logsumexp(-1)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i
            # with I(j->i) of n spans
            s_i.diagonal(-w).copy_(il + scores.diagonal(-w))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i
            # with I(i->j) of n spans
            s_i.diagonal(w).copy_(ir + scores.diagonal(w))

            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
            # disable multi words to modify the root
            s_c[0, w][lens.ne(w)] = float('-inf')

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()


class CRF2oDependency(nn.Module):
    """
    Second-order TreeCRF for calculating partition functions and marginals in O(N^3) for projective dependency trees.
    For efficient calculation The module provides a bathcified implementation
    and relpace the outside pass with back-propagation totally.


    References:
    - Yu Zhang, Zhenghua Li and Min Zhang (ACL'20)
      Efficient Second-Order TreeCRF for Neural Dependency Parsing
      https://www.aclweb.org/anthology/2020.acl-main.302/
    """

    def __init__(self):
        super().__init__()

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=True, partial=False):
        """
        Args:
            scores (Tuple[Tensor, Tensor]):
                Tuple of two tensors s_arc and s_sib.
                s_arc ([batch_size, seq_len, seq_len]) holds The scores of all possible dependent-head pairs.
                s_sib ([batch_size, seq_len, seq_len, seq_len]) holds the scores of dependent-head-sibling triples.
            mask (BoolTensor): [batch_size, seq_len]
                Mask to avoid aggregation on padding tokens.
                The first column with pseudo words as roots should be set to False.
            target (LongTensor, default: None): [batch_size, seq_len]
                Tensors of gold-standard dependent-head pairs and dependent-head-sibling triples.
                If partially annotated, the unannotated positions should be filled with -1.
            mbr (bool, default: False):
                If True, marginals will be returned to perform minimum Bayes-risk (mbr) decoding.
            partial (bool, default: False):
                True indicates that the trees are partially annotated.

        Returns:
            loss (Tensor): scalar
                Loss averaged by number of tokens. This won't be returned if target is None.
            probs (Tensor): [batch_size, seq_len, seq_len]
                Marginals if performing mbr decoding, original scores otherwise.
        """

        s_arc, s_sib = scores
        training = s_arc.requires_grad
        batch_size, seq_len, _ = s_arc.shape
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside((s.requires_grad_() for s in scores), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        probs = s_arc
        if mbr:
            probs, = autograd.grad(logZ, s_arc, retain_graph=training)

        if target is None:
            return probs
        arcs, sibs = target
        # the second inside process is needed if use partial annotation
        if partial:
            score = self.inside(scores, mask, arcs)
        else:
            arc_seq, sib_seq = arcs[mask], sibs[mask]
            arc_mask, sib_mask = mask, sib_seq.gt(0)
            sib_seq = sib_seq[sib_mask]
            s_sib = s_sib[mask][torch.arange(len(arc_seq)), arc_seq]
            s_arc = s_arc[arc_mask].gather(-1, arc_seq.unsqueeze(-1))
            s_sib = s_sib[sib_mask].gather(-1, sib_seq.unsqueeze(-1))
            score = s_arc.sum() + s_sib.sum()
        loss = (logZ - score) / mask.sum()

        return loss, probs

    def inside(self, scores, mask, cands=None):
        # the end position of each sentence in a batch
        lens = mask.sum(1)
        s_arc, s_sib = scores
        batch_size, seq_len, _ = s_arc.shape
        # [seq_len, seq_len, batch_size]
        s_arc = s_arc.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size]
        s_sib = s_sib.permute(2, 1, 3, 0)
        s_i = torch.full_like(s_arc, float('-inf'))
        s_s = torch.full_like(s_arc, float('-inf'))
        s_c = torch.full_like(s_arc, float('-inf'))
        s_c.diagonal().fill_(0)

        # set the scores of arcs excluded by cands to -inf
        if cands is not None:
            mask = mask.index_fill(1, lens.new_tensor(0), 1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
            cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
            cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
            cands = cands.permute(2, 1, 0) & mask
            s_arc = s_arc.masked_fill(~cands, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w
            # I(j->i) = logsum(exp(I(j->r) + S(j->r, i)) +, i < r < j
            #                  exp(C(j->j) + C(i->j-1)))
            #           + s(j->i)
            # [n, w, batch_size]
            il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
            il += stripe(s_sib[range(w, n+w), range(n)], n, w, (0, 1))
            # [n, 1, batch_size]
            il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
            # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
            il[:, -1] = il0.index_fill_(0, lens.new_tensor(0), 0).squeeze(1)
            if il.requires_grad:
                il.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            il = il.permute(2, 0, 1).logsumexp(-1)
            s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
            # I(i->j) = logsum(exp(I(i->r) + S(i->r, j)) +, i < r < j
            #                  exp(C(i->i) + C(j->i+1)))
            #           + s(i->j)
            # [n, w, batch_size]
            ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
            ir += stripe(s_sib[range(n), range(w, n+w)], n, w)
            ir[0] = float('-inf')
            # [n, 1, batch_size]
            ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
            ir[:, 0] = ir0.squeeze(1)
            if ir.requires_grad:
                ir.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            ir = ir.permute(2, 0, 1).logsumexp(-1)
            s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))

            # [n, w, batch_size]
            slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
            if slr.requires_grad:
                slr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            slr = slr.permute(2, 0, 1).logsumexp(-1)
            # S(j, i) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(-w).copy_(slr)
            # S(i, j) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(w).copy_(slr)

            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
            # disable multi words to modify the root
            s_c[0, w][lens.ne(w)] = float('-inf')

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()


class CRFConstituency(nn.Module):
    """
    TreeCRF for calculating partition functions and marginals in O(N^3) for constituency trees.
    For efficient calculation The module provides a bathcified implementation
    and relpace the outside pass with back-propagation totally.

    References:
    - Yu Zhang, houquan Zhou and Zhenghua Li (IJCAI'20)
      Fast and Accurate Neural CRF Constituency Parsing
      https://www.ijcai.org/Proceedings/2020/560/
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False):
        """
        Args:
            scores (Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible constituents.
            mask (BoolTensor): [batch_size, seq_len, seq_len]
                Mask to avoid parsing over padding tokens.
                For each square matrix in a batch, the positions except upper triangular part should be masked out.
            target (BoolTensor, default: None): [batch_size, seq_len, seq_len]
                Tensor of gold-standard constituents. True if a constituent exists.
            mbr (bool, default: False):
                If True, marginals will be returned to perform minimum Bayes-risk (mbr) decoding.

        Returns:
            loss (Tensor): scalar
                Loss averaged by number of tokens. This won't be returned if target is None.
            probs (Tensor): [batch_size, seq_len, seq_len]
                Marginals if performing mbr decoding, original scores otherwise.
        """

        training = scores.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        probs = scores
        if mbr:
            probs, = autograd.grad(logZ, scores, retain_graph=training)
        if target is None:
            return probs
        loss = (logZ - scores[mask & target].sum()) / mask[:, 0].sum()

        return loss, probs

    def inside(self, scores, mask):
        lens = mask[:, 0].sum(-1)
        batch_size, seq_len, _ = scores.shape
        # [seq_len, seq_len, batch_size]
        scores, mask = scores.permute(1, 2, 0), mask.permute(1, 2, 0)
        s = torch.full_like(scores, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w

            if w == 1:
                s.diagonal(w).copy_(scores.diagonal(w))
                continue
            # [n, w, batch_size]
            s_s = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
            # [batch_size, n, w]
            s_s = s_s.permute(2, 0, 1)
            if s_s.requires_grad:
                s_s.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_s = s_s.logsumexp(-1)
            s.diagonal(w).copy_(s_s + scores.diagonal(w))

        return s[0].gather(0, lens.unsqueeze(0)).sum()
