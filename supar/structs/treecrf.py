# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
from supar.utils.fn import stripe


class MatrixTree(nn.Module):
    r"""
    MatrixTree for calculating partitions and marginals of directed spanning trees (a.k.a. non-projective trees)
    in :math:`O(n^3)` by an adaptation of Kirchhoff's MatrixTree Theorem :cite:`koo-etal-2007-structured`.

    Different from the original paper, marginals are computed via back-propagation rather than matrix inversion.
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False, partial=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs. Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        training = scores.requires_grad
        logZ = self.matrix_tree(scores.requires_grad_(), mask)
        marginals = scores
        # calculate the marginals
        if mbr:
            marginals, = autograd.grad(logZ, marginals, retain_graph=training)

        if target is None:
            return marginals
        # the second inside process is needed if using partial annotation
        if partial:
            score = self.matrix_tree(scores, mask, target)
        else:
            score = scores.gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
        loss = (logZ - score) / mask.sum()

        return loss, marginals

    def matrix_tree(self, s_arc, mask, cands=None):
        lens = mask.sum(-1)
        batch_size, seq_len, _ = s_arc.shape
        mask = mask.index_fill(1, lens.new_tensor(0), 1)
        chart_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        s_arc = s_arc.masked_fill(~chart_mask, float('-inf'))

        # set the arcs scores excluded by cands to -inf
        if cands is not None:
            cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
            cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
            cands = cands & chart_mask
            s_arc = s_arc.masked_fill(~cands, float('-inf'))

        # A(i, j) = exp(s(i, j))
        # double precision to prevent overflows
        A = torch.exp(s_arc).double()
        # Weighted degree matrix
        # D(i, j) = sum_j(A(i, j)), if h == m
        #           0,              otherwise
        D = torch.zeros_like(A)
        D.diagonal(0, 1, 2).copy_(A.sum(-1))
        # Laplacian matrix
        # L(i, j) = D(i, j) - A(i, j)
        L = nn.init.eye_(torch.empty_like(A[0])).repeat(batch_size, 1, 1).masked_scatter_(mask.unsqueeze(-1), (D - A)[mask])
        # Z = L^(0, 0), which is the minor of L w.r.t row 0 and column 0
        logZ = L[:, 1:, 1:].logdet().sum().float()

        return logZ


class CRFDependency(nn.Module):
    r"""
    First-order TreeCRF for calculating partitions and marginals of projective dependency trees
    in :math:`O(n^3)` :cite:`zhang-etal-2020-efficient`.
    """

    def __init__(self, multiroot=False):
        super().__init__()

        self.multiroot = multiroot

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False, partial=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs.
                This should be provided for loss calculation.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        training = scores.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        marginals = scores
        if mbr:
            marginals, = autograd.grad(logZ, scores, retain_graph=training)

        if target is None:
            return marginals
        # the second inside process is needed if using partial annotation
        if partial:
            score = self.inside(scores, mask, target)
        else:
            score = scores.gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
        loss = (logZ - score) / mask.sum()

        return loss, marginals

    def inside(self, s_arc, mask, cands=None):
        # the end position of each sentence in a batch
        lens = mask.sum(1)
        batch_size, seq_len, _ = s_arc.shape
        # [seq_len, seq_len, batch_size]
        s_arc = s_arc.permute(2, 1, 0)
        s_i = torch.full_like(s_arc, float('-inf'))
        s_c = torch.full_like(s_arc, float('-inf'))
        s_c.diagonal().fill_(0)

        # set the arcs scores excluded by cands to -inf
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

            # ilr = C(i->r) + C(j->r+1)
            # [n, w, batch_size]
            ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
            if ilr.requires_grad:
                ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            il = ir = ilr.permute(2, 0, 1).logsumexp(-1)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i with I(j->i) of n spans
            s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i with I(i->j) of n spans
            s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))

            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
            if not self.multiroot:
                s_c[0, w][lens.ne(w)] = float('-inf')

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()


class CRF2oDependency(nn.Module):
    r"""
    Second-order TreeCRF for calculating partitions and marginals of projective dependency trees
    in :math:`O(n^3)` :cite:`zhang-etal-2020-efficient`.
    """

    def __init__(self, multiroot=False):
        super().__init__()

        self.multiroot = multiroot

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=True, partial=False):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of two tensors `s_arc` and `s_sib`.
                `s_arc` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                Tensors of gold-standard dependent-head pairs and dependent-head-sibling triples.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        s_arc, s_sib = scores
        training = s_arc.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(*(s.requires_grad_() for s in scores), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        marginals = s_arc
        if mbr:
            marginals, = autograd.grad(logZ, s_arc, retain_graph=training)

        if target is None:
            return marginals
        arcs, sibs = target
        # the second inside process is needed if using partial annotation
        if partial:
            score = self.inside(s_arc, s_sib, mask, arcs)
        else:
            s_arc = s_arc.gather(-1, arcs.unsqueeze(-1))[mask]
            s_sib = s_sib.gather(-1, sibs.unsqueeze(-1))[sibs.gt(0)]
            score = s_arc.sum() + s_sib.sum()
        loss = (logZ - score) / mask.sum()

        return loss, marginals

    def inside(self, s_arc, s_sib, mask, cands=None):
        # the end position of each sentence in a batch
        lens = mask.sum(1)
        batch_size, seq_len, _ = s_arc.shape
        # [seq_len, seq_len, batch_size]
        s_arc = s_arc.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size]
        s_sib = s_sib.permute(2, 1, 3, 0)
        s_i = torch.full_like(s_arc, float('-inf'))
        s_s = torch.full_like(s_arc, float('-inf'))
        s_c = torch.full_like(s_arc, float('-inf'))
        s_c.diagonal().fill_(0)

        # set the arcs scores excluded by cands to -inf
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
            if not self.multiroot:
                s_c[0, w][lens.ne(w)] = float('-inf')

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()


class CRFConstituency(nn.Module):
    r"""
    TreeCRF for calculating partitions and marginals of constituency trees in :math:`O(n^3)` :cite:`zhang-etal-2020-fast`.
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible constituents.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid parsing over padding tokens.
                For each square matrix in a batch, the positions except upper triangular part should be masked out.
            target (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard constituents. ``True`` if a constituent exists. Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        training = scores.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        marginals = scores
        if mbr:
            marginals, = autograd.grad(logZ, scores, retain_graph=training)
        if target is None:
            return marginals
        loss = (logZ - scores[mask & target].sum()) / mask[:, 0].sum()

        return loss, marginals

    def inside(self, scores, mask):
        lens = mask[:, 0].sum(-1)
        batch_size, seq_len, _ = scores.shape
        # [seq_len, seq_len, batch_size]
        scores, mask = scores.permute(1, 2, 0), mask.permute(1, 2, 0)
        s = torch.full_like(scores, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of constituents to iterate,
            # from constituent (0, w) to constituent (n, n+w) given width w
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
