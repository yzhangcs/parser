# -*- coding: utf-8 -*-

import torch
from supar.utils.fn import pad, stripe


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # count the frequency of each datapoint
    d, indices, f = x.unique(return_inverse=True, return_counts=True)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # the number of clusters must not be greater than that of datapoints
    k = min(len(d), k)
    # initialize k centroids randomly
    c, old = d[torch.randperm(len(d))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)

    while old is None or not c.equal(old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        mask = y.eq(torch.arange(k).unsqueeze(-1))
        # update the centroids
        c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, assigned = y[indices], y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


def eisner(scores, mask):
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    p_i = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = scores.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        il = ir = ilr.permute(2, 0, 1)
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span + scores.diagonal(-w))
        p_i.diagonal(-w).copy_(il_path + starts)
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span + scores.diagonal(w))
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    def backtrack(p_i, p_c, heads, i, j, complete):
        if i == j:
            return
        if complete:
            r = p_c[i, j]
            backtrack(p_i, p_c, heads, i, r, False)
            backtrack(p_i, p_c, heads, r, j, True)
        else:
            r, heads[j] = p_i[i, j], i
            i, j = sorted((i, j))
            backtrack(p_i, p_c, heads, i, r, True)
            backtrack(p_i, p_c, heads, j, r + 1, True)

    preds = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_zeros(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        preds.append(heads.to(mask.device))

    return pad(preds, total_length=seq_len).to(mask.device)


def cky(scores, mask):
    lens = mask[:, 0].sum(-1)
    scores = scores.permute(1, 2, 0)
    seq_len, seq_len, batch_size = scores.shape
    s = scores.new_zeros(seq_len, seq_len, batch_size)
    p = scores.new_zeros(seq_len, seq_len, batch_size).long()

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p.new_tensor(range(n)).unsqueeze(0)

        if w == 1:
            s.diagonal(w).copy_(scores.diagonal(w))
            continue
        # [n, w, batch_size]
        s_span = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
        # [batch_size, n, w]
        s_span = s_span.permute(2, 0, 1)
        # [batch_size, n]
        s_span, p_span = s_span.max(-1)
        s.diagonal(w).copy_(s_span + scores.diagonal(w))
        p.diagonal(w).copy_(p_span + starts + 1)

    def backtrack(p, i, j):
        if j == i + 1:
            return [(i, j)]
        split = p[i][j]
        ltree = backtrack(p, i, split)
        rtree = backtrack(p, split, j)
        return [(i, j)] + ltree + rtree

    p = p.permute(2, 0, 1).tolist()
    trees = [backtrack(p[i], 0, length)
             for i, length in enumerate(lens.tolist())]

    return trees


def tarjan(sequence):
    sequence[0] = -1
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False

            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)


def chuliu_edmonds(s):
    tree = s.argmax(-1)
    # return the cycle finded by tarjan algorithm lazily
    cycle = next(tarjan(tree.tolist()), None)
    # if `tree` has no cycles, then it is a MST
    if not cycle:
        return tree
    # indices of cycle in the original tree
    cycle = torch.tensor(cycle)
    # indices of noncycle in the original tree
    noncycle = torch.ones(len(s)).index_fill_(0, cycle, 0)
    noncycle = torch.where(noncycle.gt(0))[0]

    def contract(s):
        # heads of cycle in original tree
        cycle_heads = tree[cycle]
        # scores of cycle in original tree
        s_cycle = s[cycle, cycle_heads]

        # calculate the scores of cycle's potential dependents
        # s(c->x) = max(s(x'->x)), x in noncycle and x' in cycle
        s_dep = s[noncycle][:, cycle]
        # find the best cycle head for each noncycle dependent
        deps = s_dep.argmax(1)
        # calculate the scores of cycle's potential heads
        # s(x->c) = max(s(x'->x) - s(a(x')->x') + s(cycle)),
        #           x in noncycle and x' in cycle
        #           a(v) is the predecessor of v in cycle
        #           s(cycle) = sum(s(a(v)->v))
        s_head = s[cycle][:, noncycle] - s_cycle.view(-1, 1) + s_cycle.sum()
        # find the best noncycle head for each cycle dependent
        heads = s_head.argmax(0)

        contracted = torch.cat((noncycle, torch.tensor([-1])))
        # calculate the scores of contracted graph
        s = s[contracted][:, contracted]
        # set the contracted graph scores of cycle's potential dependents
        s[:-1, -1] = s_dep[range(len(deps)), deps]
        # set the contracted graph scores of cycle's potential heads
        s[-1, :-1] = s_head[heads, range(len(heads))]

        return s, heads, deps

    # keep track of the endpoints of the edges into and out of cycle
    # for reconstruction later
    s, heads, deps = contract(s)

    # y is the contracted tree
    y = chuliu_edmonds(s)
    # exclude head of cycle from y
    y, cycle_head = y[:-1], y[-1]

    # fix the subtree with no heads coming from the cycle
    # len(y) denotes heads coming from the cycle
    subtree = y < len(y)
    # add the nodes to the new tree
    tree[noncycle[subtree]] = noncycle[y[subtree]]
    # fix the subtree with heads coming from the cycle
    subtree = ~subtree
    # add the nodes to the tree
    tree[noncycle[subtree]] = cycle[deps[subtree]]
    # fix the root of the cycle
    cycle_root = heads[cycle_head]
    # break the cycle and add the root of the cycle to the tree
    tree[cycle[cycle_root]] = noncycle[cycle_head]

    return tree


def mst(scores, mask, multiroot=False):
    r'''
    Please note that the ChuLiu/Edmod algorithm does not guarantee
    to parse a tree with single root.
    '''
    batch_size, seq_len, _ = scores.shape
    scores = scores.cpu().unbind()

    preds = []
    for i, length in enumerate(mask.sum(1).tolist()):
        s = scores[i][:length+1, :length+1]
        tree = chuliu_edmonds(s)
        roots = torch.where(tree[1:].eq(0))[0] + 1
        if not multiroot and len(roots) > 1:
            s_root = s[:, 0]
            s_best = float('-inf')
            s = s.index_fill(1, torch.tensor(0), float('-inf'))
            for root in roots:
                s[:, 0] = float('-inf')
                s[root, 0] = s_root[root]
                t = chuliu_edmonds(s)
                s_tree = s[1:].gather(1, t[1:].unsqueeze(-1)).sum()
                if s_tree > s_best:
                    s_best, tree = s_tree, t
        preds.append(tree)

    return pad(preds, total_length=seq_len).to(mask.device)
