# -*- coding: utf-8 -*-

import operator
from typing import Iterable, Tuple, Union

import torch
from torch.autograd import Function

from supar.utils.common import MIN
from supar.utils.fn import pad


def tarjan(sequence: Iterable[int]) -> Iterable[int]:
    r"""
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.

    Args:
        sequence (list):
            List of head indices.

    Yields:
        A list of indices making up a SCC. All self-loops are ignored.

    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """

    sequence = [-1] + sequence
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


def chuliu_edmonds(s: torch.Tensor) -> torch.Tensor:
    r"""
    ChuLiu/Edmonds algorithm for non-projective decoding :cite:`mcdonald-etal-2005-non`.

    Some code is borrowed from `tdozat's implementation`_.
    Descriptions of notations and formulas can be found in :cite:`mcdonald-etal-2005-non`.

    Notes:
        The algorithm does not guarantee to parse a single-root tree.

    Args:
        s (~torch.Tensor): ``[seq_len, seq_len]``.
            Scores of all dependent-head pairs.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[seq_len]`` for the resulting non-projective parse tree.

    .. _tdozat's implementation:
        https://github.com/tdozat/Parser-v3
    """

    s[0, 1:] = MIN
    # prevent self-loops
    s.diagonal()[1:].fill_(MIN)
    # select heads with highest scores
    tree = s.argmax(-1)
    # return the cycle finded by tarjan algorithm lazily
    cycle = next(tarjan(tree.tolist()[1:]), None)
    # if the tree has no cycles, then it is a MST
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
        # s(x->c) = max(s(x'->x) - s(a(x')->x') + s(cycle)), x in noncycle and x' in cycle
        #                                                    a(v) is the predecessor of v in cycle
        #                                                    s(cycle) = sum(s(a(v)->v))
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

    # keep track of the endpoints of the edges into and out of cycle for reconstruction later
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


def mst(scores: torch.Tensor, mask: torch.BoolTensor, multiroot: bool = False) -> torch.Tensor:
    r"""
    MST algorithm for decoding non-projective trees.
    This is a wrapper for ChuLiu/Edmonds algorithm.

    The algorithm first runs ChuLiu/Edmonds to parse a tree and then have a check of multi-roots,
    If ``multiroot=True`` and there indeed exist multi-roots, the algorithm seeks to find
    best single-root trees by iterating all possible single-root trees parsed by ChuLiu/Edmonds.
    Otherwise the resulting trees are directly taken as the final outputs.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all dependent-head pairs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.
        multiroot (bool):
            Ensures to parse a single-root tree If ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting non-projective parse trees.

    Examples:
        >>> scores = torch.tensor([[[-11.9436, -13.1464,  -6.4789, -13.8917],
                                    [-60.6957, -60.2866, -48.6457, -63.8125],
                                    [-38.1747, -49.9296, -45.2733, -49.5571],
                                    [-19.7504, -23.9066,  -9.9139, -16.2088]]])
        >>> scores[:, 0, 1:] = MIN
        >>> scores.diagonal(0, 1, 2)[1:].fill_(MIN)
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> mst(scores, mask)
        tensor([[0, 2, 0, 2]])
    """

    _, seq_len, _ = scores.shape
    scores = scores.cpu().unbind()

    preds = []
    for i, length in enumerate(mask.sum(1).tolist()):
        s = scores[i][:length+1, :length+1]
        tree = chuliu_edmonds(s)
        roots = torch.where(tree[1:].eq(0))[0] + 1
        if not multiroot and len(roots) > 1:
            s_root = s[:, 0]
            s_best = MIN
            s = s.index_fill(1, torch.tensor(0), MIN)
            for root in roots:
                s[:, 0] = MIN
                s[root, 0] = s_root[root]
                t = chuliu_edmonds(s)
                s_tree = s[1:].gather(1, t[1:].unsqueeze(-1)).sum()
                if s_tree > s_best:
                    s_best, tree = s_tree, t
        preds.append(tree)

    return pad(preds, total_length=seq_len).to(mask.device)


def levenshtein(x: Iterable, y: Iterable, costs: Tuple = (1, 1, 1), align: bool = False) -> int:
    """
    Calculates the Levenshtein edit-distance between two sequencess,
    which refers to the total number of characters that must be
    substituted, deleted or inserted to transform `x` into `y`.

    The code is revised from `nltk`_ and `wiki`_'s implementations.

    Args:
        x/y (Iterable):
            The sequences to be analysed.
        costs (Tuple):
            Edit costs for substitution, deletion or insertion. Default: `(1, 1, 1)`.
        align (bool):
            Whether to return the alignments based on the minimum Levenshtein edit-distance. Default: ``False``.

    Examples:
        >>> from supar.structs.fn import levenshtein
        >>> levenshtein('intention', 'execution', align=True)
        (5, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)])

    .. _nltk:
        https://github.com/nltk/nltk/blob/develop/nltk/metrics/distance.py
    .. _wiki:
        https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    """

    # set up a 2-D array
    len1, len2 = len(x), len(y)
    lev = [list(range(len2 + 1))] + [[i] + [0] * len2 for i in range(1, len1 + 1)]
    alg = [[2] * (len2 + 1)] + [[1] + [-1] * len2 for _ in range(1, len1 + 1)] if align else None

    # iterate over the array
    # i and j start from 1 and not 0 to stay close to the wikipedia pseudo-code
    # see https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # substitution / keep
            s = lev[i - 1][j - 1] + (costs[0] if x[i - 1] != y[j - 1] else 0)
            # deletion
            a = lev[i - 1][j] + costs[1]
            # insertion
            b = lev[i][j - 1] + costs[2]

            edit, lev[i][j] = min(enumerate((s, a, b)), key=operator.itemgetter(1))
            if align:
                alg[i][j] = edit
    distance = lev[-1][-1]
    if align:
        i, j = len1, len2
        alignments = [(i, j)]
        while (i, j) != (0, 0):
            grids = [
                (i - 1, j - 1),  # substitution
                (i - 1, j),  # deletion
                (i, j - 1),  # insertion
            ]
            i, j = grids[alg[i][j]]
            alignments.append((i, j))
        alignments = list(reversed(alignments))
    return (distance, alignments) if align else distance


class Logsumexp(Function):

    r"""
    Safer ``logsumexp`` to cure unnecessary NaN values that arise from inf arguments.
    See discussions at http://github.com/pytorch/pytorch/issues/49724.
    To be optimized with C++/Cuda extensions.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        output = x.logsumexp(dim)
        ctx.dim = dim
        ctx.save_for_backward(x, output)
        return output.clone()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g: torch.Tensor) -> Union[torch.Tensor, None]:
        x, output, dim = *ctx.saved_tensors, ctx.dim
        g, output = g.unsqueeze(dim), output.unsqueeze(dim)
        mask = g.eq(0).expand_as(x)
        grad = g * (x - output).exp()
        return torch.where(mask, x.new_tensor(0.), grad), None


class Logaddexp(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = torch.logaddexp(x, y)
        ctx.save_for_backward(x, y, output)
        return output.clone()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        x, y, output = ctx.saved_tensors
        mask = g.eq(0)
        grad_x, grad_y = (x - output).exp(), (y - output).exp()
        grad_x = torch.where(mask, x.new_tensor(0.), grad_x)
        grad_y = torch.where(mask, y.new_tensor(0.), grad_y)
        return grad_x, grad_y


class SampledLogsumexp(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx.dim = dim
        ctx.save_for_backward(x)
        return x.logsumexp(dim=dim)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g: torch.Tensor) -> Union[torch.Tensor, None]:
        from torch.distributions import OneHotCategorical
        (x, ), dim = ctx.saved_tensors, ctx.dim
        return g.unsqueeze(dim).mul(OneHotCategorical(logits=x.movedim(dim, -1)).sample().movedim(-1, dim)), None


class Sparsemax(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx.dim = dim
        sorted_x, _ = x.sort(dim, True)
        z = sorted_x.cumsum(dim) - 1
        k = x.new_tensor(range(1, sorted_x.size(dim) + 1)).view(-1, *[1] * (x.dim() - 1)).transpose(0, dim)
        k = (k * sorted_x).gt(z).sum(dim, True)
        tau = z.gather(dim, k - 1) / k
        p = torch.clamp(x - tau, 0)
        ctx.save_for_backward(k, p)
        return p

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g: torch.Tensor) -> Tuple[torch.Tensor, None]:
        k, p, dim = *ctx.saved_tensors, ctx.dim
        grad = g.masked_fill(p.eq(0), 0)
        grad = torch.where(p.ne(0), grad - grad.sum(dim, True) / k, grad)
        return grad, None


logsumexp = Logsumexp.apply

logaddexp = Logaddexp.apply

sampled_logsumexp = SampledLogsumexp.apply

sparsemax = Sparsemax.apply
