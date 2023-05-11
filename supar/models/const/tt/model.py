# -*- coding: utf-8 -*-

from typing import List, Tuple

import torch
import torch.nn as nn
from supar.config import Config
from supar.model import Model
from supar.utils.common import INF


class TetraTaggingConstituencyModel(Model):
    r"""
    The implementation of TetraTagging Constituency Parser :cite:`kitaev-klein-2020-tetra`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layers. Default: .33.
        n_gnn_layers (int):
            The number of GNN layers. Default: 3.
        gnn_dropout (float):
            The dropout ratio of GNN layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, True),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 n_encoder_hidden=800,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_gnn_layers=3,
                 gnn_dropout=.33,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.proj = nn.Linear(self.args.n_encoder_hidden, self.args.n_leaves + self.args.n_nodes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        words: torch.LongTensor,
        feats: List[torch.LongTensor] = None
    ) -> torch.Tensor:
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Scores for all leaves (``[batch_size, seq_len, n_leaves]``) and nodes (``[batch_size, seq_len, n_nodes]``).
        """

        s = self.proj(self.encode(words, feats)[:, 1:-1])
        s_leaf, s_node = s[..., :self.args.n_leaves], s[..., self.args.n_leaves:]
        return s_leaf, s_node

    def loss(
        self,
        s_leaf: torch.Tensor,
        s_node: torch.Tensor,
        leaves: torch.LongTensor,
        nodes: torch.LongTensor,
        mask: torch.BoolTensor
    ) -> torch.Tensor:
        r"""
        Args:
            s_leaf (~torch.Tensor): ``[batch_size, seq_len, n_leaves]``.
                Leaf scores.
            s_node (~torch.Tensor): ``[batch_size, seq_len, n_nodes]``.
                Non-terminal scores.
            leaves (~torch.LongTensor): ``[batch_size, seq_len]``.
                Actions for leaves.
            nodes (~torch.LongTensor): ``[batch_size, seq_len]``.
                Actions for non-terminals.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        leaf_mask, node_mask = mask, mask[:, 1:]
        leaf_loss = self.criterion(s_leaf[leaf_mask], leaves[leaf_mask])
        node_loss = self.criterion(s_node[:, :-1][node_mask], nodes[node_mask]) if nodes.shape[1] > 0 else 0
        return leaf_loss + node_loss

    def decode(
        self,
        s_leaf: torch.Tensor,
        s_node: torch.Tensor,
        mask: torch.BoolTensor,
        left_mask: torch.BoolTensor,
        depth: int = 8
    ) -> List[List[Tuple]]:
        r"""
        Args:
            s_leaf (~torch.Tensor): ``[batch_size, seq_len, n_leaves]``.
                Leaf scores.
            s_node (~torch.Tensor): ``[batch_size, seq_len, n_nodes]``.
                Non-terminal scores.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens in each chart.
            left_mask (~torch.BoolTensor): ``[n_leaves + n_nodes]``.
                The mask for distingushing left/rightward actions.
            depth (int):
                Stack depth. Default: 8.

        Returns:
            List[List[Tuple]]:
                Sequences of factorized labeled trees.
        """
        from torch_scatter import scatter_max

        lens = mask.sum(-1)
        batch_size, seq_len, n_leaves = s_leaf.shape
        leaf_left_mask, node_left_mask = left_mask[:n_leaves], left_mask[n_leaves:]
        # [n_leaves], [n_nodes]
        changes = (torch.where(leaf_left_mask, 1, 0), torch.where(node_left_mask, 0, -1))
        # [batch_size, depth]
        depths = lens.new_full((depth,), -2).index_fill_(-1, lens.new_tensor(0), -1).repeat(batch_size, 1)
        # [2, batch_size, depth, seq_len]
        labels, paths = lens.new_zeros(2, batch_size, depth, seq_len), lens.new_zeros(2, batch_size, depth, seq_len)
        # [batch_size, depth]
        s = s_leaf.new_zeros(batch_size, depth)

        def advance(s, s_t, depths, changes):
            batch_size, n_labels = s_t.shape
            # [batch_size, depth * n_labels]
            depths = (depths.unsqueeze(-1) + changes).view(batch_size, -1)
            # [batch_size, depth, n_labels]
            s_t = s.unsqueeze(-1) + s_t.unsqueeze(1)
            # [batch_size, depth * n_labels]
            # fill scores of invalid depths with -INF
            s_t = s_t.view(batch_size, -1).masked_fill_((depths < 0).logical_or_(depths >= depth), -INF)
            # [batch_size, depth]
            # for each depth, we use the `scatter_max` trick to obtain the 1-best label
            s, ls = scatter_max(s_t, depths.clamp(0, depth - 1), -1, s_t.new_full((batch_size, depth), -INF))
            # [batch_size, depth]
            depths = depths.gather(-1, ls.clamp(0, depths.shape[1] - 1)).masked_fill_(s.eq(-INF), -1)
            ll = ls % n_labels
            lp = depths - changes[ll]
            return s, ll, lp, depths

        for t in range(seq_len):
            m = lens.gt(t)
            s[m], labels[0, m, :, t], paths[0, m, :, t], depths[m] = advance(s[m], s_leaf[m, t], depths[m], changes[0])
            if t == seq_len - 1:
                break
            m = lens.gt(t + 1)
            s[m], labels[1, m, :, t], paths[1, m, :, t], depths[m] = advance(s[m], s_node[m, t], depths[m], changes[1])

        lens = lens.tolist()
        labels, paths = labels.movedim((0, 2), (2, 3))[mask].split(lens), paths.movedim((0, 2), (2, 3))[mask].split(lens)
        leaves, nodes = [], []
        for i, length in enumerate(lens):
            leaf_labels, node_labels = labels[i].transpose(0, 1).tolist()
            leaf_paths, node_paths = paths[i].transpose(0, 1).tolist()
            leaf_pred, node_pred, prev = [leaf_labels[-1][0]], [], leaf_paths[-1][0]
            for j in reversed(range(length - 1)):
                node_pred.append(node_labels[j][prev])
                prev = node_paths[j][prev]
                leaf_pred.append(leaf_labels[j][prev])
                prev = leaf_paths[j][prev]
            leaves.append(list(reversed(leaf_pred)))
            nodes.append(list(reversed(node_pred)))
        return leaves, nodes
