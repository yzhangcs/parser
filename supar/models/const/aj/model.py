# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.model import Model
from supar.modules import GraphConvolutionalNetwork
from supar.utils import AttachJuxtaposeTree, Config
from supar.utils.common import INF
from supar.utils.fn import pad


class AttachJuxtaposeConstituencyModel(Model):
    r"""
    The implementation of AttachJuxtapose Constituency Parser :cite:`yang-deng-2020-aj`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
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
            The dropout ratio of encoder layer. Default: .33.
        n_span_mlp (int):
            Span MLP size. Default: 500.
        n_label_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_labels,
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
                 n_span_mlp=500,
                 n_label_mlp=100,
                 mlp_dropout=.33,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        # the last one represents the dummy node in the initial states
        self.label_embed = nn.Embedding(n_labels+1, self.args.n_encoder_hidden)
        self.gnn_layers = GraphConvolutionalNetwork(n_model=self.args.n_encoder_hidden,
                                                    n_layers=self.args.n_gnn_layers,
                                                    dropout=self.args.gnn_dropout)

        self.node_classifier = nn.Sequential(
            nn.Linear(2 * self.args.n_encoder_hidden, self.args.n_encoder_hidden // 2),
            nn.LayerNorm(self.args.n_encoder_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.args.n_encoder_hidden // 2, 1),
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(2 * self.args.n_encoder_hidden, self.args.n_encoder_hidden // 2),
            nn.LayerNorm(self.args.n_encoder_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.args.n_encoder_hidden // 2, 2 * n_labels),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
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
            ~torch.Tensor:
                Contextualized output hidden states of shape ``[batch_size, seq_len, n_model]`` of the input.
        """

        return self.encode(words, feats)

    def loss(self, x, nodes, parents, news, mask):
        r"""
        Args:
            x (~torch.Tensor): ``[batch_size, seq_len, n_model]``.
                Contextualized output hidden states.
            nodes (~torch.LongTensor): ``[batch_size, seq_len]``.
                The target node positions on rightmost chains.
            parents (~torch.LongTensor): ``[batch_size, seq_len]``.
                The parent node labels of terminals.
            news (~torch.LongTensor): ``[batch_size, seq_len]``.
                The parent node labels of juxtaposed targets and terminals.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        spans, s_node, x_node = None, [], []
        actions = torch.stack((nodes, parents, news))
        for t, action in enumerate(actions.unbind(-1)):
            x_p, x_t, mask_p, mask_t = x[:, :t], x[:, t], mask[:, :t], mask[:, t]
            lens = mask_p.sum(-1)
            if t == 0:
                x_span = self.label_embed(lens.new_full((x.shape[0], 1), self.args.n_labels))
                span_mask = mask_t.unsqueeze(1)
            else:
                span_mask = spans[:, :-1, 1:].ge(0)
                span_lens = span_mask.sum((-1, -2))
                span_indices = torch.where(span_mask)
                span_labels = spans[:, :-1, 1:][span_indices]
                x_span = self.label_embed(span_labels)
                x_span += x[span_indices[0], span_indices[1]] + x[span_indices[0], span_indices[2]]
                node_lens = lens + span_lens
                adj_mask = node_lens.unsqueeze(-1).gt(x.new_tensor(range(node_lens.max())))
                x_mask = lens.unsqueeze(-1).gt(x.new_tensor(range(adj_mask.shape[-1])))
                span_mask = ~x_mask & adj_mask
                # concatenate terminals and spans
                x_tree = x.new_zeros(*adj_mask.shape, x.shape[-1]).masked_scatter_(x_mask.unsqueeze(-1), x_p[mask_p])
                x_tree = x_tree.masked_scatter_(span_mask.unsqueeze(-1), x_span)
                adj = mask.new_zeros(*x_tree.shape[:-1], x_tree.shape[1])
                adj_spans = lens.new_tensor(range(x_tree.shape[1])).view(1, 1, -1).repeat(2, x.shape[0], 1)
                adj_spans = adj_spans.masked_scatter_(span_mask.unsqueeze(0), torch.stack(span_indices[1:]))
                adj_l, adj_r, adj_w = *adj_spans.unbind(), adj_spans[1] - adj_spans[0]
                adj_parent = adj_l.unsqueeze(-1).ge(adj_l.unsqueeze(-2)) & adj_r.unsqueeze(-1).le(adj_r.unsqueeze(-2))
                # set the parent of root as itself
                adj_parent.diagonal(0, 1, 2).copy_(adj_w.eq(t - 1))
                adj_parent = adj_parent & span_mask.unsqueeze(1)
                # closet ancestor spans as parents
                adj_parent = (adj_w.unsqueeze(-2) - adj_w.unsqueeze(-1)).masked_fill_(~adj_parent, t).argmin(-1)
                adj.scatter_(-1, adj_parent.unsqueeze(-1), 1)
                adj = (adj | adj.transpose(-1, -2)).float()
                x_tree = self.gnn_layers(x_tree, adj, adj_mask)
                span_mask = span_mask.masked_scatter(span_mask, span_indices[2].eq(t-1))
                span_lens = span_mask.sum(-1)
                x_tree, span_mask = x_tree[span_mask], span_lens.unsqueeze(-1).gt(x.new_tensor(range(span_lens.max())))
                x_span = x.new_zeros(*span_mask.shape, x.shape[-1]).masked_scatter_(span_mask.unsqueeze(-1), x_tree)
            x_rightmost = torch.cat((x_span, x_t.unsqueeze(1).expand_as(x_span)), -1)
            s_node.append(self.node_classifier(x_rightmost).squeeze(-1))
            # we found softmax is slightly better than sigmoid in the original paper
            s_node[-1] = s_node[-1].masked_fill_(~span_mask, -INF).masked_fill(~span_mask.any(-1).unsqueeze(-1), 0)
            x_node.append(torch.bmm(s_node[-1].softmax(-1).unsqueeze(1), x_span).squeeze(1))
            spans = AttachJuxtaposeTree.action2span(action, spans, self.args.nul_index, mask_t)
        attach_mask = x.new_tensor(range(self.args.n_labels)).eq(self.args.nul_index)
        s_node, x_node = pad(s_node, padding_value=-INF).transpose(0, 1), torch.stack(x_node, 1)
        s_parent, s_new = self.label_classifier(torch.cat((x, x_node), -1)).chunk(2, -1)
        s_parent = torch.cat((s_parent[:, :1].masked_fill(attach_mask, -INF), s_parent[:, 1:]), 1)
        s_new = torch.cat((s_new[:, :1].masked_fill(~attach_mask, -INF), s_new[:, 1:]), 1)
        node_loss = self.criterion(s_node[mask], nodes[mask])
        label_loss = self.criterion(s_parent[mask], parents[mask]) + self.criterion(s_new[mask], news[mask])
        return node_loss + label_loss

    def decode(self, x, mask):
        r"""
        Args:
            x (~torch.Tensor): ``[batch_size, seq_len, n_model]``.
                Contextualized output hidden states.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            List[List[Tuple]]:
                Sequences of factorized labeled trees.
        """

        spans = None
        batch_size, *_ = x.shape
        beam_size, n_labels = self.args.beam_size, self.args.n_labels
        # [batch_size * beam_size, ...]
        x = x.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *x.shape[1:])
        mask = mask.unsqueeze(1).repeat(1, beam_size, 1).view(-1, *mask.shape[1:])
        # [batch_size]
        batches = x.new_tensor(range(batch_size)).long() * beam_size
        # accumulated scores
        scores = x.new_full((batch_size, beam_size), -INF).index_fill_(-1, x.new_tensor(0).long(), 0).view(-1)
        for t in range(x.shape[1]):
            x_p, x_t, mask_p, mask_t = x[:, :t], x[:, t], mask[:, :t], mask[:, t]
            lens = mask_p.sum(-1)
            if t == 0:
                x_span = self.label_embed(lens.new_full((x.shape[0], 1), n_labels))
                span_mask = mask_t.unsqueeze(1)
            else:
                span_mask = spans[:, :-1, 1:].ge(0)
                span_lens = span_mask.sum((-1, -2))
                span_indices = torch.where(span_mask)
                span_labels = spans[:, :-1, 1:][span_indices]
                x_span = self.label_embed(span_labels)
                x_span += x[span_indices[0], span_indices[1]] + x[span_indices[0], span_indices[2]]
                node_lens = lens + span_lens
                adj_mask = node_lens.unsqueeze(-1).gt(x.new_tensor(range(node_lens.max())))
                x_mask = lens.unsqueeze(-1).gt(x.new_tensor(range(adj_mask.shape[-1])))
                span_mask = ~x_mask & adj_mask
                # concatenate terminals and spans
                x_tree = x.new_zeros(*adj_mask.shape, x.shape[-1]).masked_scatter_(x_mask.unsqueeze(-1), x_p[mask_p])
                x_tree = x_tree.masked_scatter_(span_mask.unsqueeze(-1), x_span)
                adj = mask.new_zeros(*x_tree.shape[:-1], x_tree.shape[1])
                adj_spans = lens.new_tensor(range(x_tree.shape[1])).view(1, 1, -1).repeat(2, x.shape[0], 1)
                adj_spans = adj_spans.masked_scatter_(span_mask.unsqueeze(0), torch.stack(span_indices[1:]))
                adj_l, adj_r, adj_w = *adj_spans.unbind(), adj_spans[1] - adj_spans[0]
                adj_parent = adj_l.unsqueeze(-1).ge(adj_l.unsqueeze(-2)) & adj_r.unsqueeze(-1).le(adj_r.unsqueeze(-2))
                # set the parent of root as itself
                adj_parent.diagonal(0, 1, 2).copy_(adj_w.eq(t - 1))
                adj_parent = adj_parent & span_mask.unsqueeze(1)
                # closet ancestor spans as parents
                adj_parent = (adj_w.unsqueeze(-2) - adj_w.unsqueeze(-1)).masked_fill_(~adj_parent, t).argmin(-1)
                adj.scatter_(-1, adj_parent.unsqueeze(-1), 1)
                adj = (adj | adj.transpose(-1, -2)).float()
                x_tree = self.gnn_layers(x_tree, adj, adj_mask)
                span_mask = span_mask.masked_scatter(span_mask, span_indices[2].eq(t-1))
                span_lens = span_mask.sum(-1)
                x_tree, span_mask = x_tree[span_mask], span_lens.unsqueeze(-1).gt(x.new_tensor(range(span_lens.max())))
                x_span = x.new_zeros(*span_mask.shape, x.shape[-1]).masked_scatter_(span_mask.unsqueeze(-1), x_tree)
            s_node = self.node_classifier(torch.cat((x_span, x_t.unsqueeze(1).expand_as(x_span)), -1)).squeeze(-1)
            s_node = s_node.masked_fill_(~span_mask, -INF).masked_fill(~span_mask.any(-1).unsqueeze(-1), 0).log_softmax(-1)
            # we found softmax is slightly better than sigmoid in the original paper
            x_node = torch.bmm(s_node.exp().unsqueeze(1), x_span).squeeze(1)
            s_parent, s_new = self.label_classifier(torch.cat((x_t, x_node), -1)).chunk(2, -1)
            s_parent, s_new = s_parent.log_softmax(-1), s_new.log_softmax(-1)
            if t == 0:
                s_parent[:, self.args.nul_index] = -INF
                s_new[:, s_new.new_tensor(range(self.args.n_labels)).ne(self.args.nul_index)] = -INF
            s_node, nodes = s_node.topk(min(s_node.shape[-1], beam_size), -1)
            s_parent, parents = s_parent.topk(min(n_labels, beam_size), -1)
            s_new, news = s_new.topk(min(n_labels, beam_size), -1)
            s_action = s_node.unsqueeze(2) + (s_parent.unsqueeze(2) + s_new.unsqueeze(1)).view(x.shape[0], 1, -1)
            s_action = s_action.view(x.shape[0], -1)
            k_beam, k_node, k_parent = s_action.shape[-1], parents.shape[-1] * news.shape[-1], news.shape[-1]
            # [batch_size * beam_size, k_beam]
            scores = scores.unsqueeze(-1) + s_action
            # [batch_size, beam_size]
            scores, cands = scores.view(batch_size, -1).topk(beam_size, -1)
            # [batch_size * beam_size]
            scores = scores.view(-1)
            beams = cands.div(k_beam, rounding_mode='floor')
            nodes = nodes.view(batch_size, -1).gather(-1, cands.div(k_node, rounding_mode='floor'))
            indices = (batches.unsqueeze(-1) + beams).view(-1)
            parents = parents[indices].view(batch_size, -1).gather(-1, cands.div(k_parent, rounding_mode='floor') % k_parent)
            news = news[indices].view(batch_size, -1).gather(-1, cands % k_parent)
            action = torch.stack((nodes, parents, news)).view(3, -1)
            spans = spans[indices] if spans is not None else None
            spans = AttachJuxtaposeTree.action2span(action, spans, self.args.nul_index, mask_t)
        mask = mask.view(batch_size, beam_size, -1)[:, 0]
        # select an 1-best tree for each sentence
        spans = spans[batches + scores.view(batch_size, -1).argmax(-1)]
        span_mask = spans.ge(0)
        span_mask[:, :-1, 1:] &= mask.unsqueeze(1) & mask.unsqueeze(2)
        span_indices = torch.where(span_mask)
        span_labels = spans[span_indices]
        chart_preds = [[] for _ in range(x.shape[0])]
        for i, *span in zip(*[s.tolist() for s in span_indices], span_labels.tolist()):
            chart_preds[i].append(span)
        return chart_preds
