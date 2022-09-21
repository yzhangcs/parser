# -*- coding: utf-8 -*-

import os

import torch
from supar.models.sdp.biaffine import BiaffineSemanticDependencyModel
from supar.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.logging import get_logger
from supar.utils.metric import ChartMetric
from supar.utils.parallel import parallel
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch, CoNLL

logger = get_logger(__name__)


class BiaffineSemanticDependencyParser(Parser):
    r"""
    The implementation of Biaffine Semantic Dependency Parser :cite:`dozat-manning-2018-simpler`.
    """

    NAME = 'biaffine-semantic-dependency'
    MODEL = BiaffineSemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (Union[str, Iterable]):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, verbose=True, **kwargs):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for evaluation. Both a filename and a list of instances are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, prob=False,
                verbose=True, **kwargs):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for prediction.
                - a filename. If ends with `.txt`, the parser will seek to make predictions line by line from plain texts.
                - a list of instances.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @parallel()
    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, labels = batch
        mask = batch.mask
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0
        s_edge, s_label = self.model(words, feats)
        loss = self.model.loss(s_edge, s_label, labels, mask)
        return loss

    @parallel(training=False)
    def eval_step(self, batch: Batch) -> ChartMetric:
        words, *feats, labels = batch
        mask = batch.mask
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0
        s_edge, s_label = self.model(words, feats)
        loss = self.model.loss(s_edge, s_label, labels, mask)
        label_preds = self.model.decode(s_edge, s_label)
        return ChartMetric(loss, label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))

    @parallel(training=False, op=None)
    def pred_step(self, batch: Batch) -> Batch:
        words, *feats = batch
        mask, lens = batch.mask, (batch.lens - 1).tolist()
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0
        with torch.autocast(self.device, enabled=self.args.amp):
            s_edge, s_label = self.model(words, feats)
        label_preds = self.model.decode(s_edge, s_label).masked_fill(~mask, -1)
        batch.labels = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                                               for row in chart[1:i, :i].tolist()])
                        for i, chart in zip(lens, label_preds)]
        if self.args.prob:
            batch.probs = [prob[1:i, :i].cpu() for i, prob in zip(lens, s_edge.softmax(-1).unbind())]
        return batch

    @classmethod
    def build(cls, path, min_freq=7, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (Dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.transform.FORM[0].embed).to(parser.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
        TAG, CHAR, LEMMA, ELMO, BERT = None, None, None, None, None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'elmo' in args.feat:
                from allennlp.modules.elmo import batch_to_ids
                ELMO = RawField('elmo')
                ELMO.compose = lambda x: batch_to_ids(x).to(WORD.device)
            if 'bert' in args.feat:
                t = TransformerTokenizer(args.bert)
                BERT = SubwordField('bert', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
                BERT.vocab = t.vocab
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        transform = CoNLL(FORM=(WORD, CHAR, ELMO, BERT), LEMMA=LEMMA, POS=TAG, PHEAD=LABEL)

        train = Dataset(transform, args.train, **args)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
            if LEMMA is not None:
                LEMMA.build(train)
        LABEL.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_labels': len(LABEL.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser
