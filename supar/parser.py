# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterable, Union

import dill
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler

import supar
from supar.config import Config
from supar.utils import Dataset
from supar.utils.field import Field
from supar.utils.fn import download, get_rng_state, set_rng_state
from supar.utils.logging import get_logger, init_logger, progress_bar
from supar.utils.metric import Metric
from supar.utils.optim import InverseSquareRootLR, LinearLR
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import gather, is_dist, is_master, reduce
from supar.utils.transform import Batch

logger = get_logger(__name__)


class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

    def __repr__(self):
        s = f'{self.__class__.__name__}(\n'
        s += '\n'.join(['  '+i for i in str(self.model).split('\n')]) + '\n'
        s += '\n'.join(['  '+i for i in str(self.transform).split('\n')]) + '\n)'
        return s

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def sync_grad(self):
        return self.step % self.args.update_steps == 0 or self.step % self.n_batches == 0

    @contextmanager
    def sync(self):
        context = getattr(contextlib, 'suppress' if sys.version < '3.7' else 'nullcontext')
        if is_dist() and not self.sync_grad:
            context = self.model.no_sync
        with context():
            yield

    @contextmanager
    def join(self):
        context = getattr(contextlib, 'suppress' if sys.version < '3.7' else 'nullcontext')
        if not is_dist():
            with context():
                yield
        elif self.model.training:
            with self.model.join():
                yield
        else:
            try:
                dist_model = self.model
                # https://github.com/pytorch/pytorch/issues/54059
                if hasattr(self.model, 'module'):
                    self.model = self.model.module
                yield
            finally:
                self.model = dist_model

    def train(
        self,
        train: Union[str, Iterable],
        dev: Union[str, Iterable],
        test: Union[str, Iterable],
        epochs: int,
        patience: int,
        batch_size: int = 5000,
        update_steps: int = 1,
        buckets: int = 32,
        workers: int = 0,
        clip: float = 5.0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> None:
        r"""
        Args:
            train/dev/test (Union[str, Iterable]):
                Filenames of the train/dev/test datasets.
            epochs (int):
                The number of training iterations.
            patience (int):
                The number of consecutive iterations after which the training process would be early stopped if no improvement.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            clip (float):
                Clips gradient of an iterable of parameters at specified value. Default: 5.0.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
        """

        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        eval_batch_size = args.get('eval_batch_size', batch_size)
        if is_dist():
            batch_size = batch_size // dist.get_world_size()
            eval_batch_size = eval_batch_size // dist.get_world_size()
        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        args.even = args.get('even', is_dist())
        train = Dataset(self.transform, args.train, **args).build(
            batch_size=batch_size,
            n_buckets=buckets,
            shuffle=True,
            distributed=is_dist(),
            even=args.even,
            seed=args.seed,
            n_workers=workers
        )
        dev = Dataset(self.transform, args.dev, **args).build(
            batch_size=eval_batch_size,
            n_buckets=buckets,
            shuffle=False,
            distributed=is_dist(),
            even=False,
            n_workers=workers
        )
        logger.info(f"{'train:':6} {train}")
        if not args.test:
            logger.info(f"{'dev:':6} {dev}\n")
        else:
            test = Dataset(self.transform, args.test, **args).build(
                batch_size=eval_batch_size,
                n_buckets=buckets,
                shuffle=False,
                distributed=is_dist(),
                even=False,
                n_workers=workers
            )
            logger.info(f"{'dev:':6} {dev}")
            logger.info(f"{'test:':6} {test}\n")
        loader, sampler = train.loader, train.loader.batch_sampler
        args.steps = len(loader) * epochs // args.update_steps
        args.save(f"{args.path}.yaml")

        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(module=self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get('find_unused_parameters', True),
                             static_graph=args.get('static_graph', False))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(dist.group.WORLD, fp16_compress_hook)
        if args.wandb and is_master():
            import wandb
            # start a new wandb run to track this script
            wandb.init(config=args.primitive_config,
                       project=args.get('project', self.NAME),
                       name=args.get('name', args.path),
                       resume=self.args.checkpoint)
        self.step, self.epoch, self.best_e, self.patience = 1, 1, 1, patience
        # uneven batches are excluded
        self.n_batches = min(gather(len(loader))) if is_dist() else len(loader)
        self.best_metric, self.elapsed = Metric(), timedelta()
        if args.checkpoint:
            try:
                self.optimizer.load_state_dict(self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                sampler.set_epoch(self.epoch)
            except AttributeError:
                logger.warning("No checkpoint found. Try re-launching the training procedure instead")

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar, metric = progress_bar(loader), Metric()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self.model.train()
            with self.join():
                # we should reset `step` as the number of batches in different processes is not necessarily equal
                self.step = 1
                for batch in bar:
                    with self.sync():
                        with torch.autocast(self.device, enabled=args.amp):
                            loss = self.train_step(batch)
                        self.backward(loss)
                    if self.sync_grad:
                        self.clip_grad_norm_(self.model.parameters(), args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(True)
                    bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
                    # log metrics to wandb
                    if args.wandb and is_master():
                        wandb.log({'lr': self.scheduler.get_last_lr()[0], 'loss': loss})
                    self.step += 1
                logger.info(f"{bar.postfix}")
            self.model.eval()
            with self.join(), torch.autocast(self.device, enabled=args.amp):
                metric = self.reduce(sum([self.eval_step(i) for i in progress_bar(dev.loader)], Metric()))
                logger.info(f"{'dev:':5} {metric}")
                if args.wandb and is_master():
                    wandb.log({'dev': metric.values, 'epochs': epoch})
                if args.test:
                    test_metric = self.reduce(sum([self.eval_step(i) for i in progress_bar(test.loader)], Metric()))
                    logger.info(f"{'test:':5} {test_metric}")
                    if args.wandb and is_master():
                        wandb.log({'test': test_metric.values, 'epochs': epoch})

            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t

            if metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, metric
                if is_master():
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        if is_dist():
            dist.barrier()

        best = self.load(**args)
        # only allow the master device to save models
        if is_master():
            best.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        if args.test:
            best.model.eval()
            with best.join():
                test_metric = sum([best.eval_step(i) for i in progress_bar(test.loader)], Metric())
                logger.info(f"{'test:':5} {best.reduce(test_metric)}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")
        if args.wandb and is_master():
            wandb.finish()

    def evaluate(
        self,
        data: Union[str, Iterable],
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for evaluation. Both a filename and a list of instances are allowed.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.

        Returns:
            The evaluation results.
        """

        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        if is_dist():
            batch_size = batch_size // dist.get_world_size()
        data = Dataset(self.transform, **args)
        data.build(batch_size=batch_size,
                   n_buckets=buckets,
                   shuffle=False,
                   distributed=is_dist(),
                   even=False,
                   n_workers=workers)
        logger.info(f"\n{data}")

        logger.info("Evaluating the data")
        start = datetime.now()
        self.model.eval()
        with self.join():
            bar, metric = progress_bar(data.loader), Metric()
            for batch in bar:
                metric += self.eval_step(batch)
                bar.set_postfix_str(metric)
            metric = self.reduce(metric)
        elapsed = datetime.now() - start
        logger.info(f"{metric}")
        logger.info(f"{elapsed}s elapsed, "
                    f"{sum(data.sizes)/elapsed.total_seconds():.2f} Tokens/s, "
                    f"{len(data)/elapsed.total_seconds():.2f} Sents/s")

        return metric

    def predict(
        self,
        data: Union[str, Iterable],
        pred: str = None,
        lang: str = None,
        prob: bool = False,
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
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
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        if is_dist():
            batch_size = batch_size // dist.get_world_size()
        data = Dataset(self.transform, **args)
        data.build(batch_size=batch_size,
                   n_buckets=buckets,
                   shuffle=False,
                   distributed=is_dist(),
                   even=False,
                   n_workers=workers)
        logger.info(f"\n{data}")

        logger.info("Making predictions on the data")
        start = datetime.now()
        self.model.eval()
        with tempfile.TemporaryDirectory() as t:
            # we have clustered the sentences by length here to speed up prediction,
            # so the order of the yielded sentences can't be guaranteed
            for batch in progress_bar(data.loader):
                batch = self.pred_step(batch)
                if is_dist() or args.cache:
                    for s in batch.sentences:
                        with open(os.path.join(t, f"{s.index}"), 'w') as f:
                            f.write(str(s) + '\n')
            elapsed = datetime.now() - start

            if is_dist():
                dist.barrier()
            tdirs = gather(t) if is_dist() else (t,)
            if pred is not None and is_master():
                logger.info(f"Saving predicted results to {pred}")
                with open(pred, 'w') as f:
                    # merge all predictions into one single file
                    if is_dist() or args.cache:
                        sentences = (os.path.join(i, s) for i in tdirs for s in os.listdir(i))
                        for i in progress_bar(sorted(sentences, key=lambda x: int(os.path.basename(x)))):
                            with open(i) as s:
                                shutil.copyfileobj(s, f)
                    else:
                        for s in progress_bar(data):
                            f.write(str(s) + '\n')
            # exit util all files have been merged
            if is_dist():
                dist.barrier()
        logger.info(f"{elapsed}s elapsed, "
                    f"{sum(data.sizes)/elapsed.total_seconds():.2f} Tokens/s, "
                    f"{len(data)/elapsed.total_seconds():.2f} Sents/s")

        if not cache:
            return data

    def backward(self, loss: torch.Tensor, **kwargs):
        loss /= self.args.update_steps
        if hasattr(self, 'scaler'):
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def clip_grad_norm_(
        self,
        params: Union[Iterable[torch.Tensor], torch.Tensor],
        max_norm: float,
        norm_type: float = 2
    ) -> torch.Tensor:
        self.scaler.unscale_(self.optimizer)
        return nn.utils.clip_grad_norm_(params, max_norm, norm_type)

    def clip_grad_value_(
        self,
        params: Union[Iterable[torch.Tensor], torch.Tensor],
        clip_value: float
    ) -> None:
        self.scaler.unscale_(self.optimizer)
        return nn.utils.clip_grad_value_(params, clip_value)

    def reduce(self, obj: Any) -> Any:
        if not is_dist():
            return obj
        return reduce(obj)

    def train_step(self, batch: Batch) -> torch.Tensor:
        ...

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> Metric:
        ...

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        ...

    def init_optimizer(self) -> Optimizer:
        if self.args.encoder in ('lstm', 'transformer'):
            optimizer = Adam(params=self.model.parameters(),
                             lr=self.args.lr,
                             betas=(self.args.get('mu', 0.9), self.args.get('nu', 0.999)),
                             eps=self.args.get('eps', 1e-8),
                             weight_decay=self.args.get('weight_decay', 0))
        else:
            # we found that Huggingface's AdamW is more robust and empirically better than the native implementation
            from transformers import AdamW
            optimizer = AdamW(params=[{'params': p, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)}
                                      for n, p in self.model.named_parameters()],
                              lr=self.args.lr,
                              betas=(self.args.get('mu', 0.9), self.args.get('nu', 0.999)),
                              eps=self.args.get('eps', 1e-8),
                              weight_decay=self.args.get('weight_decay', 0))
        return optimizer

    def init_scheduler(self) -> _LRScheduler:
        if self.args.encoder == 'lstm':
            scheduler = ExponentialLR(optimizer=self.optimizer,
                                      gamma=self.args.decay**(1/self.args.decay_steps))
        elif self.args.encoder == 'transformer':
            scheduler = InverseSquareRootLR(optimizer=self.optimizer,
                                            warmup_steps=self.args.warmup_steps)
        else:
            scheduler = LinearLR(optimizer=self.optimizer,
                                 warmup_steps=self.args.get('warmup_steps', int(self.args.steps*self.args.get('warmup', 0))),
                                 steps=self.args.steps)
        return scheduler

    @classmethod
    def build(cls, path, **kwargs):
        ...

    @classmethod
    def load(
        cls,
        path: str,
        reload: bool = False,
        src: str = 'github',
        checkpoint: bool = False,
        **kwargs
    ) -> Parser:
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'dep-biaffine-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            src (str):
                Specifies where to download the model.
                ``'github'``: github release page.
                ``'hlt'``: hlt homepage, only accessible from 9:00 to 18:00 (UTC+8).
                Default: ``'github'``.
            checkpoint (bool):
                If ``True``, loads all checkpoint states to restore the training process. Default: ``False``.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('dep-biaffine-en')
            >>> parser = Parser.load('./ptb.biaffine.dep.lstm.char')
        """

        args = Config(**locals())
        if not os.path.exists(path):
            path = download(supar.MODEL[src].get(path, path), reload=reload)
        state = torch.load(path, map_location='cpu')
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        transform = state['transform']
        parser = cls(args, model, transform)
        parser.checkpoint_state_dict = state.get('checkpoint_state_dict', None) if checkpoint else None
        parser.model.to(parser.device)
        return parser

    def save(self, path: str) -> None:
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': model.args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)

    def save_checkpoint(self, path: str) -> None:
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        checkpoint_state_dict = {k: getattr(self, k) for k in ['epoch', 'best_e', 'patience', 'best_metric', 'elapsed']}
        checkpoint_state_dict.update({'optimizer_state_dict': self.optimizer.state_dict(),
                                      'scheduler_state_dict': self.scheduler.state_dict(),
                                      'scaler_state_dict': self.scaler.state_dict(),
                                      'rng_state': get_rng_state()})
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': model.args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'checkpoint_state_dict': checkpoint_state_dict,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)
