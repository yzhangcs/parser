# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
from datetime import datetime, timedelta

import dill
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import supar
from supar.utils import Config, Dataset
from supar.utils.field import Field
from supar.utils.fn import download, get_rng_state, set_rng_state
from supar.utils.logging import get_logger, init_logger, progress_bar
from supar.utils.metric import Metric
from supar.utils.optim import InverseSquareRootLR, LinearLR
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import gather, is_master, parallel, sync
from supar.utils.transform import Batch

logger = get_logger(__name__)


class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              clip=5.0, epochs=5000, patience=100, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        logger.info("Loading the data")
        train = Dataset(self.transform, args.train, **args).build(batch_size, buckets, True, dist.is_initialized(), workers)
        dev = Dataset(self.transform, args.dev, **args).build(batch_size, buckets, False, dist.is_initialized(), workers)
        logger.info(f"{'train:':6} {train}")
        if not args.test:
            logger.info(f"{'dev:':6} {dev}\n")
        else:
            test = Dataset(self.transform, args.test, **args).build(batch_size, buckets, False, dist.is_initialized(), workers)
            logger.info(f"{'dev:':6} {dev}")
            logger.info(f"{'test:':6} {test}\n")

        if args.encoder == 'lstm':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        elif args.encoder == 'transformer':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = InverseSquareRootLR(self.optimizer, args.warmup_steps)
        else:
            # we found that Huggingface's AdamW is more robust and empirically better than the native implementation
            from transformers import AdamW
            steps = len(train.loader) * epochs // args.update_steps
            self.optimizer = AdamW(
                [{'params': p, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
                 for n, p in self.model.named_parameters()],
                args.lr,
                (args.mu, args.nu),
                args.eps,
                args.weight_decay
            )
            self.scheduler = LinearLR(self.optimizer, int(steps*args.warmup), steps)
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get('find_unused_parameters', True))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(dist.group.WORLD, fp16_compress_hook)

        self.epoch, self.best_e, self.patience, self.best_metric, self.elapsed = 1, 1, patience, Metric(), timedelta()
        if self.args.checkpoint:
            try:
                self.optimizer.load_state_dict(self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                train.loader.batch_sampler.epoch = self.epoch
            except AttributeError:
                logger.warning("No checkpoint found. Try re-launching the traing procedure instead")

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar, metric = progress_bar(train.loader), Metric()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            for i, batch in enumerate(bar, 1):
                with sync(self.model, i % self.args.update_steps == 0):
                    with torch.autocast(self.device, enabled=self.args.amp):
                        loss = self.train_step(batch)
                    loss.backward()
                if i % self.args.update_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(True)
                bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e}")
            logger.info(f"{bar.postfix}")
            with torch.autocast(self.device, enabled=self.args.amp):
                metric = sum([self.eval_step(batch) for batch in progress_bar(dev.loader)], Metric())
                logger.info(f"{'dev:':5} {metric}")
                if args.test:
                    logger.info(f"{'test:':5} {sum([self.eval_step(batch) for batch in progress_bar(test.loader)], Metric())}")

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
        if dist.is_initialized():
            dist.barrier()

        parser = self.load(**args)
        # only allow the master device to save models
        if is_master():
            parser.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        if args.test:
            logger.info(f"{'test:':5} {parser._evaluate(test.loader)}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        data = Dataset(self.transform, **args)
        data.build(batch_size, buckets, False, dist.is_initialized(), workers)
        logger.info(f"\n{data}")

        logger.info("Evaluating the data")
        start = datetime.now()
        metric = sum([self.eval_step(batch) for batch in progress_bar(data.loader)], Metric())
        elapsed = datetime.now() - start
        logger.info(f"{metric}")
        logger.info(f"{elapsed}s elapsed, {len(data)/elapsed.total_seconds():.2f} Sents/s")

        return metric

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, prob=False, cache=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        data = Dataset(self.transform, **args)
        data.build(batch_size, buckets, False, dist.is_initialized(), workers)
        logger.info(f"\n{data}")

        logger.info("Making predictions on the data")
        start = datetime.now()
        with tempfile.TemporaryDirectory() as t, parallel(False, None):
            # we have clustered the sentences by length here to speed up prediction,
            # so the order of the yielded sentences can't be guaranteed
            for batch in progress_bar(data.loader):
                batch = self.pred_step(batch)
                if args.cache:
                    for s in batch:
                        with open(os.path.join(t, f"{s.index}"), 'w') as f:
                            f.write(str(s) + '\n')
            elapsed = datetime.now() - start

            if dist.is_initialized():
                dist.barrier()
            if args.cache:
                tdirs = gather(t) if dist.is_initialized() else (t,)
            if pred is not None and is_master():
                logger.info(f"Saving predicted results to {pred}")
                with open(pred, 'w') as f:
                    # merge all predictions into one single file
                    if args.cache:
                        sentences = (os.path.join(i, s) for i in tdirs for s in os.listdir(i))
                        for i in progress_bar(sorted(sentences, key=lambda x: int(os.path.basename(x)))):
                            with open(i) as s:
                                shutil.copyfileobj(s, f)
                    else:
                        for s in progress_bar(data):
                            f.write(str(s) + '\n')
            # exit util all files have been merged
            if dist.is_initialized():
                dist.barrier()
        logger.info(f"{elapsed}s elapsed, {len(data) / elapsed.total_seconds():.2f} Sents/s")

        if not cache:
            return data

    @parallel()
    def train_step(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError

    @parallel(training=False)
    def eval_step(self, batch: Batch) -> Metric:
        raise NotImplementedError

    @parallel(training=False, op=None)
    def pred_step(self, batch: Batch) -> Batch:
        raise NotImplementedError

    def backward(self, loss: torch.Tensor, **kwargs):
        loss /= self.args.update_steps
        if hasattr(self, 'scaler'):
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, reload=False, src='github', checkpoint=False, **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'biaffine-dep-en'``.
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
            kwargs (Dict):
                A dict holding unconsumed arguments for updating training configs and initializing the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('biaffine-dep-en')
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

    def save(self, path):
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

    def save_checkpoint(self, path):
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
