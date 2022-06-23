# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
from datetime import datetime, timedelta

import dill
import supar
import torch
import torch.distributed as dist
from supar.utils import Config, Dataset
from supar.utils.field import Field
from supar.utils.fn import download, get_rng_state, set_rng_state
from supar.utils.logging import init_logger, logger, progress_bar
from supar.utils.metric import Metric
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import gather, is_master, parallel
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

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
        test = Dataset(self.transform, args.test, **args).build(batch_size, buckets, False, dist.is_initialized(), workers)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        if args.encoder == 'lstm':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        else:
            from transformers import AdamW, get_linear_schedule_with_warmup
            steps = len(train.loader) * epochs // args.update_steps
            self.optimizer = AdamW(
                [{'params': p, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
                 for n, p in self.model.named_parameters()],
                args.lr)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(steps*args.warmup), steps)
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)

        self.epoch, self.best_e, self.patience, self.best_metric, self.elapsed = 1, 1, patience, Metric(), timedelta()
        if self.args.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint_state_dict.pop('optimizer_state_dict'))
            self.scheduler.load_state_dict(self.checkpoint_state_dict.pop('scheduler_state_dict'))
            self.scaler.load_state_dict(self.checkpoint_state_dict.pop('scaler_state_dict'))
            set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
            for k, v in self.checkpoint_state_dict.items():
                setattr(self, k, v)
            train.loader.batch_sampler.epoch = self.epoch

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train.loader)
            dev_loss, dev_metric = self._evaluate(dev.loader)
            logger.info(f"{'dev:':5} loss: {dev_loss:.4f} - {dev_metric}")
            test_loss, test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':5} loss: {test_loss:.4f} - {test_metric}")

            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t

            if dev_metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, dev_metric
                if is_master():
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        if dist.is_initialized():
            dist.barrier()
        args.device = args.local_rank
        parser = self.load(**args)
        loss, metric = parser._evaluate(test.loader)
        # only allow the master device to save models
        if is_master():
            parser.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        logger.info(f"{'test:':5} {metric}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        dataset = Dataset(self.transform, **args)
        dataset.build(batch_size, buckets, False, dist.is_initialized(), workers)
        logger.info(f"\n{dataset}")

        logger.info("Evaluating the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        if dist.is_initialized():
            loss = loss / dist.get_world_size()
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, prob=False, cache=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        dataset = Dataset(self.transform, **args)
        dataset.build(batch_size, buckets, False, dist.is_initialized(), workers)
        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")
        start = datetime.now()
        with tempfile.TemporaryDirectory() as t:
            # we have clustered the sentences by length here to speed up prediction,
            # so the order of the yielded sentences can't be guaranteed
            for s in self._predict(dataset.loader):
                if args.cache:
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
                        for s in progress_bar(dataset):
                            f.write(str(s) + '\n')
            # exit util all files have been merged
            if dist.is_initialized():
                dist.barrier()
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        if not cache:
            return dataset

    @parallel()
    def _train(self, loader):
        raise NotImplementedError

    @parallel(training=False)
    def _evaluate(self, loader):
        raise NotImplementedError

    @parallel(training=False, op=None)
    def _predict(self, loader):
        raise NotImplementedError

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
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs and initializing the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('biaffine-dep-en')
            >>> parser = Parser.load('./ptb.biaffine.dep.lstm.char')
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(path):
            path = download(supar.MODEL[src].get(path, path), reload=reload)
        state = torch.load(path, map_location='cpu')
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        parser = cls(args, model, transform)
        parser.checkpoint_state_dict = state['checkpoint_state_dict'] if checkpoint else None
        return parser

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)

    def save_checkpoint(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        checkpoint_state_dict = {k: getattr(self, k) for k in ['epoch', 'best_e', 'patience', 'best_metric', 'elapsed']}
        checkpoint_state_dict.update({'optimizer_state_dict': self.optimizer.state_dict(),
                                      'scheduler_state_dict': self.scheduler.state_dict(),
                                      'scaler_state_dict': self.scaler.state_dict(),
                                      'rng_state': get_rng_state()})
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'checkpoint_state_dict': checkpoint_state_dict,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)
