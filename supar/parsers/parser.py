# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

import torch
from supar.models import MODELS
from supar.utils import Dataset
from supar.utils.field import Field
from supar.utils.logging import init_logger
from supar.utils.metric import AttachmentMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Parser(object):

    def __init__(self, args, model, transform):
        super(Parser, self).__init__()

        self.args = args
        self.model = model
        self.transform = transform

    def train(self, train, dev, test, logger=None, **kwargs):
        args = self.args.update(locals())
        logger = logger or init_logger(path=args.path)

        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev)
        test = Dataset(self.transform, args.test)
        train.build(args.batch_size, args.buckets, args.num_workers, True)
        dev.build(args.batch_size, args.buckets, args.num_workers)
        test.build(args.batch_size, args.buckets, args.num_workers)
        logger.info(f"Load the datasets\n"
                    f"{'train:':6} {train}\n"
                    f"{'dev:':6} {dev}\n"
                    f"{'test:':6} {test}\n")

        logger.info(f"{self.model}\n")
        self.optimizer = Adam(self.model.parameters(),
                              args.lr,
                              (args.mu, args.nu),
                              args.epsilon)
        self.scheduler = ExponentialLR(self.optimizer,
                                       args.decay**(1/args.decay_steps))

        elapsed = timedelta()
        best_e, best_metric = 1, AttachmentMetric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train.loader)
            loss, dev_metric = self._evaluate(dev.loader)
            logger.info(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
            loss, test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':6} - loss: {loss:.4f} - {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                self.save(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - best_e >= args.patience:
                break
        loss, metric = self.load(args.path)._evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':6} - {best_metric}")
        logger.info(f"{'test:':6} - {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, logger=None, **kwargs):
        args = self.args.update(locals())
        logger = logger or init_logger()

        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets, args.num_workers)
        logger.info(f"Load the dataset\n{dataset}")

        logger.info("Evaluate the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"{elapsed}s elapsed, "
                    f"{len(dataset)/elapsed.total_seconds():.2f} Sents/s")

    def predict(self, data, pred=None, prob=True, logger=None, **kwargs):
        args = self.args.update(locals())
        logger = logger or init_logger()

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets, args.num_workers)
        logger.info(f"Load the dataset\n{dataset}")

        logger.info("Make predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None:
            logger.info(f"Save predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, "
                    f"{len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    def _train(self, loader):
        raise NotImplementedError

    @ torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @ torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @ classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, **kwargs):
        if os.path.exists(path):
            state = torch.load(path)
        else:
            state = torch.hub.load_state_dict_from_url(path)
        args = state['args']
        args.update({'path': path, **kwargs})
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MODELS[args.model](args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        return cls(args, model, transform)

    def save(self, path):
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'args': self.args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path)
