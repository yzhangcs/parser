# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from parser.metric import AttachmentMethod

import torch
import torch.nn as nn


class Trainer(object):

    def __init__(self, model, vocab, optimizer, scheduler):
        super(Trainer, self).__init__()

        self.model = model
        self.vocab = vocab
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, train_loader, dev_loader, test_loader,
            epochs, patience, file):
        total_time = timedelta()
        max_e, max_metric = 0, 0.0

        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            self.train(train_loader)

            print(f"Epoch: {epoch} / {epochs}:")
            loss, train_metric = self.evaluate(train_loader)
            print(f"{'train:':<6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = self.evaluate(dev_loader)
            print(f"{'dev:':<6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = self.evaluate(test_loader)
            print(f"{'test:':<6} Loss: {loss:.4f} {test_metric}")
            t = datetime.now() - start
            print(f"{t}s elapsed\n")
            total_time += t

            # save the model if it is the best so far
            if dev_metric > max_metric:
                torch.save(self.model, file)
                max_e, max_metric = epoch, dev_metric
            elif epoch - max_e >= patience:
                break
        self.model = torch.load(file)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        loss, metric = self.evaluate(test_loader)

        print(f"max score of dev is {max_metric.score:.2%} at epoch {max_e}")
        print(f"the score of test at epoch {max_e} is {metric.score:.2%}")
        print(f"mean time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

    def train(self, loader):
        self.model.train()

        for x, char_x, heads, labels in loader:
            self.optimizer.zero_grad()

            mask = x.gt(0)
            s_arc, s_lab = self.model(x, char_x)
            # ignore the first token of each sentence
            mask[:, 0] = 0

            loss = self.model.get_loss(s_arc, s_lab, heads, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        loss, metric = 0, AttachmentMethod()

        for x, char_x, heads, labels in loader:
            mask = x.gt(0)
            s_arc, s_lab = self.model(x, char_x)
            # ignore the first token of each sentence
            mask[:, 0] = 0

            loss += self.model.get_loss(s_arc, s_lab, heads, labels, mask)
            metric(s_arc, s_lab, heads, labels, mask)
        loss /= len(loader)

        return loss, metric
