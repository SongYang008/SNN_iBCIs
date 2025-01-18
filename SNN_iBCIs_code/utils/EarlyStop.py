# -*- coding: utf-8 -*-
"""
@Time： 2024/6/4 15:52
@Auth： S Yang
@File：EarlyStop.py
@IDE：PyCharm
@Project: IBCI
@Motto：ABC(Always Be Coding)
"""
import copy

import torch


class EarlyStop:
    def __init__(self, model, min_acc=0., patience=100, save_path='./model.pth'):
        self.stop = False
        self.acc = min_acc
        self.counter = 0
        self.patience = patience
        self.model = model
        self.best_model = None
        self.file_path = save_path

    def __call__(self, current_acc):
        if current_acc <= self.acc:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_model = copy.deepcopy(self.model)
            self.acc = current_acc
            self.counter = 0
            self._save_model()

    def _save_model(self):
        torch.save(self.model.state_dict(), self.file_path)
