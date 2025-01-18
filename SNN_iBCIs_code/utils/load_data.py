# -*- coding: utf-8 -*-
"""
@Time： 2024/5/15 10:27
@Auth： S Yang
@File：load_data.py
@IDE：PyCharm
@Project: IBCI
@Motto：ABC(Always Be Coding)
"""
import random

import torch
from torch.utils.data.dataset import Dataset, Subset
import numpy as np
import os


class Invasive(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.from_numpy(x)
        x = x.float()
        size = x.size()

        x = torch.reshape(x, (1, size[0], size[1]))
        y = torch.from_numpy(self.target[idx])
        return x, y


class InvasiveDataset(Invasive):
    def __init__(self, root):
        self.data_dir = os.listdir(root)
        self.data_dir.sort()
        data = []
        target = []
        t = []
        for f in self.data_dir:
            f = os.path.join(root, f)
            xyz = np.load(f, allow_pickle=True)
            x = xyz['x']
            data.append(x[:, : 12000])
            target.append(xyz['y'])
            t.append(xyz['z'])
        super().__init__(data, target)


class MergedDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets[0]
        self.lengths = [len(dataset) for dataset in self.datasets]
        # self.cumulative_lengths = [sum(self.lengths[:i + 1]) for i in range(len(self.lengths))]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dataset_idx = 0
        for length in self.lengths:
            if idx >= length:
                idx -= length
                dataset_idx += 1
            else:
                break
        ans = self.datasets[dataset_idx][idx][:]
        return ans


def TrainValSet(dataset, ratio=None):
    if ratio is None:
        ratio = [0.8, 0.2]
    data_num = (len(dataset))
    idx = list(range(data_num))
    random.shuffle(idx)
    train_num = int(data_num * ratio[0])
    val_num = data_num - train_num
    train_idx = idx[0: train_num]
    val_idx = idx[train_num:]
    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx)
    return train, val


def LOO(ordatasets, session):
    test_set = ordatasets[session]
    # datasets.pop(session)
    datasets = []
    for i, d in enumerate(ordatasets):
        if i != session:
            datasets.append(d)
    train_val_set = MergedDataset(datasets)
    train_set, val_set = TrainValSet(train_val_set)
    return train_set, val_set, test_set


def load_all(root):
    sessions = os.listdir(root)
    print(sessions)
    datasets = []
    for s in sessions:
        ds = InvasiveDataset(root=os.path.join(root, s))
        datasets.append(ds)
    return datasets


def load_one(path):
    ds = InvasiveDataset(root=path)
    return ds


if __name__ == '__main__':
    load_all(root='/mnt/data2/songyang/IBCI/data1/new_spike3')
