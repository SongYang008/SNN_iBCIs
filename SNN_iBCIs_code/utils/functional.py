# -*- coding: utf-8 -*-
"""
@Time： 2024/6/9 17:40
@Auth： S Yang
@File：functional.py
@IDE：PyCharm
@Project: IBCI
@Motto：ABC(Always Be Coding)
"""
import random

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)  # Python 内置的随机数生成器
    np.random.seed(seed)  # Numpy 随机数生成器
    torch.manual_seed(seed)  # PyTorch CPU 随机数生成器
    torch.cuda.manual_seed(seed)  # PyTorch CUDA 随机数生成器（单个GPU）
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA 随机数生成器（所有GPU）
    torch.backends.cudnn.deterministic = True  # 确保CuDNN的确定性
    torch.backends.cudnn.benchmark = False  # 关闭优化，以保证每次结果一致


if __name__ == '__main__':
    pass
