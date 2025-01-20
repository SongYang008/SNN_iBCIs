# -*- coding: utf-8 -*-
"""
@Time： 2024/11/1 15:29
@Auth： S Yang
@File：SNN.py
@IDE：PyCharm
@Project: IBCI
@Motto：ABC(Always Be Coding)
"""
import numpy as np
import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import neuron
from torchsummary import summary
from einops.layers.torch import Rearrange


class Net(nn.Module):
    def __init__(self,
                 num_classes=3,
                 in_channels=80,
                 time_step=500,
                 groups=4,
                 cout=None,
                 spn='PLIF',
                 ):
        super().__init__()
        tau = math.exp(-0.5) + 1
        neuron_list = {
            'PLIF': neuron.ParametricLIFNode,
            'LIF': neuron.LIFNode,
            'QIF': neuron.QIFNode,
            'EIF': neuron.EIFNode,
            'Izhikevich': neuron.IzhikevichNode,
        }
        if spn not in neuron_list:
            raise TypeError('Spike neuron %s is not a choice.' % spn)
        spike_neuron = neuron_list[spn]
        if cout is None:
            cout = in_channels // 2

        self.TC = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=cout, kernel_size=(64,), padding=(16,), groups=1),
            nn.BatchNorm1d(num_features=cout),
            Rearrange('B C T -> T B C'),
            spike_neuron(tau, detach_reset=True, step_mode='m'),
            Rearrange('T B C -> B 1 C T')
        )
        # self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cout, kernel_size=(64,), padding=(16,),
        #                        groups=1)
        # self.bn1 = nn.BatchNorm1d(num_features=cout)
        # self.neuron1 = spike_neuron(tau, detach_reset=True, step_mode='m')
        self.SC = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(64, 1), padding=(32, 0)),
            Rearrange('B 1 C T -> B C T'),
            nn.BatchNorm1d(num_features=cout + 1),
            Rearrange('B C T -> T B C'),
            spike_neuron(tau, detach_reset=True, step_mode='m')
        )
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(64, 1), padding=(32, 0))
        self.bn3 = nn.BatchNorm1d(num_features=1 + cout)
        self.neuron3 = spike_neuron(tau, detach_reset=True, step_mode='m')
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear((cout + 1), num_classes)

    def forward(self, x, visual=False):
        if x.dim() == 4:
            N, _, C, T = x.shape
            x = torch.squeeze(x, dim=1)
        else:
            N, C, T = x.shape

        x = self.TC(x)
        out = self.SC(x)
        out = torch.mean(out, dim=0)

        feature = out
        out = self.fc(out)
        out = self.drop(out)
        if visual:
            return feature, out
        return out

    def get_feature_dim(self):
        return self.fc.in_features


if __name__ == '__main__':
    net = Net(in_channels=80, time_step=100).cuda()
    # torch.set_printoptions(threshold=torch.inf)
    # print(net.neuron1.v_threshold)
    summary(net, (1, 80, 100))
