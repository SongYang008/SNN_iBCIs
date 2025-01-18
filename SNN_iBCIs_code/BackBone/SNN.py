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
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cout, kernel_size=(64,), padding=(16,),
                               groups=1)
        self.bn1 = nn.BatchNorm1d(num_features=cout)
        self.neuron1 = spike_neuron(tau, detach_reset=True, step_mode='m')
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

        x = self.conv1(x)  # N C T
        x = self.bn1(x).permute(2, 0, 1)
        x = self.neuron1(x).permute(1, 2, 0)
        x = torch.unsqueeze(x, dim=1)

        x = self.conv3(x)  # N 1 C T
        x = torch.squeeze(x, dim=1)
        x = self.bn3(x).permute(2, 0, 1)
        out = self.neuron3(x)
        out = torch.mean(out, dim=0)

        feature = out
        out = self.fc(out)
        out = self.drop(out)
        if visual:
            return feature, out
        return out

    def get_feature_dim(self):
        return self.fc.in_features


class Netwoc1(nn.Module):
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
        # self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cout, kernel_size=(64,), padding=(16,),
        #                        groups=1)
        # self.bn1 = nn.BatchNorm1d(num_features=cout)
        # self.neuron1 = spike_neuron(tau, detach_reset=True, step_mode='m')
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(64, 1), padding=(32, 0))
        self.bn3 = nn.BatchNorm1d(num_features=1 + in_channels)
        self.neuron3 = spike_neuron(tau, detach_reset=True, step_mode='m')
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear((in_channels + 1), num_classes)

    def forward(self, x, visual=False):
        if x.dim() == 4:
            N, _, C, T = x.shape
            x = torch.squeeze(x, dim=1)
        else:
            N, C, T = x.shape

        # x = self.conv1(x)  # N C T
        # x = self.bn1(x).permute(2, 0, 1)
        # x = self.neuron1(x).permute(1, 2, 0)
        x = torch.unsqueeze(x, dim=1)

        x = self.conv3(x)  # N 1 C T
        x = torch.squeeze(x, dim=1)
        x = self.bn3(x).permute(2, 0, 1)
        out = self.neuron3(x)
        out = torch.mean(out, dim=0)

        feature = out
        out = self.fc(out)
        out = self.drop(out)
        if visual:
            return feature, out
        return out


class Netwoc2(nn.Module):
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
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cout, kernel_size=(64,), padding=(16,),
                               groups=1)
        self.bn1 = nn.BatchNorm1d(num_features=cout)
        self.neuron1 = spike_neuron(tau, detach_reset=True, step_mode='m')
        # self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(64, 1), padding=(32, 0))
        # self.bn3 = nn.BatchNorm1d(num_features=1 + cout)
        # self.neuron3 = spike_neuron(tau, detach_reset=True, step_mode='m')
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(cout, num_classes)

    def forward(self, x, visual=False):
        if x.dim() == 4:
            N, _, C, T = x.shape
            x = torch.squeeze(x, dim=1)
        else:
            N, C, T = x.shape

        x = self.conv1(x)  # N C T
        x = self.bn1(x).permute(2, 0, 1)
        x = self.neuron1(x).permute(1, 2, 0)
        # x = torch.unsqueeze(x, dim=1)

        # x = self.conv3(x)  # N 1 C T
        # x = torch.squeeze(x, dim=1)
        # x = self.bn3(x).permute(2, 0, 1)
        # out = self.neuron3(x)
        out = torch.mean(x, dim=-1)

        feature = out
        out = self.fc(out)
        out = self.drop(out)
        if visual:
            return feature, out
        return out


class NetwoPLIF(nn.Module):
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
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cout, kernel_size=(64,), padding=(16,),
                               groups=1)
        self.bn1 = nn.BatchNorm1d(num_features=cout)
        self.neuron1 = spike_neuron(tau, detach_reset=True, step_mode='m')
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(64, 1), padding=(32, 0))
        self.bn3 = nn.BatchNorm1d(num_features=1 + cout)
        # self.neuron3 = spike_neuron(tau, detach_reset=True, step_mode='m')
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear((cout + 1), num_classes)

    def forward(self, x, visual=False):
        if x.dim() == 4:
            N, _, C, T = x.shape
            x = torch.squeeze(x, dim=1)
        else:
            N, C, T = x.shape

        x = self.conv1(x)  # N C T
        x = self.bn1(x).permute(2, 0, 1)
        x = self.neuron1(x).permute(1, 2, 0)
        x = torch.unsqueeze(x, dim=1)

        x = self.conv3(x)  # N 1 C T
        x = torch.squeeze(x, dim=1)
        x = self.bn3(x).permute(2, 0, 1)
        # out = self.neuron3(x)
        out = torch.mean(x, dim=0)

        feature = out
        out = self.fc(out)
        out = self.drop(out)
        if visual:
            return feature, out
        return out


class NetReLU(nn.Module):
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
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cout, kernel_size=(64,), padding=(16,),
                               groups=1)
        self.bn1 = nn.BatchNorm1d(num_features=cout)
        self.neuron1 = spike_neuron(tau, detach_reset=True, step_mode='m')
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(64, 1), padding=(32, 0))
        self.bn3 = nn.BatchNorm1d(num_features=1 + cout)
        self.neuron3 = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear((cout + 1), num_classes)

    def forward(self, x, visual=False):
        if x.dim() == 4:
            N, _, C, T = x.shape
            x = torch.squeeze(x, dim=1)
        else:
            N, C, T = x.shape

        x = self.conv1(x)  # N C T
        x = self.bn1(x).permute(2, 0, 1)
        x = self.neuron1(x).permute(1, 2, 0)
        x = torch.unsqueeze(x, dim=1)

        x = self.conv3(x)  # N 1 C T
        x = torch.squeeze(x, dim=1)
        x = self.bn3(x).permute(2, 0, 1)
        out = self.neuron3(x)
        out = torch.mean(out, dim=0)

        feature = out
        out = self.fc(out)
        out = self.drop(out)
        if visual:
            return feature, out
        return out


class NetwMCPLIF(nn.Module):
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
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cout, kernel_size=(64,), padding=(16,),
                               groups=1)
        self.bn1 = nn.BatchNorm1d(num_features=cout)
        self.neuron1 = spike_neuron(tau, detach_reset=True, step_mode='m')
        w = torch.zeros((cout,))
        torch.fill(w, 0.5)
        w = nn.Parameter(w)
        self.neuron1.w = w
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(64, 1), padding=(32, 0))
        self.bn3 = nn.BatchNorm1d(num_features=1 + cout)
        self.neuron3 = spike_neuron(tau, detach_reset=True, step_mode='m')
        w = torch.zeros((cout + 1,))
        torch.fill(w, 0.5)
        w = nn.Parameter(w)
        self.neuron3.w = w
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear((cout + 1), num_classes)

    def forward(self, x, visual=False):
        if x.dim() == 4:
            N, _, C, T = x.shape
            x = torch.squeeze(x, dim=1)
        else:
            N, C, T = x.shape

        x = self.conv1(x)  # N C T
        x = self.bn1(x).permute(2, 0, 1)
        x = self.neuron1(x).permute(1, 2, 0)
        x = torch.unsqueeze(x, dim=1)

        x = self.conv3(x)  # N 1 C T
        x = torch.squeeze(x, dim=1)
        x = self.bn3(x).permute(2, 0, 1)
        out = self.neuron3(x)
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
    net = NetwMCPLIF(in_channels=80, time_step=100).cuda()
    # torch.set_printoptions(threshold=torch.inf)
    # print(net.neuron1.v_threshold)
    summary(net, (1, 80, 100))
