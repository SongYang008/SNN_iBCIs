# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torchsummary import summary

from BackBone.SNN import Net


class FF(nn.Module):
    def __init__(self, model, in_channels=80, time_step=100, num_classes=3, d=128):
        super().__init__()
        self.net = model(in_channels=in_channels, time_step=time_step)
        feature_dim = self.net.get_feature_dim()
        self.pool = nn.AvgPool1d((10,))
        self.proj0 = nn.Linear(feature_dim, d)
        self.proj = nn.Linear(time_step // 10 * in_channels, d)
        self.cls = nn.Linear(2 * d, num_classes)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x, visual=False):

        f1, _ = self.net.forward(x, True)
        if x.dim() > 3:
            x = torch.squeeze(x, dim=1)
        f1 = self.proj0(f1)
        f1 = self.drop(f1)
        f2 = self.pool(x) * 10
        f2 = torch.flatten(f2, start_dim=1)
        f2 = self.proj(f2)
        f2 = self.drop(f2)
        f = torch.concat((f1, f2), dim=-1)
        out = self.cls(f)
        out = self.drop(out)
        if visual:
            return f, out
        return f1, f2, out


class FFwoP(nn.Module):
    def __init__(self, model, in_channels=80, time_step=100, num_classes=3):
        super().__init__()
        self.net = model(in_channels=in_channels, time_step=time_step)
        feature_dim = self.net.get_feature_dim()
        self.pool = nn.AvgPool1d((10,))
        # self.proj0 = nn.Linear(feature_dim, 128)
        # self.proj = nn.Linear(time_step // 10 * in_channels, 128)
        self.cls = nn.Linear(feature_dim + time_step // 10 * in_channels, num_classes)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x, visual=False):

        f1, _ = self.net.forward(x, True)
        if x.dim() > 3:
            x = torch.squeeze(x, dim=1)
        # f1 = self.proj0(f1)
        # f1 = self.drop(f1)
        f2 = self.pool(x)
        f2 = torch.flatten(f2, start_dim=1)
        # f2 = self.proj(f2)
        # f2 = self.drop(f2)
        f = torch.concat((f1, f2), dim=-1)
        out = self.cls(f)
        out = self.drop(out)
        if visual:
            return f, out
        return f1, f2, out


class FFwoPn(nn.Module):
    def __init__(self, model, in_channels=80, time_step=100, num_classes=3, d=128):
        super().__init__()
        self.net = model(in_channels=in_channels, time_step=time_step)
        feature_dim = self.net.get_feature_dim()
        self.pool = nn.AvgPool1d((10,))
        self.proj0 = nn.Linear(feature_dim, 128)
        # self.proj = nn.Linear(time_step // 10 * in_channels, 128)
        self.cls = nn.Linear(128 + time_step // 10 * in_channels, num_classes)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x, visual=False):

        f1, _ = self.net.forward(x, True)
        if x.dim() > 3:
            x = torch.squeeze(x, dim=1)
        f1 = self.proj0(f1)
        f1 = self.drop(f1)
        f2 = self.pool(x)
        f2 = torch.flatten(f2, start_dim=1)
        # f2 = self.proj(f2)
        # f2 = self.drop(f2)
        f = torch.concat((f1, f2), dim=-1)
        out = self.cls(f)
        out = self.drop(out)
        if visual:
            return f, out
        return f1, f2, out


class FFwoPd(nn.Module):
    def __init__(self, model, in_channels=80, time_step=100, num_classes=3, d=128):
        super().__init__()
        self.net = model(in_channels=in_channels, time_step=time_step)
        feature_dim = self.net.get_feature_dim()
        self.pool = nn.AvgPool1d((10,))
        # self.proj0 = nn.Linear(feature_dim, 128)
        self.proj = nn.Linear(time_step // 10 * in_channels, 128)
        self.cls = nn.Linear(128 + feature_dim, num_classes)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x, visual=False):

        f1, _ = self.net.forward(x, True)
        if x.dim() > 3:
            x = torch.squeeze(x, dim=1)
        # f1 = self.proj0(f1)
        # f1 = self.drop(f1)
        f2 = self.pool(x)
        f2 = torch.flatten(f2, start_dim=1)
        f2 = self.proj(f2)
        f2 = self.drop(f2)
        f = torch.concat((f1, f2), dim=-1)
        out = self.cls(f)
        out = self.drop(out)
        if visual:
            return f, out
        return f1, f2, out


if __name__ == '__main__':
    model = Net
    net = FF(model=model, in_channels=80, time_step=100).cuda()
    summary(net, (1, 80, 100))
