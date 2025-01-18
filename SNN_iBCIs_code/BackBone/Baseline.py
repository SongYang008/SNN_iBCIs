# -*- coding: utf-8 -*-

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torchsummary import summary

from BackBone import EEGConFormer as EGC
from BackBone import EEGDeformer as EGD
# from BackBone.EEGDeformer import Conv2dWithConstraint, Transformer


class EEGNet(nn.Module):
    def __init__(self,
                 num_classes=3,
                 in_channels=80,
                 time_step=250,
                 kernel_size=64,
                 F1=8,
                 D=2,
                 F2=16,
                 drop_rate=0.5
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.channels = in_channels
        self.sample_point = time_step
        self.kernel = kernel_size
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.drop_rate = drop_rate
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernel // 2 - 1, self.kernel - self.kernel // 2, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=self.F1,
                      kernel_size=(1, self.kernel), stride=1, bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            nn.Conv2d(in_channels=self.F1, out_channels=self.F1 * self.D,
                      kernel_size=(self.channels, 1), groups=self.F1, bias=False),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.drop_rate)
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D, kernel_size=(1, 16), stride=1,
                      groups=self.F1 * self.D, bias=False, out_channels=self.F1 * self.D),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D, out_channels=self.F2, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=self.drop_rate),
            nn.Flatten(start_dim=1)
        )
        self.fc = nn.Linear(
            in_features=self.F2 * (self.sample_point // (4 * 8)),
            out_features=self.num_classes,
            bias=True
        )

    def forward(self, x, visual=False):
        # x = torch.unsqueeze(x, dim=1)
        output = self.block1(x)
        feature = self.block2(output)
        output = self.fc(feature)
        if visual:
            return feature, output
        return output

    def get_feature_dim(self):
        return self.fc.in_features


class DeepConvNet(torch.nn.Module):
    def __init__(self, num_classes=3, in_channels=80, time_step=100):
        super().__init__()

        self.model = nn.Sequential(
            # Conv2d(1, 25, kernel_size=(1,5),padding='VALID',bias=False),
            # Conv2d(25, 25, kernel_size=(2,1), padding='VALID',bias=False),
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.4),

            # Conv2d(25, 50, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.4),

            # Conv2d(50, 100, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.4),

            # Conv2d(100, 200, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.4),

            nn.Flatten(),

        )
        with torch.no_grad():
            input = torch.randn(1, 1, in_channels, time_step)
            out = self.model(input)
            in_features = out.shape[1]
        self.fc = nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x, visual=False):
        feature = self.model(x)
        out = self.fc(feature)
        if visual:
            return feature, out
        return out

    def get_feature_dim(self):
        return self.fc.in_features


class ShallowConvNet(nn.Module):
    def __init__(self, classes_num=3, in_channels=80, time_step=100, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.classes_num = classes_num
        n_ch1 = 40

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))
        self.layer1.eval()
        out = self.layer1(torch.zeros(1, 1, in_channels, time_step))
        out = torch.nn.functional.avg_pool2d(out, (1, 75), 15)
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]
        self.clf = nn.Linear(self.n_outputs, self.classes_num)

    def forward(self, x, visual=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [N, C, T] -> [N, 1, C, T]
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 75), 15)
        x = torch.log(x)
        x = torch.nn.functional.dropout(x)
        feature = x.view(x.size()[0], -1)
        out = self.clf(feature)
        if visual:
            return feature, out
        return out

    def get_feature_dim(self):
        return self.clf.in_features


class SVM(nn.Module):
    def __init__(self, num_classes=3, in_channels=80, time_step=8):
        super().__init__()

        self.border = nn.Linear(time_step * in_channels, num_classes)

    def forward(self, x):
        N, _, _, _ = x.size()
        x = x.view(N, -1)
        return self.border(x)


class EEGConFormer(nn.Module):
    def __init__(self, in_channels=80, time_step=100, num_classes=3):
        super().__init__()
        self.ch = in_channels
        self.transformer = EGC.Transformer(CH=in_channels, L=6, num_classes=3, time_step=time_step)

    def forward(self, x, visual=False):
        N, _, T, C = x.shape
        if C != self.ch:
            x = x.permute(0, 1, 3, 2)
        return self.transformer(x, visual)

    def get_feature_dim(self):
        return self.transformer.fc.in_features




class Deformer(nn.Module):
    def cnn_block(self, out_chan, kernel_size, num_chan):
        return nn.Sequential(
            EGD.Conv2dWithConstraint(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            EGD.Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def __init__(self, *, in_channels=80, time_step=100, temporal_kernel=11, num_kernel=64,
                 num_classes=3, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.):
        super().__init__()

        self.cnn_encoder = self.cnn_block(out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=in_channels)

        dim = int(0.5 * time_step)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = EGD.Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel,
        )

        L = self.get_hidden_size(input_size=dim, num_layer=depth)

        out_size = int(num_kernel * L[-1]) + int(num_kernel * depth)

        self.mlp_head = nn.Linear(out_size, num_classes)

    def forward(self, eeg, visual=False):
        # eeg: (b, chan, time)
        if eeg.dim() <= 3:
            eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time)
        x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time)

        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        x += self.pos_embedding
        feature = self.transformer(x)
        out = self.mlp_head(feature)
        if visual:
            return feature, out
        return out

    def get_padding(self, kernel):
        return 0, int(0.5 * (kernel - 1))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]

    def get_feature_dim(self):
        return self.mlp_head.in_features


class NAVSVM(nn.Module):
    def __init__(self, *, in_channels=80, time_step=100,
                 num_classes=3, ):
        super().__init__()
        self.pool = nn.AvgPool1d((10,))
        self.cls = nn.Linear(time_step // 10 * in_channels, num_classes)

    def forward(self, x):
        if x.dim() > 3:
            x = torch.squeeze(x, dim=1)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        out = self.cls(x)
        return out


if __name__ == '__main__':
    net = NAVSVM(in_channels=80, time_step=100).cuda()
    summary(net, (1, 80, 100))
    # net = DeepConvNet(in_channels=80, time_step=100).cuda()
    # summary(net, (1, 80, 100))
    # net = EEGConFormer(in_channels=80, time_step=100).cuda()
    # summary(net, (1, 80, 100))
    # net = EEGNet(in_channels=80, time_step=100).cuda()
    # summary(net, (1, 80, 100))
    #
    # flops, params = get_model_complexity_info(net, (1, 80, 100), as_strings=True, print_per_layer_stat=True)
    # print(f"FLOPs: {flops}")
    # print(f"Params: {params}")