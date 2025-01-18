import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.proj_q(x).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        k = self.proj_k(x).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        v = self.proj_v(x).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        product = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(product, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)

        # combine heads
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, N, self.embed_dim)
        return self.proj_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfEncoderLayer(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64, num_heads=4):
        super().__init__()
        self.Attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.FeedForward = FeedForward(embed_dim, hidden_dim)
        self.Identity = nn.Identity()

    def forward(self, x):
        residual = self.Identity(x)
        a = residual + self.Attention(self.norm1(x))
        residual = self.Identity(a)
        a = residual + self.FeedForward(self.norm2(a))
        return a


class PositionEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=601):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        d_model = embed_dim if embed_dim % 2 == 0 else embed_dim + 1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0, dtype=torch.float)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # self.pe = pe

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :x.size(2)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, CH, time_step=100, L=1, num_classes=600, FFD=128):
        super().__init__()
        self.num_classes = num_classes

        self.layer = L
        self.Embedding = EEGEmbed(in_channels=CH)
        # self.Embedding = EmbedNet(in_channels=CH)
        with torch.no_grad():
            x = torch.randn(1, 1, time_step, CH)
            out = self.Embedding(x)
            embed_dim = out.shape[2]

            max_len = out.shape[1]
        self.embed_dim = embed_dim
        self.PE = PositionEncoding(embed_dim)
        self.cls_s = nn.Parameter(torch.rand(1, 1, self.embed_dim))

        self.blocks = nn.ModuleList([
            SelfEncoderLayer(embed_dim=self.embed_dim, hidden_dim=FFD, num_heads=4)
            for i in range(self.layer)])

        self.logits_eeg = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Flatten(start_dim=1),
            # nn.Linear(self.embed_dim * (max_len + 1), num_classes)
        )
        self.fc = nn.Linear(self.embed_dim * (max_len + 1), num_classes)

    def forward(self, x, visual=False):
        embed = self.Embedding(x)
        cls_token = self.cls_s.expand(x.shape[0], -1, -1)
        EEG_embed = torch.cat([cls_token, embed], dim=1)
        EEG_embed = self.PE(EEG_embed)

        for i, blk in enumerate(self.blocks):
            EEG_embed = blk(EEG_embed)

        feature = self.logits_eeg(EEG_embed)
        output = self.fc(feature)
        if visual:
            return feature, output
        return output


class EEGEmbed(nn.Module):
    def __init__(self, in_channels=59, kernLenght=64, F1=16, D=2, F2=64, dropoutRate=0.5):
        super().__init__()
        self.Chans = in_channels
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D * 2,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D * 2),
            nn.Conv2d(in_channels=self.F1 * self.D * 2,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 63:
            x = x.permute(0, 1, 3, 2)
        output = self.block1(x)
        output = self.block2(output)
        output = torch.reshape(output, (output.shape[0], output.shape[1], -1))
        return output.permute([0, 2, 1])


class ShallowEmbed(nn.Module):
    def __init__(self, emb_size=64):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 32, (1, 31), (1, 1)),
            # nn.Conv2d(64, 64, (32, 1), (1, 1)),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(63, 1),
                      groups=32,
                      bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(64, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            nn.BatchNorm2d(emb_size),
            nn.AvgPool2d((1, 8)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        if x.shape[-1] == 63:
            x = x.permute(0, 1, 3, 2)
        # x = x.permute(0, 1, 3, 2)
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


if __name__ == '__main__':
    pass
