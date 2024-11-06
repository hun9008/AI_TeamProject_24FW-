import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
from tqdm import tqdm

import itertools
from torchinfo import summary

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ConvNorm, self).__init__()
        self.linear = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x
    
class Stem16(nn.Module):
    def __init__(self):
        super(Stem16, self).__init__()
        self.conv1 = ConvNorm(3, 32)
        self.act1 = nn.Hardswish()
        self.conv2 = ConvNorm(32, 64)
        self.act2 = nn.Hardswish()
        self.conv3 = ConvNorm(64, 128)
        self.act3 = nn.Hardswish()
        self.conv4 = ConvNorm(128, 256)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.conv4(x)
        return x
    
class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # Flatten if the input is 3D to apply BatchNorm1d
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(B * N, C)  # view에서 reshape로 변경
            x = self.bn(self.linear(x))
            x = x.reshape(B, N, -1)  # 원래 모양으로 다시 reshape
        else:
            x = self.bn(self.linear(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, attn_ratio=2):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        inner_dim = head_dim * num_heads * 3  
        self.qkv = LinearNorm(dim, inner_dim)
        
        self.proj = nn.Sequential(
            nn.Hardswish(),
            LinearNorm(dim, dim)  
        )

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
    
class LevitMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(LevitMlp, self).__init__()
        self.ln1 = LinearNorm(in_features, hidden_features)
        self.act = nn.Hardswish()
        self.drop = nn.Dropout(p=0.0, inplace=False)
        self.ln2 = LinearNorm(hidden_features, out_features)

    def forward(self, x):
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.ln2(x)
        return x
    
class LevitBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2):
        super(LevitBlock, self).__init__()
        self.attn = Attention(dim, num_heads)
        self.drop_path1 = nn.Identity()
        self.mlp = LevitMlp(dim, dim * mlp_ratio, dim)
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(x))
        x = x + self.drop_path2(self.mlp(x))
        return x
    
class AttentionDownsample(nn.Module):
    def __init__(self, dim, out_dim, num_heads, attn_ratio=2):
        super(AttentionDownsample, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        inner_dim = dim * attn_ratio * num_heads
        self.kv = LinearNorm(dim, inner_dim)
        
        # Downsample 적용
        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )
        
        # proj 부분 수정
        self.proj = nn.Sequential(
            nn.Hardswish(),
            LinearNorm(dim, out_dim)  # dim과 out_dim이 올바르게 설정되었는지 확인
        )

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # N의 제곱근을 통해 H, W 결정 (e.g., N=196 -> H=W=14)
        x = x.reshape(B, C, H, W)  # view에서 reshape로 변경하여 연속된 메모리 레이아웃 문제 해결

        kv = self.kv(x.flatten(2).transpose(1, 2))  # kv 생성
        q = self.q(x)  # 다운샘플링된 q 생성
        
        # 다운샘플링 후 적절한 차원으로 변환
        q = q.reshape(B, -1, C)  # view에서 reshape로 변경
        x = self.proj(q)
        return x
    
class LevitDownsample(nn.Module):
    def __init__(self, dim, out_dim, num_heads, attn_ratio=2):
        super(LevitDownsample, self).__init__()
        self.attn_downsample = AttentionDownsample(dim, out_dim, num_heads, attn_ratio)
        self.mlp = LevitMlp(out_dim, out_dim * attn_ratio, out_dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = self.attn_downsample(x)
        x = self.drop_path(self.mlp(x))
        return x
    
class LevitStage(nn.Module):
    def __init__(self, dim, out_dim, num_heads, num_blocks, downsample=True):
        super(LevitStage, self).__init__()
        self.downsample = LevitDownsample(dim, out_dim, num_heads) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[LevitBlock(out_dim, num_heads) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x

class NormLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.0):
        super(NormLinear, self).__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.drop = nn.Dropout(p=dropout_prob, inplace=False)
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.drop(x)
        x = self.linear(x)
        return x

class LevitDistilled(nn.Module):
    def __init__(self, num_classes=37):
        super(LevitDistilled, self).__init__()
        
        # Stem layer
        self.stem = Stem16()
        
        # Stages: LevitStage layers with the adjusted number of blocks and output dimensions
        self.stages = nn.Sequential(
            LevitStage(dim=256, out_dim=256, num_heads=4, num_blocks=3, downsample=False),
            LevitStage(dim=256, out_dim=384, num_heads=6, num_blocks=3, downsample=True),
            LevitStage(dim=384, out_dim=512, num_heads=8, num_blocks=2, downsample=True)
        )
        
        # Output heads
        self.head = NormLinear(in_features=512, out_features=num_classes, dropout_prob=0.0)
        self.head_dist = NormLinear(in_features=512, out_features=num_classes, dropout_prob=0.0)

    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)
        x = self.stages(x)
        out = self.head(x.mean(dim=1))
        out_dist = self.head_dist(x.mean(dim=1))
        return out, out_dist
    
def test():
    model = LevitDistilled()
    summary(model, input_size=(32, 3, 224, 224))

if __name__ == '__main__':
    test()