import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from openstl.modules import Attention, PreNorm
import math
import matplotlib.pyplot as plt

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                           for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

class TDBlock(nn.Module):
    def __init__(self, d_model, alphas=(0.05,)):
        super().__init__()
        self.d_model = d_model
        if isinstance(alphas, (int, float)):
            alphas = (alphas,)
        self.alphas = alphas
        self.M = len(alphas)
        if d_model % self.M != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by the number of alphas ({self.M})")
        
        self.group_dim = d_model // self.M
        self.td_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False)

        with torch.no_grad():
            weights = torch.zeros(d_model, 1, 3)
            for i, alpha in enumerate(self.alphas):
                start = i * self.group_dim
                end = (i + 1) * self.group_dim
                weights[start:end, 0, 0] = -alpha
                weights[start:end, 0, 1] = 2 * alpha
                weights[start:end, 0, 2] = -alpha
            self.td_conv.weight.copy_(weights)
        
        self.td_conv.weight.requires_grad = False

    def forward(self, x):
        identity = x
        x = x.permute(0, 2, 1)
        x = self.td_conv(x)
        x = x.permute(0, 2, 1)
        return identity + x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x_out

class FNOFFN(nn.Module):
    def __init__(self, dim, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(dim, dim, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x
        return identity + self.spectral_conv(x) + self.pointwise_conv(x)

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_g = nn.Linear(in_features, hidden_features)
        self.fc1_x = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1_g(x)) * self.fc1_x(x)
        x = self.fc2(x)
        return self.drop(x)

class GatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff, drop in self.layers:
            x = x + drop(attn(x))
            x = x + drop(ff(x))
        return self.norm(x)

class MTformerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 attn_dropout=0., drop_path=0.1, use_td=True, td_alpha=0.05,
                 num_patches_h=0, num_patches_w=0, use_fno=True):
        super().__init__()
        self.use_td = use_td
        self.use_fno = use_fno
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

        self.ts_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path)
        self.ts_space_transformer = GatedTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path)
        self.st_space_transformer = GatedTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path)
        self.st_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path)

        if self.use_td:
            self.td_ts = TDBlock(dim, td_alpha)
            self.td_st = TDBlock(dim, td_alpha)

        if self.use_fno:
            m1, m2 = max(num_patches_h // 2, 1), max(num_patches_w // 2, 1)
            self.fno_ts = FNOFFN(dim, m1, m2)
            self.fno_st = FNOFFN(dim, m1, m2)

    def forward(self, x):
        b, t, n, d = x.shape
        h, w = self.num_patches_h, self.num_patches_w

        x_ts = rearrange(x, 'b t n d -> (b n) t d')
        x_ts = self.ts_temporal_transformer(x_ts)
        if self.use_td:
            x_ts = self.td_ts(x_ts)

        x_ts = rearrange(x_ts, '(b n) t d -> (b t) n d', b=b)
        x_ts = self.ts_space_transformer(x_ts)
        if self.use_fno:
            x_ts = rearrange(x_ts, "(bt) (h w) d -> (bt) d h w", h=h, w=w)
            x_ts = self.fno_ts(x_ts)
            x_ts = rearrange(x_ts, "(bt) d h w -> (bt) (h w) d")

        x_st = rearrange(x_ts, '(b t) n d -> b t n d', b=b)
        x_st = rearrange(x_st, 'b t n d -> (b t) n d')
        x_st = self.st_space_transformer(x_st)
        if self.use_fno:
            x_st = rearrange(x_st, "(bt) (h w) d -> (bt) d h w", h=h, w=w)
            x_st = self.fno_st(x_st)
            x_st = rearrange(x_st, "(bt) d h w -> (bt) (h w) d")

        x_st = rearrange(x_st, '(b t) n d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')
        x_st = self.st_temporal_transformer(x_st)
        if self.use_td:
            x_st = self.td_st(x_st)

        return rearrange(x_st, '(b n) t d -> b t n d', b=b)

class MTformer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        h, w, ps = model_config['height'], model_config['width'], model_config['patch_size']
        self.num_h, self.num_w = h // ps, w // ps
        self.dim = model_config['dim']
        self.pre_seq = model_config['pre_seq']
        
        td_alpha = model_config.get('physics_alpha', 0.05)
        
        self.to_patch = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=ps, p2=ps),
            nn.Linear(model_config['num_channels'] * ps**2, self.dim)
        )

        self.blocks = nn.ModuleList([
            MTformerLayer(
                dim=self.dim, depth=model_config['depth'], heads=model_config['heads'],
                dim_head=model_config['dim_head'], mlp_dim=self.dim * model_config['scale_dim'],
                dropout=model_config['dropout'], attn_dropout=model_config['attn_dropout'],
                drop_path=model_config['drop_path'], 
                use_td=model_config.get('use_physics', True), td_alpha=td_alpha,
                num_patches_h=self.num_h, num_patches_w=self.num_w,
                use_fno=model_config.get('use_fno', True)
            ) for _ in range(model_config['Ndepth'])
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, model_config['num_channels'] * ps**2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.to_patch(x)
        pe = sinusoidal_embedding(T, self.dim).to(x.device)
        x = x + pe.unsqueeze(2) 

        for blk in self.blocks:
            x = blk(x)

        x = self.head(x).view(B, T, self.num_h, self.num_w, C, H//self.num_h, W//self.num_w)
        return x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
