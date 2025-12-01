import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os
from openstl.utils import measure_throughput
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from openstl.modules import Attention, PreNorm, FeedForward
import math


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
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOFFN(nn.Module):
    def __init__(self, dim, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(dim, dim, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x
        x_spectral = self.spectral_conv(x)
        x_pointwise = self.pointwise_conv(x)
        return identity + x_spectral + x_pointwise


class SpatialDiffusionFFN(nn.Module):
    def __init__(self, d_model, num_nodes=1, alpha=0.05):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = max(1, int(num_nodes))
        self.alpha = alpha

        self.st_conv = nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1), groups=d_model)

        with torch.no_grad():
            self.st_conv.weight.data.fill_(0.0)
            self.st_conv.weight.data[:, 0, 1, 1] = 4.0 * self.alpha
            self.st_conv.weight.data[:, 0, 0, 1] = -self.alpha
            self.st_conv.weight.data[:, 0, 2, 1] = -self.alpha
            self.st_conv.weight.data[:, 0, 1, 0] = -self.alpha
            self.st_conv.weight.data[:, 0, 1, 2] = -self.alpha
            if self.st_conv.bias is not None:
                self.st_conv.bias.data.fill_(0.0)

    def forward(self, x):
        identity = x
        batch, seq_len, dim = x.shape
        device = x.device

        if self.num_nodes > 1 and (batch % self.num_nodes == 0):
            b = batch // self.num_nodes
            x_resh = x.view(b, self.num_nodes, seq_len, dim)
            x_conv = x_resh.permute(0, 3, 1, 2).contiguous()
            x_out = self.st_conv(x_conv)
            x_out = x_out.permute(0, 2, 3, 1).contiguous()
            x_out = x_out.view(batch, seq_len, dim)
            return identity + x_out
        else:
            x_t = x.permute(0, 2, 1).unsqueeze(-1)
            x_t = self.st_conv(x_t).squeeze(-1)
            x_t = x_t.permute(0, 2, 1)
            return identity + x_t


class GatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)


class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MTformerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 attn_dropout=0., drop_path=0.1, use_physics=True, physics_alpha=0.05,
                 num_patches_h=0, num_patches_w=0, use_fno=True):
        super(MTformerLayer, self).__init__()
        self.use_physics = use_physics
        self.use_fno = use_fno
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

        if self.num_patches_h == 0 or self.num_patches_w == 0:
            raise ValueError(f"Invalid patch dimensions: num_patches_h={self.num_patches_h}, num_patches_w={self.num_patches_w}")

        self.num_patches = self.num_patches_h * self.num_patches_w

        self.ts_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head,
                                                       mlp_dim, dropout, attn_dropout, drop_path)
        self.ts_space_transformer = GatedTransformer(dim, depth, heads, dim_head,
                                                     mlp_dim, dropout, attn_dropout, drop_path)
        self.st_space_transformer = GatedTransformer(dim, depth, heads, dim_head,
                                                     mlp_dim, dropout, attn_dropout, drop_path)
        self.st_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head,
                                                        mlp_dim, dropout, attn_dropout, drop_path)

        if self.use_physics:
            self.physics_ffn_ts_temporal = SpatialDiffusionFFN(dim, num_nodes=self.num_patches, alpha=physics_alpha)
            self.physics_ffn_st_temporal = SpatialDiffusionFFN(dim, num_nodes=self.num_patches, alpha=physics_alpha)

        if self.use_fno:
            modes1 = max(self.num_patches_h // 2, 1)
            modes2 = max(self.num_patches_w // 2, 1)
            self.fno_ts_space = FNOFFN(dim, modes1, modes2)
            self.fno_st_space = FNOFFN(dim, modes1, modes2)

    def forward(self, x):
        b, t, n, d = x.shape
        expected_patches = self.num_patches_h * self.num_patches_w
        if n != expected_patches:
            raise ValueError(f"Input patch count mismatch: expected {expected_patches}, got {n}")

        x_ts, x_ori = x, x

        x_ts = rearrange(x_ts, 'b t n d -> b n t d')
        x_ts = rearrange(x_ts, 'b n t d -> (b n) t d')
        x_ts = self.ts_temporal_transformer(x_ts)

        if self.use_physics:
            x_ts = self.physics_ffn_ts_temporal(x_ts)

        x_ts = rearrange(x_ts, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d')
        x_ts = self.ts_space_transformer(x_ts)

        if self.use_fno:
            x_ts = rearrange(x_ts, "(b t) (h w) d -> (b t) d h w", b=b, t=t, h=self.num_patches_h, w=self.num_patches_w)
            x_ts = self.fno_ts_space(x_ts)
            x_ts = rearrange(x_ts, "(b t) d h w -> (b t) (h w) d", b=b, t=t, h=self.num_patches_h, w=self.num_patches_w)

        x_ts = rearrange(x_ts, '(b t) n d -> b t n d', b=b)

        x_st, x_ori = x_ts, x_ts

        x_st = rearrange(x_st, 'b t n d -> (b t) n d')
        x_st = self.st_space_transformer(x_st)

        if self.use_fno:
            x_st = rearrange(x_st, "(b t) (h w) d -> (b t) d h w", b=b, t=t, h=self.num_patches_h, w=self.num_patches_w)
            x_st = self.fno_st_space(x_st)
            x_st = rearrange(x_st, "(b t) d h w -> (b t) (h w) d", b=b, t=t, h=self.num_patches_h, w=self.num_patches_w)

        x_st = rearrange(x_st, '(b t) ... -> b t ...', b=b)
        x_st = x_st.permute(0, 2, 1, 3)
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')
        x_st = self.st_temporal_transformer(x_st)

        if self.use_physics:
            x_st = self.physics_ffn_st_temporal(x_st)

        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> b t n d', b=b)

        return x_st


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')


class MTformer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        self.image_height = model_config['height']
        self.image_width = model_config['width']
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size

        if self.num_patches_h == 0 or self.num_patches_w == 0:
            raise ValueError(
                f"Invalid patch dimensions: height={self.image_height}, width={self.image_width}, "
                f"patch_size={self.patch_size}, num_patches_h={self.num_patches_h}, num_patches_w={self.num_patches_w}"
            )

        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.num_channels = model_config['num_channels']

        self.dim = model_config['dim']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.depth = model_config['depth']
        self.Ndepth = model_config['Ndepth']

        self.use_physics = model_config.get('use_physics', True)
        self.physics_alpha = model_config.get('physics_alpha', 0.05)
        self.use_fno = model_config.get('use_fno', True)

        self.patch_dim = self.num_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)',
                      p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
        )

        self.blocks = nn.ModuleList([
            MTformerLayer(
                dim=self.dim,
                depth=self.depth,
                heads=self.heads,
                dim_head=self.dim_head,
                mlp_dim=self.dim * self.scale_dim,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                drop_path=self.drop_path,
                use_physics=self.use_physics,
                physics_alpha=self.physics_alpha,
                use_fno=self.use_fno,
                num_patches_h=self.num_patches_h,
                num_patches_w=self.num_patches_w
            ) for _ in range(self.Ndepth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        if H != self.image_height or W != self.image_width:
            raise ValueError(f"Input dimensions mismatch: expected H={self.image_height}, W={self.image_width}, got H={H}, W={W}")

        x = self.to_patch_embedding(x)

        _, _, n, d = x.shape
        if n != self.num_patches:
            raise ValueError(f"Patch count mismatch: expected {self.num_patches}, got {n}")
        pos = sinusoidal_embedding(T * n, d).view(1, T, n, d).to(x.device)
        x = x + pos

        for blk in self.blocks:
            x = blk(x)

        x = self.mlp_head(x.reshape(-1, d))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w,
                   C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        return x