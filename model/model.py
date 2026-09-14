import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
import numpy as np
from functools import lru_cache



 
class LightWaveletUpsample(nn.Module):
    """
    轻量小波上采样：用编码器保存的高频做逆变换重建
    - 门控过滤噪声高频
    - hf_scale 初始值小，训练初期不被噪声干扰
    """

    def __init__(self, in_dim, out_dim, hf_dim):
        """
        Args:
            in_dim:  输入通道数（解码器深层特征的通道数）
            out_dim: 输出通道数（上采样后目标通道数）
            hf_dim:  高频子带通道数（=编码器对应层的dim）
        """
        super().__init__()
        self.idwt = IDWT_2D()  # 用代码2自己的 IDWT_2D

        # 深层特征 → 低频 LL
        self.ll_proj = nn.Sequential(
            nn.Conv2d(in_dim, hf_dim, kernel_size=1),
            nn.GroupNorm(2, hf_dim),
            nn.GELU()
        )

        # 高频过滤门控：学习哪些高频是边缘、哪些是噪声
        self.hf_gate = nn.Sequential(
            nn.Conv2d(hf_dim * 3, hf_dim, kernel_size=3, padding=1),
            nn.GroupNorm(2, hf_dim),
            nn.GELU(),
            nn.Conv2d(hf_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 全局高频强度控制（初始化小，慢慢学）
        self.hf_scale = nn.Parameter(torch.ones(1) * 0.1)

        # 重建后调整到目标通道数
        self.out_proj = nn.Sequential(
            nn.Conv2d(hf_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_dim),
            nn.ReLU()
        )

    def forward(self, x, hf_subbands):
        """
        x:            [B, in_dim, H, W]     解码器深层特征
        hf_subbands:  (LH, HL, HH)          各 [B, hf_dim, H, W]
        输出:         [B, out_dim, 2H, 2W]   分辨率翻倍
        """
        lh, hl, hh = hf_subbands

        # 深层特征投影为 LL
        ll = self.ll_proj(x)  # [B, hf_dim, H, W]

        # 过滤高频
        hf_cat = torch.cat([lh, hl, hh], dim=1)        # [B, 3*hf_dim, H, W]
        gate = self.hf_gate(hf_cat)                      # [B, 1, H, W]
        scale = torch.sigmoid(self.hf_scale)              # 标量

        lh_filtered = lh * gate * scale
        hl_filtered = hl * gate * scale
        hh_filtered = hh * gate * scale

        # 逆小波变换（分辨率×2）
        reconstructed = self.idwt(ll, lh_filtered, hl_filtered, hh_filtered)
        # [B, hf_dim, 2H, 2W]

        out = self.out_proj(reconstructed)  # [B, out_dim, 2H, 2W]
        return out






# 小波
# ==================== 基础小波变换 ====================
class DWT_2D(nn.Module):
    """
    2D Haar离散小波变换
    适合小目标：Haar支撑长度最短，不会模糊细节
    """
    def __init__(self):
        super().__init__()
        self.requires_grad = False
    
    def forward(self, x):
        """
        输入: [B, C, H, W]
        输出: [B, C, H/2, W/2] × 4 (LL, LH, HL, HH)
        或者: [B, 4C, H/2, W/2] (拼接版本)
        """
        # 按行列分离
        x01 = x[:, :, 0::2, :] / 2  # 偶数行
        x02 = x[:, :, 1::2, :] / 2  # 奇数行
        
        x1 = x01[:, :, :, 0::2]  # 偶行偶列
        x2 = x02[:, :, :, 0::2]  # 奇行偶列
        x3 = x01[:, :, :, 1::2]  # 偶行奇列
        x4 = x02[:, :, :, 1::2]  # 奇行奇列
        
        # 四个子带
        LL = x1 + x2 + x3 + x4  # 低频（平均/近似）
        LH = -x1 - x2 + x3 + x4  # 水平边缘
        HL = -x1 + x2 - x3 + x4  # 垂直边缘
        HH = x1 - x2 - x3 + x4   # 对角边缘
        
        return LL, LH, HL, HH

class IDWT_2D(nn.Module):
    """2D Haar逆离散小波变换"""
    def __init__(self):
        super().__init__()
        self.requires_grad = False
    
    def forward(self, LL, LH, HL, HH):
        """
        输入: 四个子带 [B, C, H, W]
        输出: [B, C, 2H, 2W]
        """
        B, C, H, W = LL.shape
        
        # 归一化
        LL, LH, HL, HH = LL / 2, LH / 2, HL / 2, HH / 2
        
        # 重建
        out = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
        
        out[:, :, 0::2, 0::2] = LL - LH - HL + HH  # 偶行偶列
        out[:, :, 1::2, 0::2] = LL - LH + HL - HH  # 奇行偶列
        out[:, :, 0::2, 1::2] = LL + LH - HL - HH  # 偶行奇列
        out[:, :, 1::2, 1::2] = LL + LH + HL + HH  # 奇行奇列
        
        return out




# ==================== 高频注意力模块（核心：小目标增强） ====================
class WaveletHighFreqAttention(nn.Module):
    """
    基于小波高频的注意力模块
    核心思想：小目标边缘占比高，高频信息强
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        
        # 高频特征处理
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GroupNorm(4, channels),
            nn.GELU()
        )
        
        # 空间注意力（定位小目标）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
        # 通道注意力（选择重要特征）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 高频增强系数
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        输入输出: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 确保尺寸为偶数
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # 小波分解
        LL, LH, HL, HH = self.dwt(x)
        
        # 合并高频分量
        high_freq = torch.cat([LH, HL, HH], dim=1)  # [B, 3C, H/2, W/2]
        
        # 高频特征增强
        high_feat = self.high_freq_conv(high_freq)  # [B, C, H/2, W/2]
        
        # 计算注意力
        spatial_weight = self.spatial_attn(high_feat)  # [B, 1, H/2, W/2]
        channel_weight = self.channel_attn(high_feat)  # [B, C, 1, 1]
        
        # 增强高频
        attn = spatial_weight * channel_weight  # [B, C, H/2, W/2]
        
        LH = LH * (1 + self.gamma * attn)
        HL = HL * (1 + self.gamma * attn)
        HH = HH * (1 + self.gamma * attn)
        
        # 逆变换重建
        out = self.idwt(LL, LH, HL, HH)
        
        # 移除padding
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        
        return out + x  # 残差连接





class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias)
        self.norm = nn.GroupNorm(num_groups=2, num_channels=out_ch)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        return out


class AdaptiveDilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=True, bias=False, dilation_range=[1]):
        super().__init__()
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]

        self.conv_options = nn.ModuleList([
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      dilation=d,
                      padding=d * (kernel_size - 1) // 2,
                      groups=groups,
                      bias=bias)
            for d in dilation_range
        ])

        self.weights = nn.Parameter(torch.zeros(len(dilation_range)))
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        weights = F.softmax(self.weights, dim=0)
        output = sum(weights[i] * conv(x) for i, conv in enumerate(self.conv_options))
        return self.act(self.norm(output))


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, norm=nn.BatchNorm2d):
        super().__init__()
        pad_size = kernel_size // 2

        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=pad_size)
        self.residual = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.residual(shortcut)
        return x

class Conv_Stem(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        self.conv1 = BasicBlock(in_ch, out_ch // 2, kernel_size=kernel_size)

        self.conv2_rgb = AdaptiveDilatedConv2d(out_ch // 2, out_ch, kernel_size=kernel_size, stride=2, dilation_range=[1])
       
        self.conv2_nir = AdaptiveDilatedConv2d(out_ch // 2, out_ch, kernel_size=kernel_size, stride=2,dilation_range=[1])

        self._initialize_sobel(out_ch)
        self.edge_fusion = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self._init_fusion_weights()

    def _init_fusion_weights(self):
        nn.init.xavier_normal_(self.edge_fusion.weight)
        nn.init.constant_(self.edge_fusion.bias, 0)

    def _initialize_sobel(self, channels):
        sobel_x_kernel = torch.tensor([
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        ], dtype=torch.float32) / 4.0

        sobel_y_kernel = torch.tensor([
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        ], dtype=torch.float32) / 4.0

        self.sobel_x = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=1, bias=False, groups=channels)

        self.sobel_y = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=1, bias=False, groups=channels)

        with torch.no_grad():
            self.sobel_x.weight.copy_(sobel_x_kernel.expand(channels, 1, 3, 3))
            self.sobel_y.weight.copy_(sobel_y_kernel.expand(channels, 1, 3, 3))

    def edge_enhance(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        edge = self.edge_fusion(edge)
        edge = F.relu(edge)
        return edge

    def forward(self, x, modality):
        x_2 = self.conv1(x)

        # ===【修改3】删除depth条件分支===
        if modality in ['rgb']:
            x = self.conv2_rgb(x_2)
        # elif modality in ['depth']:
        #     x = self.conv2_depth(x_2)
        elif modality in ['nir']:
            x = self.conv2_nir(x_2)

        edge = self.edge_enhance(x)
        x = x + edge

        return x_2, x


class CoordAtt(nn.Module):
    """
    2D Coordinate Attention with dual-path fusion
    """

    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()

        self.inp = inp
        self.oup = oup

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(mip)
        self.relu2 = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        if inp != oup:
            self.channel_conv = nn.Conv2d(inp, oup, kernel_size=1)
        else:
            self.channel_conv = nn.Identity()

    def forward(self, g, x):
        b, c, h, w = x.size()

        g_h = F.adaptive_avg_pool2d(g, (h, 1))
        g_w = F.adaptive_avg_pool2d(g, (1, w))
        g_w = g_w.permute(0, 1, 3, 2)

        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        x_w = x_w.permute(0, 1, 3, 2)

        g_y = torch.cat([g_h, g_w], dim=2)
        g_y = self.conv1(g_y)
        g_y = self.bn1(g_y)
        g_y = self.relu1(g_y)

        x_y = torch.cat([x_h, x_w], dim=2)
        x_y = self.conv2(x_y)
        x_y = self.bn2(x_y)
        x_y = self.relu2(x_y)

        g_h, g_w = torch.split(g_y, [h, w], dim=2)
        g_w = g_w.permute(0, 1, 3, 2)

        x_h, x_w = torch.split(x_y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = (g_h + x_h) / 2
        a_w = (g_w + x_w) / 2

        a_h = torch.sigmoid(self.conv_h(a_h))
        a_w = torch.sigmoid(self.conv_w(a_w))

        x = self.channel_conv(x)
        out = x * a_h * a_w

        return out


class SEBlock(nn.Module):

    def __init__(self, in_ch, ratio=4, act=nn.ReLU):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // ratio, kernel_size=1),
            act(),
            nn.Conv2d(in_ch // ratio, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = self.excitation(out)
        return x * out

class AMBConv2D_NIR(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, se=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        expanded = expansion * in_ch
        self.use_se = se
        self.expand_proj = nn.Identity() if (expansion == 1) else ConvNormAct(in_ch, expanded, kernel_size=1, padding=0)
        self.depthwise = AdaptiveDilatedConv2d(expanded, expanded, kernel_size=kernel_size,stride=stride, dilation_range=[1], groups=expanded)

        if self.use_se:
            self.se = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.expand_proj(x)
        x = self.depthwise(x)
        if self.use_se:
            x = self.se(x)
        x = self.pointwise(x)
        x = x.permute(0, 2, 3, 1)
        return x


class AMBConv2D_RGB(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, se=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        expanded = expansion * in_ch
        self.use_se = se
        self.expand_proj = nn.Identity() if (expansion == 1) else ConvNormAct(in_ch, expanded, kernel_size=1, padding=0)
        self.depthwise = AdaptiveDilatedConv2d(expanded, expanded, kernel_size=kernel_size,stride=stride, dilation_range=[1], groups=expanded)

        if self.use_se:
            self.se = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.expand_proj(x)
        x = self.depthwise(x)
        if self.use_se:
            x = self.se(x)
        x = self.pointwise(x)
        x = x.permute(0, 2, 3, 1)
        return x




class MBConv(nn.Module):

    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, se=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        expanded = expansion * in_ch
        self.se = se

        self.expand_proj = nn.Identity() if (expansion == 1) else ConvNormAct(in_ch, expanded, kernel_size=1, padding=0)
        self.depthwise = ConvNormAct(expanded, expanded, kernel_size=kernel_size, stride=stride, padding=padding,groups=expanded)

        if self.se:
            self.se = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.expand_proj(x)
        x = self.depthwise(x)
        if self.se:
            x = self.se(x)
        x = self.pointwise(x)
        x = x.permute(0, 2, 3, 1)
        return x


def window_partition_2d(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], C)
    return windows


def window_reverse_2d(windows, window_size, B, H, W):
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_window_size_2d(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class CrossWindowAttention2D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        B_, N, C = x.shape

        q = self.query(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class WindowAttention2D(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class MultiModalSwinTransformerBlock2D(nn.Module):
    def __init__(self, dim=64, num_heads=8, window_size=(7, 7), shift_size=(0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,use_wavelet_attn=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm_m1_1 = norm_layer(dim)
        self.norm_m3_1 = norm_layer(dim)

        self.self_attn_m1 = WindowAttention2D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.self_attn_m3 = WindowAttention2D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


        self.cross_attn_13 = CrossWindowAttention2D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_m1_2 = norm_layer(dim)

        self.norm_m3_2 = norm_layer(dim)

        self.mlp_m1 = AMBConv2D_RGB(in_ch=dim, out_ch=dim)
        self.mlp_m3 = AMBConv2D_NIR(in_ch=dim, out_ch=dim)


        # 添加小波注意力14
        self.use_wavelet_attn = use_wavelet_attn
        if use_wavelet_attn:
            self.wavelet_attn = WaveletHighFreqAttention(dim)

    def forward_part1(self, m1, m3, mask_matrix, cross=False):
        B, H, W, C = m1.shape
        window_size, shift_size = get_window_size_2d((H, W), self.window_size, self.shift_size)

        m1 = self.norm_m1_1(m1)
        m3 = self.norm_m3_1(m3)

        # Padding
        pad_l = pad_t = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        m1 = F.pad(m1, (0, 0, pad_l, pad_r, pad_t, pad_b))
        m3 = F.pad(m3, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = m1.shape

        # Cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_m1 = torch.roll(m1, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            shifted_m3 = torch.roll(m3, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_m1 = m1

            shifted_m3 = m3
            attn_mask = None

        # Partition windows
        m1_windows = window_partition_2d(shifted_m1, window_size)
        m3_windows = window_partition_2d(shifted_m3, window_size)

        # Window attention
        attn_windows_m1 = self.self_attn_m1(m1_windows, mask=attn_mask)
        attn_windows_m3 = self.self_attn_m3(m3_windows, mask=attn_mask)

        # Merge windows
        attn_windows_m1 = attn_windows_m1.view(-1, *(window_size + (C,)))
        attn_windows_m3 = attn_windows_m3.view(-1, *(window_size + (C,)))

        shifted_m1 = window_reverse_2d(attn_windows_m1, window_size, B, Hp, Wp)
        shifted_m3 = window_reverse_2d(attn_windows_m3, window_size, B, Hp, Wp)

        # Reverse cyclic shift
        if any(i > 0 for i in shift_size):
            m1 = torch.roll(shifted_m1, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
            m3 = torch.roll(shifted_m3, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            m1 = shifted_m1
            # m2 = shifted_m2  # 删除
            m3 = shifted_m3

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            m1 = m1[:, :H, :W, :].contiguous()
            m3 = m3[:, :H, :W, :].contiguous()

        return m1, m3

    def forward_part2(self, m1, m3):
        m1 = self.drop_path(self.mlp_m1(self.norm_m1_2(m1)))
        # m2 = self.drop_path(self.mlp_m2(self.norm_m2_2(m2)))  # 删除
        m3 = self.drop_path(self.mlp_m3(self.norm_m3_2(m3)))
        return m1, m3

    def forward(self, m1, m3, mask_matrix, cross=False):
        # Attention block
        m1_shortcut, m3_shortcut = m1, m3
        m1, m3 = self.forward_part1(m1, m3, mask_matrix, cross)
        m1 = m1_shortcut + self.drop_path(m1)
        # m2 = m2_shortcut + self.drop_path(m2)  # 删除
        m3 = m3_shortcut + self.drop_path(m3)

        # MLP block
        m1_shortcut, m3_shortcut = m1, m3
        m1, m3 = self.forward_part2(m1, m3)
        m1 = m1_shortcut + m1
        # m2 = m2_shortcut + m2  # 删除
        m3 = m3_shortcut + m3


        # 在最后添加小波增强
        if self.use_wavelet_attn:
        # 转换维度 [B, H, W, C] -> [B, C, H, W]
            m1_conv = m1.permute(0, 3, 1, 2)
            m3_conv = m3.permute(0, 3, 1, 2)
            
            m1_conv = self.wavelet_attn(m1_conv)
            m3_conv = self.wavelet_attn(m3_conv)
            
            m1 = m1_conv.permute(0, 2, 3, 1)
            m3 = m3_conv.permute(0, 2, 3, 1)

        return m1, m3


class BottleneckBlock(nn.Module):
    def __init__(self, dim=64, num_heads=8, window_size=(7, 7), shift_size=(0, 0), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention2D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MBConv(in_ch=dim, out_ch=dim)

    def forward_part1(self, x, mask_matrix):
        B, H, W, C = x.shape
        window_size, shift_size = get_window_size_2d((H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition_2d(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse_2d(attn_windows, window_size, B, Hp, Wp)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        return x

    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, mask_matrix):
        x_shortcut = x
        x = self.forward_part1(x, mask_matrix)
        x = x_shortcut + self.drop_path(x)

        x_shortcut = x
        x = self.forward_part2(x)
        x = x_shortcut + x

        return x


class PatchMerging(nn.Module):
    """使用小波的Patch Merging（改进：可选保存高频给解码器）"""

    def __init__(self, dim, norm_layer=nn.LayerNorm, save_high_freq=False):  # ⭐ 新增参数
        super().__init__()
        self.dim = dim
        self.dwt = DWT_2D()
        self.save_high_freq = save_high_freq  # ⭐ 新增

        self.reduction = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.norm = norm_layer(dim * 4)
        self.high_freq_weight = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        """
        输入:  [B, H, W, C]
        输出:  [B, H/2, W/2, 2C], hf_subbands or None   ← ⭐ 改为返回元组
        """
        x = F.gelu(x)
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)

        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        LL, LH, HL, HH = self.dwt(x)

        # ⭐ 新增：后期层保存高频
        if self.save_high_freq:
            hf_subbands = (LH.clone(), HL.clone(), HH.clone())
        else:
            hf_subbands = None

        # 以下和原代码2完全一样 ─────────────
        high_weight = torch.sigmoid(self.high_freq_weight)
        LH = LH * (1 + high_weight)
        HL = HL * (1 + high_weight)
        HH = HH * (1 + high_weight)

        x = torch.cat([LL, LH, HL, HH], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 1)

        return x, hf_subbands  # ⭐ 原来只返回 x，现在多返回一个













@lru_cache() 
def compute_mask_2d(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None))
    w_slices = (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition_2d(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask




# ============================================================
# 代码2的 MultiModalBasicLayer → 修改 forward 传递 hf_storage
# ============================================================
class MultiModalBasicLayer(nn.Module):

    def __init__(self, dim, depth, num_heads, window_size=(7, 7), mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, save_high_freq=False):  # ⭐ 新增 save_high_freq
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # blocks 不变
        self.blocks = nn.ModuleList([
            MultiModalSwinTransformerBlock2D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # ⭐ 改动：传入 save_high_freq 给 PatchMerging
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer,
                                          save_high_freq=save_high_freq)
        else:
            self.downsample = None

    # ⭐ forward 签名改了：加 hf_storage，返回也加 hf_storage
    def forward(self, m1, m3, extract_feature, hf_storage):
        B, C, H, W = m1.shape
        window_size, shift_size = get_window_size_2d((H, W), self.window_size, self.shift_size)

        m1 = rearrange(m1, 'b c h w -> b h w c')
        m3 = rearrange(m3, 'b c h w -> b h w c')

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        attn_mask = compute_mask_2d(Hp, Wp, window_size, shift_size, m1.device)

        for depth_idx, blk in enumerate(self.blocks):
            m1, m3 = blk(m1, m3, attn_mask, cross=(depth_idx == len(self.blocks) - 1))

        # 不变
        fused_features = torch.cat([
            rearrange(m1, 'b h w c -> b c h w'),
            rearrange(m3, 'b h w c -> b c h w')], dim=1)
        extract_feature.append(fused_features)

        m1 = m1.reshape(B, H, W, -1)
        m3 = m3.reshape(B, H, W, -1)

        # ⭐ 改动：处理下采样返回的高频
        if self.downsample is not None:
            m1, m1_hf = self.downsample(m1)   # 现在返回元组
            m3, m3_hf = self.downsample(m3)

            if m1_hf is not None and m3_hf is not None:
                # 两个模态的高频取平均（轻量融合）
                fused_hf = tuple(
                    (h1 + h2) / 2.0 for h1, h2 in zip(m1_hf, m3_hf)
                )
                hf_storage.append(fused_hf)
            else:
                hf_storage.append(None)  # 占位

        m1 = rearrange(m1, 'b h w c -> b c h w')
        m3 = rearrange(m3, 'b h w c -> b c h w')

        return extract_feature, m1, m3, hf_storage  # ⭐ 多返回 hf_storage





class PatchEmbed2D(nn.Module):

    def __init__(self, img_size=(128, 128), patch_size=(4, 4), in_chans=3, embed_dim=64, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.proj = Conv_Stem(in_chans, embed_dim)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, modality):
        _, _, H, W = x.shape

        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x_2, x = self.proj(x ,modality)

        if self.norm is not None:
            H, W = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, H, W)

        return x_2, x




# ============================================================
# 代码2的 Encoder2D → 控制哪层保存高频 + 返回 hf_storage
# ============================================================
class Encoder2D(nn.Module):

    def __init__(self, embed_dim=32, img_size=(128, 128), patch_size=(4, 4),
                 in_chans_m1=3, in_chans_m3=1,
                 depths=[2, 2, 2], num_heads=[2, 4, 8], window_size=(7, 7), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                 hf_save_start_layer=1):  # ⭐ 新增：从第几层开始保存高频
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # 不变
        self.patch_embed_m1 = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans_m1,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed_m3 = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans_m3,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        # 小波注意力（不变）
        self.wavelet_attns = nn.ModuleList()
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                out_dim = int(embed_dim * 2 ** (i + 1))
            else:
                out_dim = int(embed_dim * 2 ** i)
            self.wavelet_attns.append(WaveletHighFreqAttention(out_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ⭐ 改动：构建层时传入 save_high_freq
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            has_downsample = i_layer < self.num_layers - 1

            # ⭐ 核心逻辑：只有后期层才保存高频
            save_hf = has_downsample and (i_layer >= hf_save_start_layer)

            layer = MultiModalBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if has_downsample else None,
                save_high_freq=save_hf)  # ⭐ 传入
            self.layers.append(layer)

        # Bottleneck（不变）
        bottleneck_dim = embed_dim * 2 ** (self.num_layers - 1) * 2

        self.bottleneck = nn.ModuleList([
            BottleneckBlock(
                dim=bottleneck_dim, num_heads=num_heads[-1],
                window_size=window_size, shift_size=(0, 0),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, norm_layer=norm_layer),
            BottleneckBlock(
                dim=bottleneck_dim, num_heads=num_heads[-1],
                window_size=window_size,
                shift_size=tuple(i // 2 for i in window_size),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, norm_layer=norm_layer)])

        self.norm = norm_layer(bottleneck_dim)

    def forward(self, m1, m3):
        extract_feature = []
        hf_storage = []  # ⭐ 新增

        m1_stem, m1 = self.patch_embed_m1(m1, 'rgb')
        m3_stem, m3 = self.patch_embed_m3(m3, 'nir')
        m1 = self.pos_drop(m1)
        m3 = self.pos_drop(m3)

        for i, layer in enumerate(self.layers):
            # ⭐ 改动：传入和接收 hf_storage
            extract_feature, m1, m3, hf_storage = layer(m1, m3, extract_feature, hf_storage)

            m1 = self.wavelet_attns[i](m1)
            m3 = self.wavelet_attns[i](m3)

        # Bottleneck（不变）
        x = torch.cat([m1, m3], dim=1)
        B, C, H, W = x.shape
        shift_size = tuple(i // 2 for i in (4, 4))
        window_size, shift_size = get_window_size_2d((H, W), (4, 4), shift_size)

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        attn_mask = compute_mask_2d(Hp, Wp, window_size, shift_size, x.device)

        x = rearrange(x, 'b c h w -> b h w c')
        for blk in self.bottleneck:
            x = blk(x, attn_mask)
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')

        stem_features = torch.cat([m1_stem, m3_stem], dim=1)

        # ⭐ 改动：多返回 hf_storage
        return stem_features, extract_feature[0], extract_feature[1], extract_feature[2], x, hf_storage










class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,stride=stride, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.GroupNorm(num_groups=2, num_channels=out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)



# ============================================================
# 代码2的 MultiModalDecoder2D → 第一处上采样改为小波上采样
# ============================================================
class MultiModalDecoder2D(nn.Module):

    def __init__(self, num_classes=2, embed_dim=32):
        super().__init__()

        # z4+z3 融合（不变）
        self.cca_z4_z3 = CoordAtt(inp=2 * embed_dim * 4, oup=2 * embed_dim * 4)
        self.process_z4_z3 = Conv2DBlock(4 * embed_dim * 4, 2 * embed_dim * 4)


        self.wavelet_up_to_z2 = LightWaveletUpsample(
            in_dim=2 * embed_dim * 4,   # 256 (z4+z3处理后的通道)
            out_dim=2 * embed_dim * 2,   # 128 (目标通道)
            hf_dim=embed_dim * 2         # 64  (编码器Layer1下采样时的通道)
        )
        # =============================================

        self.cca_z2 = CoordAtt(inp=2 * embed_dim * 2, oup=2 * embed_dim * 2)
        self.process_z2 = Conv2DBlock(4 * embed_dim * 2, 2 * embed_dim * 2)

        # H/8 → H/4: 保持反卷积不变（这层没有保存高频）
        self.up_to_z1 = SingleDeconv2DBlock(2 * embed_dim * 2, 2 * embed_dim)
        self.cca_z1 = CoordAtt(inp=2 * embed_dim, oup=2 * embed_dim)
        self.process_z1 = Conv2DBlock(4 * embed_dim, 2 * embed_dim)

        # ===== 以下全部和代码2原版一模一样 =====
        self.up_to_z0 = SingleDeconv2DBlock(2 * embed_dim, 2 * embed_dim // 2)
        self.fuse_z0 = Conv2DBlock(2 * embed_dim, embed_dim)

        self.deconv1 = SingleDeconv2DBlock(embed_dim, embed_dim // 2)
        self.deconv1_process = Conv2DBlock(embed_dim // 2, embed_dim // 2)
        self.deconv2 = SingleDeconv2DBlock(embed_dim // 2, embed_dim // 4)
        self.deconv2_process = Conv2DBlock(embed_dim // 4, embed_dim // 4)

        self.zbase_deconv1 = SingleDeconv2DBlock(4, embed_dim // 4)
        self.zbase_process1 = Conv2DBlock(embed_dim // 4, embed_dim // 4)
        self.zbase_deconv2 = Conv2DBlock(embed_dim // 4, embed_dim // 4)

        self.final_fuse = Conv2DBlock(embed_dim // 2, embed_dim // 4)

        self.final_deconv1 = nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, stride=2, padding=1)
        self.final_process1 = Conv2DBlock(embed_dim // 4, embed_dim // 4)

        self.final_deconv2 = Conv2DBlock(embed_dim // 4, embed_dim // 8)
        self.final_deconv3 = Conv2DBlock(embed_dim // 8, embed_dim // 8)

        self.seg_head = SingleConv2DBlock(embed_dim // 8, num_classes, 1)

    def forward(self, z0, z1, z2, z3, z4, zbase, hf_storage):  # ⭐ 加了 hf_storage
        """
        hf_storage 结构（hf_save_start_layer=1 时）:
            hf_storage[0] = None                   ← Layer0 不保存
            hf_storage[1] = (LH, HL, HH)           ← Layer1 保存的高频
                             各 [B, 64, H/16, W/16]
        """

        skip_z3 = self.cca_z4_z3(z4, z3)
        x = torch.cat([z4, skip_z3], dim=1)
        x = self.process_z4_z3(x)

        # =============================================
        # ⭐ 改动：用小波上采样（原来是 x = self.up_to_z2(x)）
        x = self.wavelet_up_to_z2(x, hf_storage[1])
        # =============================================

        skip_z2  = self.cca_z2(x, z2)
        x = torch.cat([x, skip_z2], dim=1)
        x = self.process_z2(x)

        # 这里保持不变（没有保存的高频）
        x  = self.up_to_z1(x)
        skip_z1 = self.cca_z1(x, z1)
        x = torch.cat([x, skip_z1], dim=1)
        x = self.process_z1(x)

        # ===== 以下和代码2原版一模一样 =====
        x = self.up_to_z0(x)
        x = torch.cat([x, z0], dim=1)
        x = self.fuse_z0(x)

        x = self.deconv1(x)
        x = self.deconv1_process(x)
        x = self.deconv2(x)
        x = self.deconv2_process(x)

        zb = self.zbase_deconv1(zbase)
        zb = self.zbase_process1(zb)
        zb = self.zbase_deconv2(zb)

        x = torch.cat([x, zb], dim=1)
        x = self.final_fuse(x)

        x = self.final_deconv1(x)
        x = self.final_process1(x)
        x = self.final_deconv2(x)
        x = self.final_deconv3(x)

        out = self.seg_head(x)
        return out




# ============================================================
# 代码2的 MultiModalCKD2D → 传递 hf_storage
# ============================================================
class MultiModalCKD2D(nn.Module):
    def __init__(self,
                 in_channels_rgb=3,
                 in_channels_nir=1,
                 num_classes=3,
                 img_size=(128, 128),
                 patch_size=(4, 4),
                 embed_dim=32,
                 depths=[2, 2, 2],
                 num_heads=[2, 4, 8],
                 window_size=(7, 7),
                 mlp_ratio=4.):
        super().__init__()

        self.in_channels_rgb = in_channels_rgb
        self.in_channels_nir = in_channels_nir

        self.encoder = Encoder2D(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            in_chans_m1=in_channels_rgb,
            in_chans_m3=in_channels_nir,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            hf_save_start_layer=1  # ⭐ 只在后期层(Layer1)保存高频
        )

        self.decoder = MultiModalDecoder2D(
            num_classes=num_classes,
            embed_dim=embed_dim
        )

    def forward(self, inputs):
        if isinstance(inputs, dict):
            rgb = inputs['rgb']
            nir = inputs['nir']
        elif isinstance(inputs, (tuple, list)):
            rgb, nir = inputs
        else:
            raise ValueError("Input must be dict or tuple/list")

        # ⭐ 改动：接收 hf_storage
        z0, z1, z2, z3, z4, hf_storage = self.encoder(rgb, nir)

        zbase = torch.cat([rgb, nir], dim=1)

        # ⭐ 改动：传递 hf_storage 给解码器
        out = self.decoder(z0, z1, z2, z3, z4, zbase, hf_storage)
        return out




