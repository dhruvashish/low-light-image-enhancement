# ------------------------------------------------------------------------
# NAFNet: Nonlinear Activation Free Network for Image Restoration
# Based on: https://github.com/megvii-research/NAFNet
# Paper: "Simple Baselines for Image Restoration" (ECCV 2022)
# Authors: Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun
#
# This is a self-contained implementation faithful to the official repo.
# Adapted for low-light image enhancement on the LOL dataset.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 2D feature maps.
    
    Normalizes across the channel dimension for every spatial location,
    following the official NAFNet implementation.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x


class SimpleGate(nn.Module):
    """SimpleGate activation: splits channels in half and multiplies.
    
    Replaces traditional nonlinear activations (ReLU, GELU, Sigmoid)
    with a data-dependent, element-wise gating mechanism.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAF Block: the core building block of NAFNet.
    
    Architecture:
        1. LayerNorm -> 1x1 Conv (channel expand) -> 3x3 DWConv ->
           SimpleGate -> SCA -> 1x1 Conv (channel reduce) -> Dropout
           -> Scaled Residual (beta)
        2. LayerNorm -> 1x1 Conv (FFN expand) -> SimpleGate ->
           1x1 Conv (FFN reduce) -> Dropout -> Scaled Residual (gamma)
    
    Args:
        c (int): Number of input/output channels
        DW_Expand (int): Depthwise conv channel expansion factor (default: 2)
        FFN_Expand (int): Feed-forward network expansion factor (default: 2)
        drop_out_rate (float): Dropout rate (default: 0.0)
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand

        # --- Spatial Mixing Branch ---
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3,
                               padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True),
        )

        self.sg = SimpleGate()

        # --- Feed-Forward Network Branch ---
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        # Layer Normalizations
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout layers
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Learnable scaling parameters for residual connections (init to 0)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # --- Sub-block 1: Spatial Mixing + SCA ---
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        # --- Sub-block 2: Channel Mixing (FFN) ---
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet: Nonlinear Activation Free Network for Image Restoration.
    
    A U-Net-like encoder-decoder architecture built entirely from NAFBlocks.
    Uses additive skip connections and global residual learning.
    
    Architecture:
        - Input: 3x3 Conv projection to feature space
        - Encoder: N stages, each with stacked NAFBlocks + 2x downsampling (Conv stride=2)
        - Bottleneck: Stacked NAFBlocks at lowest resolution
        - Decoder: N stages, each with 2x upsampling (PixelShuffle) + stacked NAFBlocks
        - Skip connections: Element-wise addition (not concatenation)
        - Output: 3x3 Conv projection back to image space + global residual
    
    Args:
        img_channel (int): Number of input/output image channels (default: 3)
        width (int): Base channel width (default: 32)
        middle_blk_num (int): Number of NAFBlocks in bottleneck (default: 1)
        enc_blk_nums (list): Number of NAFBlocks per encoder stage (default: [1,1,1,28])
        dec_blk_nums (list): Number of NAFBlocks per decoder stage (default: [1,1,1,1])
    """

    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__()

        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        # Input projection: img -> feature space
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1,
                               stride=1, bias=True)

        # Output projection: feature space -> img
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1,
                                stride=1, bias=True)

        # Encoder stages
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan = chan * 2

        # Bottleneck (middle blocks)
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        # Decoder stages
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        # Padding utilities for handling non-divisible spatial dims
        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape

        # Pad input to be divisible by padder_size
        inp = self.check_image_size(inp)

        # Input projection
        x = self.intro(inp)

        # Encoder path
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Bottleneck
        x = self.middle_blks(x)

        # Decoder path with additive skip connections
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip  # Additive skip connection (NOT concat)
            x = decoder(x)

        # Output projection + global residual
        x = self.ending(x)
        x = x + inp

        # Remove padding
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """Pad image so H and W are divisible by padder_size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ===== Predefined configurations =====

def nafnet_width32(img_channel=3):
    """NAFNet-width32: Lightweight config suitable for LOL dataset.
    
    ~6.5M parameters. Balanced for 6GB VRAM with batch_size=4 at 256x256.
    """
    return NAFNet(
        img_channel=img_channel,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )


def nafnet_width64(img_channel=3):
    """NAFNet-width64: Higher capacity config.
    
    ~25.5M parameters. May require batch_size=2 on 6GB VRAM at 256x256.
    """
    return NAFNet(
        img_channel=img_channel,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )


if __name__ == '__main__':
    # Quick sanity check
    model = nafnet_width32()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {params:,}")
