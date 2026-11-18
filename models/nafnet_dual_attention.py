# ------------------------------------------------------------------------
# NAFNet + Dual (Spatial + Channel) Attention for Low-Light Enhancement
# Architecture 5: Combines SE Channel Attention and CBAM Spatial Attention.
#
# Applies channel attention first (SE: recalibrate channels), then spatial
# attention (CBAM: focus on important spatial regions) sequentially in
# each NAFBlock. This dual-attention mechanism captures both cross-channel
# dependencies and spatial saliency for comprehensive feature refinement.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nafnet import LayerNorm2d, SimpleGate
from models.nafnet_channel_attention import ChannelAttention
from models.nafnet_spatial_attention import SpatialAttention


class NAFBlockDualAttn(nn.Module):
    """NAF Block with Dual (Channel + Spatial) Attention.
    
    Architecture:
        Sub-block 1: LayerNorm -> 1x1 Conv -> 3x3 DWConv -> SimpleGate ->
                     SE Channel Attention -> Spatial Attention ->
                     1x1 Conv -> Dropout -> Scaled Residual (beta)
        Sub-block 2: LayerNorm -> 1x1 Conv (FFN) -> SimpleGate ->
                     1x1 Conv -> Dropout -> Scaled Residual (gamma)
    
    Args:
        c (int): Number of input/output channels
        DW_Expand (int): Depthwise conv expansion factor (default: 2)
        FFN_Expand (int): FFN expansion factor (default: 2)
        drop_out_rate (float): Dropout rate (default: 0.0)
        reduction (int): SE reduction ratio (default: 16)
        spatial_kernel (int): Spatial attention kernel size (default: 7)
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,
                 reduction=16, spatial_kernel=7):
        super().__init__()
        dw_channel = c * DW_Expand

        # --- Spatial Mixing Branch ---
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3,
                               padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        # Dual Attention: Channel (SE) followed by Spatial (CBAM)
        self.channel_attn = ChannelAttention(dw_channel // 2, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

        self.sg = SimpleGate()

        # --- FFN Branch ---
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        # Layer Normalizations
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Learnable residual scaling
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # Sub-block 1: Spatial Mixing + Channel Attn + Spatial Attn
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.channel_attn(x)     # SE Channel Attention
        x = self.spatial_attn(x)     # CBAM Spatial Attention
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        # Sub-block 2: FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNetDualAttn(nn.Module):
    """NAFNet with Dual (Channel + Spatial) Attention.
    
    U-Net encoder-decoder using NAFBlockDualAttn blocks. Each block applies
    SE channel attention followed by CBAM spatial attention, combining
    both channel recalibration and spatial feature selection.
    
    Args:
        img_channel (int): Input/output image channels (default: 3)
        width (int): Base channel width (default: 32)
        middle_blk_num (int): Number of bottleneck blocks (default: 1)
        enc_blk_nums (list): Blocks per encoder stage (default: [2,2,4,8])
        dec_blk_nums (list): Blocks per decoder stage (default: [2,2,2,2])
    """
    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__()

        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1,
                               stride=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1,
                                stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlockDualAttn(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlockDualAttn(chan) for _ in range(middle_blk_num)]
        )

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlockDualAttn(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def nafnet_dual_attn_width32(img_channel=3):
    """NAFNet-DualAttn-width32: Dual attention config for LOL dataset."""
    return NAFNetDualAttn(
        img_channel=img_channel,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )


if __name__ == '__main__':
    model = nafnet_dual_attn_width32()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NAFNet + Dual Attention (Channel + Spatial)")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {params:,}")

