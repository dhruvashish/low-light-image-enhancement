# ------------------------------------------------------------------------
# NAFNet + Spatial Attention for Low-Light Image Enhancement
#
# Extends NAFNet by adding a CBAM-style spatial attention module after
# the Simplified Channel Attention (SCA) in each NAF block. The spatial
# attention generates a 2D attention map by applying a 7×7 convolution
# on max-pooled and average-pooled features along the channel axis,
# allowing the network to focus on spatially important dark/bright regions.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nafnet import LayerNorm2d, SimpleGate


class SpatialAttention(nn.Module):
    """Spatial Attention module (CBAM-style).

    Generates a spatial attention map by:
    1. Computing channel-wise max-pool and avg-pool of the input feature map.
    2. Concatenating them along the channel axis (2 channels).
    3. Applying a 7×7 convolution to produce a 1-channel attention map.
    4. Passing through sigmoid to get the spatial mask in [0, 1].

    Args:
        kernel_size (int): Convolution kernel size (default: 7).
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)        # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)      # [B, 2, H, W]
        attn_map = self.sigmoid(self.conv(combined))         # [B, 1, H, W]
        return x * attn_map


class NAFBlockSpatialAttn(nn.Module):
    """NAF Block with additional Spatial Attention after SCA.

    Architecture:
        Sub-block 1: LayerNorm → 1×1 Conv (expand) → 3×3 DWConv →
                     SimpleGate → SCA → **SpatialAttention** → 1×1 Conv (reduce)
                     → Dropout → Scaled Residual (beta)
        Sub-block 2: LayerNorm → 1×1 Conv (FFN expand) → SimpleGate →
                     1×1 Conv (FFN reduce) → Dropout → Scaled Residual (gamma)

    Args:
        c (int): Number of input/output channels.
        DW_Expand (int): Depthwise conv expansion factor (default: 2).
        FFN_Expand (int): FFN expansion factor (default: 2).
        drop_out_rate (float): Dropout rate (default: 0.0).
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand

        # --- Spatial Mixing Branch ---
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3,
                               padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        # Simplified Channel Attention (SCA) — kept from original
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True),
        )

        # Spatial Attention — added after SCA
        self.spatial_attn = SpatialAttention(kernel_size=7)

        self.sg = SimpleGate()

        # --- Feed-Forward Network Branch ---
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        # Layer Normalizations
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Learnable scaling parameters
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # --- Sub-block 1: Spatial Mixing + SCA + SpatialAttention ---
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)         # SCA
        x = self.spatial_attn(x)     # Spatial Attention (NEW)
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


class NAFNetSpatialAttn(nn.Module):
    """NAFNet with Spatial Attention for Low-Light Image Enhancement.

    A U-Net-like encoder-decoder architecture using NAFBlockSpatialAttn
    blocks. Adds CBAM-style spatial attention after SCA in every block,
    enabling spatially-aware feature refinement for dark regions.

    Architecture:
        - Input: 3×3 Conv projection
        - Encoder: N stages of NAFBlockSpatialAttn + stride-2 downsampling
        - Bottleneck: Stacked NAFBlockSpatialAttn at lowest resolution
        - Decoder: PixelShuffle upsampling + additive skips + NAFBlockSpatialAttn
        - Output: 3×3 Conv projection + global residual

    Args:
        img_channel (int): Input/output image channels (default: 3).
        width (int): Base channel width (default: 32).
        middle_blk_num (int): Number of bottleneck blocks (default: 1).
        enc_blk_nums (list): Blocks per encoder stage (default: [2,2,4,8]).
        dec_blk_nums (list): Blocks per decoder stage (default: [2,2,2,2]).
    """

    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__()

        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        # Input projection
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1,
                               stride=1, bias=True)
        # Output projection
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1,
                                stride=1, bias=True)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlockSpatialAttn(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan *= 2

        # Bottleneck
        self.middle_blks = nn.Sequential(
            *[NAFBlockSpatialAttn(chan) for _ in range(middle_blk_num)]
        )

        # Decoder
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
                nn.Sequential(*[NAFBlockSpatialAttn(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        # Encoder
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Bottleneck
        x = self.middle_blks(x)

        # Decoder with additive skip connections
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        # Output projection + global residual
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """Pad image so H and W are divisible by padder_size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ===== Factory =====

def nafnet_spatial_attn_width32(img_channel=3):
    """NAFNet-SpatialAttn-width32: NAFNet with spatial attention, base width 32.

    ~6.6M parameters. Adds CBAM-style spatial attention after SCA in
    every NAF block for spatially-aware low-light enhancement.
    """
    return NAFNetSpatialAttn(
        img_channel=img_channel,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )


if __name__ == '__main__':
    model = nafnet_spatial_attn_width32()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {params:,}")
