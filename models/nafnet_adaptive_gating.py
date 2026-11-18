# ------------------------------------------------------------------------
# NAFNet + Adaptive Gating Mechanism for Low-Light Image Enhancement
# Architecture 6: Learns element-wise gating between encoder skip
# features and decoder features at each skip connection.
#
# Instead of simple additive skip connections (x = up(x) + enc_skip),
# an adaptive gate learns a per-element blending weight:
#   gate = sigmoid(W_g * cat(decoder_feat, enc_skip))
#   fused = gate * decoder_feat + (1 - gate) * enc_skip
#
# This allows the network to adaptively decide how much to rely on
# encoder features vs. decoder features at each spatial location and
# channel, which is particularly useful in low-light enhancement where
# encoder features may contain noise from dark regions.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nafnet import LayerNorm2d, SimpleGate, NAFBlock


class AdaptiveGate(nn.Module):
    """Adaptive Gating Module for skip connections.
    
    Learns element-wise gating weights to blend two feature maps.
    Uses a 1x1 convolution on concatenated inputs followed by sigmoid.
    
    Formula:
        gate = sigmoid(conv([x_dec, x_enc]))
        output = gate * x_dec + (1 - gate) * x_enc
    
    Args:
        channels (int): Number of channels in each input feature map.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_dec, x_enc):
        """
        Args:
            x_dec: Decoder features after upsampling [B, C, H, W]
            x_enc: Encoder skip features [B, C, H, W]
        Returns:
            Adaptively gated/fused features [B, C, H, W]
        """
        combined = torch.cat([x_dec, x_enc], dim=1)  # [B, 2C, H, W]
        gate = self.gate_conv(combined)                # [B, C, H, W]
        return gate * x_dec + (1.0 - gate) * x_enc


class NAFNetAdaptiveGating(nn.Module):
    """NAFNet with Adaptive Gating at Skip Connections.
    
    Uses standard NAFBlock but replaces additive skip connections with
    learned adaptive gates. The gates decide per-element how much of the
    encoder feature vs decoder feature to use.
    
    Architecture:
        - Input: 3x3 Conv projection
        - Encoder: N stages of NAFBlock + stride-2 downsampling
        - Bottleneck: Stacked NAFBlock at lowest resolution
        - Decoder: PixelShuffle upsampling + AdaptiveGate(dec, enc_skip) + NAFBlock
        - Output: 3x3 Conv projection + global residual
    
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

        # Encoder stages — uses standard NAFBlock
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan *= 2

        # Bottleneck
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        # Decoder stages with adaptive gating
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList()  # NEW: Adaptive gates per level

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan //= 2
            self.gates.append(AdaptiveGate(chan))  # NEW
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
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

        # Decoder with adaptive gating at skip connections
        for decoder, up, gate, enc_skip in zip(
            self.decoders, self.ups, self.gates, encs[::-1]
        ):
            x = up(x)
            x = gate(x, enc_skip)  # Adaptive gating (replaces x + enc_skip)
            x = decoder(x)

        # Output projection + global residual
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def nafnet_adaptive_gating_width32(img_channel=3):
    """NAFNet-AdaptiveGating-width32: NAFNet with adaptive gating, width 32."""
    return NAFNetAdaptiveGating(
        img_channel=img_channel,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )


if __name__ == '__main__':
    model = nafnet_adaptive_gating_width32()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NAFNet + Adaptive Gating")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {params:,}")

