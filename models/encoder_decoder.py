# ------------------------------------------------------------------------
# Encoder-Decoder with Skip Connections for Low-Light Image Enhancement
# Architecture 8: U-Net style encoder-decoder with concatenation skips.
#
# Unlike the Autoencoder (no skips) and NAFNet (additive skips), this
# architecture uses CONCATENATION-based skip connections (classic U-Net
# style) with PixelShuffle upsampling. This preserves maximum spatial
# detail from the encoder while allowing the decoder to refine features.
#
# Design: 4-level encoder with stride-2 downsampling, residual bottleneck,
# PixelShuffle upsampling + concat skips + conv refinement in decoder.
# ~4M parameters.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> LN -> LeakyReLU -> Conv -> LN -> LeakyReLU.
    
    Args:
        in_ch (int): Input channels
        out_ch (int): Output channels
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class EncoderDecoderEnhancement(nn.Module):
    """Encoder-Decoder with Concatenation Skip Connections.
    
    Architecture:
        - Input projection: 3 -> base_ch (ConvBlock)
        - Encoder (4 levels):
            Each level: stride-2 Conv downsample -> ConvBlock
            Channels: base -> 2*base -> 4*base -> 8*base
        - Bottleneck: 2 ResBlocks at 8*base channels
        - Decoder (4 levels):
            Each level: PixelShuffle(2x) upsample -> Concat(skip) -> ConvBlock
            After concat: (upsampled_ch + skip_ch) channels input to ConvBlock
        - Output projection: base_ch -> 3
        - Global residual: output + input
    
    Args:
        img_channel (int): Number of input/output image channels (default: 3)
        base_channels (int): Base channel width (default: 32)
        num_res_blocks (int): Number of residual blocks in bottleneck (default: 2)
    """
    def __init__(self, img_channel=3, base_channels=32, num_res_blocks=2):
        super().__init__()
        bc = base_channels

        # Input projection
        self.intro = ConvBlock(img_channel, bc)

        # Encoder
        self.down1 = nn.Conv2d(bc, bc, kernel_size=2, stride=2, bias=True)
        self.enc1 = ConvBlock(bc, bc * 2)

        self.down2 = nn.Conv2d(bc * 2, bc * 2, kernel_size=2, stride=2, bias=True)
        self.enc2 = ConvBlock(bc * 2, bc * 4)

        self.down3 = nn.Conv2d(bc * 4, bc * 4, kernel_size=2, stride=2, bias=True)
        self.enc3 = ConvBlock(bc * 4, bc * 8)

        self.down4 = nn.Conv2d(bc * 8, bc * 8, kernel_size=2, stride=2, bias=True)
        self.enc4 = ConvBlock(bc * 8, bc * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResBlock(bc * 8) for _ in range(num_res_blocks)]
        )

        # Decoder with PixelShuffle + Concatenation skips
        # Level 4: upsample bc*8 -> bc*8//4 = bc*2 via PixelShuffle
        # But we want bc*8 output from upsample, so expand first
        self.up4_conv = nn.Conv2d(bc * 8, bc * 8 * 4, kernel_size=1, bias=False)
        self.up4_ps = nn.PixelShuffle(2)  # bc*8*4 -> bc*8 at 2x resolution
        self.dec4 = ConvBlock(bc * 8 + bc * 8, bc * 8)  # concat with enc3 skip

        self.up3_conv = nn.Conv2d(bc * 8, bc * 4 * 4, kernel_size=1, bias=False)
        self.up3_ps = nn.PixelShuffle(2)  # bc*4*4 -> bc*4
        self.dec3 = ConvBlock(bc * 4 + bc * 4, bc * 4)  # concat with enc2 skip

        self.up2_conv = nn.Conv2d(bc * 4, bc * 2 * 4, kernel_size=1, bias=False)
        self.up2_ps = nn.PixelShuffle(2)  # bc*2*4 -> bc*2
        self.dec2 = ConvBlock(bc * 2 + bc * 2, bc * 2)  # concat with enc1 skip

        self.up1_conv = nn.Conv2d(bc * 2, bc * 4, kernel_size=1, bias=False)
        self.up1_ps = nn.PixelShuffle(2)  # bc*4 -> bc
        self.dec1 = ConvBlock(bc + bc, bc)  # concat with intro skip

        # Output projection
        self.ending = nn.Conv2d(bc, img_channel, kernel_size=3, padding=1, bias=True)

        self.padder_size = 16

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # Input projection
        s0 = self.intro(inp)          # [B, bc, H, W]

        # Encoder
        x = self.down1(s0)
        s1 = self.enc1(x)            # [B, 2*bc, H/2, W/2]

        x = self.down2(s1)
        s2 = self.enc2(x)            # [B, 4*bc, H/4, W/4]

        x = self.down3(s2)
        s3 = self.enc3(x)            # [B, 8*bc, H/8, W/8]

        x = self.down4(s3)
        s4 = self.enc4(x)            # [B, 8*bc, H/16, W/16]

        # Bottleneck
        x = self.bottleneck(s4)       # [B, 8*bc, H/16, W/16]

        # Decoder with concat skips
        x = self.up4_ps(self.up4_conv(x))  # [B, 8*bc, H/8, W/8]
        x = self.dec4(torch.cat([x, s3], dim=1))  # [B, 8*bc, H/8, W/8]

        x = self.up3_ps(self.up3_conv(x))  # [B, 4*bc, H/4, W/4]
        x = self.dec3(torch.cat([x, s2], dim=1))  # [B, 4*bc, H/4, W/4]

        x = self.up2_ps(self.up2_conv(x))  # [B, 2*bc, H/2, W/2]
        x = self.dec2(torch.cat([x, s1], dim=1))  # [B, 2*bc, H/2, W/2]

        x = self.up1_ps(self.up1_conv(x))  # [B, bc, H, W]
        x = self.dec1(torch.cat([x, s0], dim=1))  # [B, bc, H, W]

        # Output + global residual
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def encoder_decoder_enhancement(img_channel=3):
    """Encoder-Decoder Enhancement Model for LOL dataset. ~4.5M params."""
    return EncoderDecoderEnhancement(
        img_channel=img_channel,
        base_channels=32,
        num_res_blocks=2,
    )


if __name__ == '__main__':
    model = encoder_decoder_enhancement()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Encoder-Decoder Enhancement")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {params:,}")

