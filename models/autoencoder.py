# ------------------------------------------------------------------------
# Autoencoder-based Low-Light Image Enhancement
# Architecture 7: Symmetric encoder-decoder WITHOUT skip connections.
#
# The autoencoder learns a compressed latent representation of the image.
# Without skip connections, all information must pass through the
# bottleneck, forcing the network to learn a compact, denoised
# representation. A global residual connection ensures the network
# only needs to learn the enhancement residual.
#
# Design: 4-level encoder with stride-2 conv downsampling, 2 residual
# blocks in the bottleneck, ConvTranspose2d upsampling in decoder.
# LeakyReLU activations throughout. ~3M parameters.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual Block with two 3x3 convolutions and LeakyReLU.
    
    Args:
        channels (int): Number of input/output channels
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(x + self.block(x))


class AutoencoderEnhancement(nn.Module):
    """Autoencoder for Low-Light Image Enhancement.
    
    Architecture:
        - Input projection: 3 -> base_ch
        - Encoder (4 levels):
            Conv(stride=2) + LeakyReLU + Conv + LeakyReLU at each level
            Channels: base -> 2*base -> 4*base -> 8*base
        - Bottleneck: 2 ResBlocks at 8*base channels
        - Decoder (4 levels):
            ConvTranspose2d(stride=2) + LeakyReLU + Conv + LeakyReLU
            Channels: 8*base -> 4*base -> 2*base -> base
        - Output projection: base_ch -> 3
        - Global residual: output + input
    
    NO skip connections — all information passes through bottleneck.
    
    Args:
        img_channel (int): Number of input/output image channels (default: 3)
        base_channels (int): Base channel width (default: 32)
        num_res_blocks (int): Number of residual blocks in bottleneck (default: 2)
    """
    def __init__(self, img_channel=3, base_channels=32, num_res_blocks=2):
        super().__init__()
        bc = base_channels

        # Input projection
        self.intro = nn.Sequential(
            nn.Conv2d(img_channel, bc, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder: 4 levels of downsampling
        self.enc1 = nn.Sequential(
            nn.Conv2d(bc, bc * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc * 2, bc * 2, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(bc * 2, bc * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc * 4, bc * 4, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(bc * 4, bc * 8, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc * 8, bc * 8, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(bc * 8, bc * 8, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc * 8, bc * 8, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck: Residual blocks
        self.bottleneck = nn.Sequential(
            *[ResBlock(bc * 8) for _ in range(num_res_blocks)]
        )

        # Decoder: 4 levels of upsampling
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(bc * 8, bc * 8, kernel_size=2, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc * 8, bc * 8, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(bc * 8, bc * 4, kernel_size=2, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc * 4, bc * 4, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(bc * 4, bc * 2, kernel_size=2, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc * 2, bc * 2, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(bc * 2, bc, kernel_size=2, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc, bc, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output projection
        self.ending = nn.Conv2d(bc, img_channel, kernel_size=3, padding=1, bias=True)

        self.padder_size = 16  # 2^4 for 4 levels

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # Input projection
        x = self.intro(inp)

        # Encoder (NO skip connections saved)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

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


def autoencoder_enhancement(img_channel=3):
    """Autoencoder Enhancement Model for LOL dataset. ~3.2M params."""
    return AutoencoderEnhancement(
        img_channel=img_channel,
        base_channels=32,
        num_res_blocks=2,
    )


if __name__ == '__main__':
    model = autoencoder_enhancement()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Autoencoder Enhancement")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {params:,}")

