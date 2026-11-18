# ------------------------------------------------------------------------
# Classic U-Net for Low-Light Image Enhancement
# Baseline Architecture 1: Classic U-Net (Ronneberger et al.)
#
# Standard 4-level U-Net with DoubleConv blocks, MaxPool downsampling,
# bilinear upsampling, concatenation skip connections, and global residual.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with bilinear interpolation then DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Classic U-Net for Low-Light Image Enhancement.
    
    4-level encoder-decoder with DoubleConv blocks, concatenation skips,
    and global residual learning (output = out_conv(feat) + inp).
    
    Args:
        n_channels (int): Input image channels (3)
        n_classes (int): Output image channels (3)
        base_channels (int): Base feature width (default: 32)
        bilinear (bool): Use bilinear upsampling (default: True)
    """
    def __init__(self, n_channels=3, n_classes=3, base_channels=32, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        bc = base_channels

        self.inc = DoubleConv(n_channels, bc)
        self.down1 = Down(bc, bc * 2)
        self.down2 = Down(bc * 2, bc * 4)
        self.down3 = Down(bc * 4, bc * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(bc * 8, bc * 16 // factor)

        self.up1 = Up(bc * 16, bc * 8 // factor, bilinear)
        self.up2 = Up(bc * 8, bc * 4 // factor, bilinear)
        self.up3 = Up(bc * 4, bc * 2 // factor, bilinear)
        self.up4 = Up(bc * 2, bc, bilinear)
        self.outc = OutConv(bc, n_classes)

        self.padder_size = 16

    def forward(self, inp):
        B, C, H, W = inp.shape
        x_pad = self.check_image_size(inp)

        x1 = self.inc(x_pad)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # Global residual connection
        out = logits + x_pad
        return out[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def unet_baseline(n_channels=3, base_channels=32):
    """Factory function for classic U-Net baseline."""
    return UNet(n_channels=n_channels, n_classes=n_channels, base_channels=base_channels)


if __name__ == '__main__':
    model = unet_baseline()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Classic UNet Baseline")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {params:,}")

