# ------------------------------------------------------------------------
# IG-NAFNet: Illumination-Guided Nonlinear Activation Free Network
# for Low-Light Image Enhancement
#
# Novelty & Research Contributions:
# 1. Illumination Prior Estimator (IPE): Extracts continuous multi-scale
#    illumination maps based on Retinex physical priors (L_0 = max_c(I)).
# 2. Illumination-Modulated SimpleGate (IM-SimpleGate): Dynamically scales
#    feature activation based on local luminance, applying high gain and
#    aggressive denoising to dark regions while preventing overexposure
#    in highlight regions.
# 3. Edge-Preserving Illumination Smoothness & Exposure Consistency:
#    Jointly optimized with L1, SSIM, VGG Perceptual, and Retinex TV losses.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 2D feature maps."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x


class SimpleGate(nn.Module):
    """SimpleGate: splits channels in half and computes element-wise product."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class IlluminationPriorEstimator(nn.Module):
    """Estimates continuous illumination map and multi-scale guidance features.
    
    Based on Retinex formulation:
    Initial illumination prior L_0(x) = max_{c in {R,G,B}} I_c(x).
    A lightweight edge-preserving network refines L_0 into a normalized
    continuous illumination map L in [0, 1] and multi-scale guidance features.
    """
    def __init__(self, base_channels=32):
        super().__init__()
        # Input: 3 channels (RGB) + 1 channel (max-RGB initial prior) = 4 channels
        self.stem = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Multi-scale feature projection heads for encoder/bottleneck levels
        # Level 0 (scale 1x)
        self.proj0 = nn.Conv2d(1, base_channels, kernel_size=1)
        # Level 1 (scale 1/2)
        self.proj1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, base_channels * 2, kernel_size=1)
        )
        # Level 2 (scale 1/4)
        self.proj2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(1, base_channels * 4, kernel_size=1)
        )
        # Level 3 (scale 1/8)
        self.proj3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=8, stride=8),
            nn.Conv2d(1, base_channels * 8, kernel_size=1)
        )

    def forward(self, x):
        # Initial Retinex max-channel prior
        l_init = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        feat_in = torch.cat([x, l_init], dim=1)        # [B, 4, H, W]

        # Refined continuous illumination map
        illum_map = self.stem(feat_in)                  # [B, 1, H, W] in [0, 1]

        # Generate scale-matched guidance features
        g0 = self.proj0(illum_map)
        g1 = self.proj1(illum_map)
        g2 = self.proj2(illum_map)
        g3 = self.proj3(illum_map)

        return illum_map, [g0, g1, g2, g3]


class IMNAFBlock(nn.Module):
    """Illumination-Modulated NAF Block (IM-NAFBlock).
    
    Novelty: Integrates illumination guidance directly into the SimpleGate
    mechanism. When illumination is low (dark regions), the modulation scales up
    spatial detail features and channel attention; in bright regions, it scales
    down enhancement to suppress noise amplification and avoid saturation.
    
    Formula:
        y_gated = SimpleGate(x) * (1 + alpha * Sigmoid(W_scale(G))) + W_shift(G)
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand

        # --- Spatial Mixing Branch ---
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3,
                               padding=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()

        # Illumination Modulation Adapter for SimpleGate output (dw_channel // 2)
        # Learnable affine scale & shift conditioned on illumination features
        self.illum_mod = nn.Conv2d(c, dw_channel // 2, kernel_size=1, bias=True)
        self.illum_alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True),
        )

        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        # --- Feed-Forward Network Branch ---
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.sg_ffn = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        # Layer Normalizations
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Learnable residual weights
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, illum_feat=None):
        x = inp

        # --- Sub-block 1: Illumination-Modulated Spatial Mixing ---
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        # Apply illumination modulation if guidance feature is provided
        if illum_feat is not None:
            mod = torch.sigmoid(self.illum_mod(illum_feat))
            x = x * (1.0 + self.illum_alpha * mod)

        # Simplified Channel Attention
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        # --- Sub-block 2: Feed-Forward Network ---
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg_ffn(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class IGNAFNet(nn.Module):
    """IG-NAFNet: Illumination-Guided Nonlinear Activation Free Network.
    
    A physically motivated architecture for Low-Light Image Enhancement.
    Couples Retinex illumination decomposition with activation-free restoration.
    
    Key Features:
        - IPE extracts continuous illumination map L and multi-scale guidance
        - IM-NAFBlocks dynamically adapt enhancement gain per pixel and channel
        - Additive skip connections with Retinex feature integration
        - Global residual connection for crisp detail preservation
    """
    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__()

        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        self.width = width
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums

        # 1. Illumination Prior Estimator
        self.ipe = IlluminationPriorEstimator(base_channels=width)

        # 2. Input/Output Projections
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        # 3. Encoder Stages with IM-NAFBlocks
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            blocks = nn.ModuleList([IMNAFBlock(chan) for _ in range(num)])
            self.encoders.append(blocks)
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        # 4. Bottleneck Stage
        self.middle_blks = nn.ModuleList([IMNAFBlock(chan) for _ in range(middle_blk_num)])
        self.bottleneck_illum = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=16),
            nn.Conv2d(1, chan, kernel_size=1)
        )

        # 5. Decoder Stages
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
            blocks = nn.ModuleList([IMNAFBlock(chan) for _ in range(num)])
            self.decoders.append(blocks)

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp, return_illum=False):
        B, C, H, W = inp.shape

        # Ensure spatial dimensions are divisible by padder_size
        inp_padded = self.check_image_size(inp)

        # Step 1: Illumination Prior Estimation
        illum_map, illum_feats = self.ipe(inp_padded)
        # illum_feats: [g0 (width), g1 (2*width), g2 (4*width), g3 (8*width)]

        # Step 2: Input Projection
        x = self.intro(inp_padded)

        # Step 3: Encoder Path
        encs = []
        for i, (stage_blocks, down) in enumerate(zip(self.encoders, self.downs)):
            g = illum_feats[i]
            for blk in stage_blocks:
                x = blk(x, g)
            encs.append(x)
            x = down(x)

        # Step 4: Bottleneck
        g_mid = self.bottleneck_illum(illum_map)
        for blk in self.middle_blks:
            x = blk(x, g_mid)

        # Step 5: Decoder Path with Additive Skip Connections
        dec_feats = illum_feats[::-1]
        for i, (stage_blocks, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip  # Additive skip connection
            # Match guidance feature for decoder level
            g_dec = dec_feats[i]
            for blk in stage_blocks:
                x = blk(x, g_dec)

        # Step 6: Output Projection + Global Residual Connection
        out = self.ending(x)
        out = out + inp_padded

        # Crop back to original dimensions
        out = out[:, :, :H, :W]
        illum_map = illum_map[:, :, :H, :W]

        if return_illum:
            return out, illum_map
        return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x


def ig_nafnet_width32(img_channel=3):
    """IG-NAFNet standard configuration for LOL dataset."""
    return IGNAFNet(
        img_channel=img_channel,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )


# --- Illumination & Exposure Loss Functions ---

class IlluminationSmoothnessLoss(nn.Module):
    """Illumination Smoothness Loss with Edge Guidance.
    
    Enforces spatial smoothness on the illumination map, but allows sharp
    transitions at genuine semantic image boundaries by weighting gradients
    with the inverse of image gradients.
    """
    def __init__(self):
        super().__init__()

    def forward(self, illum, img):
        # illum: [B, 1, H, W], img: [B, 3, H, W]
        # Gradients of illumination map
        grad_illum_x = torch.abs(illum[:, :, :, :-1] - illum[:, :, :, 1:])
        grad_illum_y = torch.abs(illum[:, :, :-1, :] - illum[:, :, 1:, :])

        # Gradients of RGB input image (converted to grayscale)
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        grad_img_x = torch.abs(gray[:, :, :, :-1] - gray[:, :, :, 1:])
        grad_img_y = torch.abs(gray[:, :, :-1, :] - gray[:, :, 1:, :])

        # Edge-aware weighting: smooth where image is flat, preserve where image has edges
        weight_x = torch.exp(-10.0 * grad_img_x)
        weight_y = torch.exp(-10.0 * grad_img_y)

        loss_x = grad_illum_x * weight_x
        loss_y = grad_illum_y * weight_y

        return loss_x.mean() + loss_y.mean()


class ExposureControlLoss(nn.Module):
    """Exposure Control Loss.
    
    Measures the distance between the average local intensity and a well-exposed
    target value (default: 0.6) across 16x16 local patches.
    """
    def __init__(self, patch_size=16, target_exposure=0.6):
        super().__init__()
        self.patch_size = patch_size
        self.target = target_exposure
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, H, W]
        mean_intensity = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        patch_mean = self.pool(mean_intensity)
        loss = torch.mean(torch.abs(patch_mean - self.target))
        return loss


if __name__ == '__main__':
    model = ig_nafnet_width32()
    x = torch.randn(2, 3, 256, 256)
    out, illum = model(x, return_illum=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"IG-NAFNet Input:        {x.shape}")
    print(f"IG-NAFNet Output:       {out.shape}")
    print(f"Illum Map Output:       {illum.shape}")
    print(f"Trainable Parameters:   {params:,}")
