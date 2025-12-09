"""
Light + DCN U-Net: Light U-Net with DCNv2 skip aligners only (no FDConv).

For ablation study comparing:
- Light U-Net (baseline reduced)
- Light + DCN (this model)
- Light + FD
- Light + Both (DCNFDUnet)
"""

import torch
from torch import nn
from torch.nn import functional as F

# Import DCNv2SkipRefiner from dcn_fd_unet to avoid duplication
from .dcn_fd_unet import DCNv2SkipRefiner


class ConvBlock(nn.Module):
    """Standard conv block."""
    
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_prob)
        )
    
    def forward(self, x):
        return self.layers(x)


class LightDCNUnet(nn.Module):
    """
    Light U-Net + DCNv2 skip aligners (no FDConv).
    
    Architecture:
    - Encoder: Standard convolutions
    - Skip Aligners: DCNv2 (adaptive alignment)
    - Bottleneck: Standard convolution (NO FDConv)
    - Decoder: Standard convolutions
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, **kwargs):
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        
        # Encoder
        self.down_sample_layers = nn.ModuleList()
        ch = chans
        for i in range(num_pool_layers):
            in_ch = in_chans if i == 0 else ch // 2
            self.down_sample_layers.append(ConvBlock(in_ch, ch, drop_prob))
            if i < num_pool_layers - 1:
                ch *= 2
        
        # DCNv2 Skip Aligners
        self.skip_aligners = nn.ModuleList()
        ch_skip = chans
        for i in range(num_pool_layers):
            self.skip_aligners.append(DCNv2SkipRefiner(ch_skip, ch_skip))
            if i < num_pool_layers - 1:
                ch_skip *= 2
        
        # Bottleneck (standard conv, no FDConv)
        self.bottleneck = ConvBlock(ch, ch, drop_prob)
        
        # Decoder
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers.append(ConvBlock(ch * 2, ch // 2, drop_prob))
            ch //= 2
        self.up_sample_layers.append(ConvBlock(ch * 2, ch, drop_prob))
        
        # Output
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
    
    def forward(self, input):
        skip_features = []
        output = input
        
        # Encoder
        for layer in self.down_sample_layers:
            output = layer(output)
            skip_features.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        
        # Bottleneck
        output = self.bottleneck(output)
        
        # Decoder with DCNv2 aligned skips
        for i, layer in enumerate(self.up_sample_layers):
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            skip_idx = len(skip_features) - 1 - i
            skip_aligned = self.skip_aligners[skip_idx](skip_features[skip_idx])
            output = torch.cat([output, skip_aligned], dim=1)
            output = layer(output)
        
        return self.conv2(output)

