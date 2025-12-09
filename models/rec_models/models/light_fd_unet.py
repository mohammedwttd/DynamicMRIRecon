"""
Light + FD U-Net: Light U-Net with FDConv bottleneck only (no DCNv2).

For ablation study comparing:
- Light U-Net (baseline reduced)
- Light + DCN
- Light + FD (this model)
- Light + Both (DCNFDUnet)
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..layers.fdconv_layer import FDConv


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


class FDConvBlock(nn.Module):
    """FDConv block for bottleneck."""
    
    def __init__(self, in_chans, out_chans, drop_prob, kernel_num=4, use_simple=False):
        super().__init__()
        
        if use_simple:
            self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        else:
            self.conv1 = FDConv(
                in_channels=in_chans, 
                out_channels=out_chans, 
                kernel_size=3, 
                padding=1, 
                kernel_num=kernel_num,
                use_fdconv_if_c_gt=16,
                fbm_cfg={
                    'k_list': [2, 4, 8],
                    'lowfreq_att': False,
                    'fs_feat': 'feat',
                    'act': 'sigmoid',
                    'spatial': 'conv',
                    'spatial_group': 1,
                    'spatial_kernel': 3,
                    'init': 'zero',
                }
            )
            self.conv2 = FDConv(
                in_channels=out_chans, 
                out_channels=out_chans, 
                kernel_size=3, 
                padding=1, 
                kernel_num=kernel_num,
                use_fdconv_if_c_gt=16,
                fbm_cfg={
                    'k_list': [2, 4, 8],
                    'lowfreq_att': False,
                    'fs_feat': 'feat',
                    'act': 'sigmoid',
                    'spatial': 'conv',
                    'spatial_group': 1,
                    'spatial_kernel': 3,
                    'init': 'zero',
                }
            )
        
        self.norm1 = nn.InstanceNorm2d(out_chans)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(drop_prob)
        self.norm2 = nn.InstanceNorm2d(out_chans)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d(drop_prob)
    
    def forward(self, x):
        out = self.drop1(self.relu1(self.norm1(self.conv1(x))))
        out = self.drop2(self.relu2(self.norm2(self.conv2(out))))
        return out


class LightFDUnet(nn.Module):
    """
    Light U-Net + FDConv bottleneck (no DCNv2 skip aligners).
    
    Architecture:
    - Encoder: Standard convolutions
    - Skip Connections: Direct (no alignment)
    - Bottleneck: FDConv (frequency filtering)
    - Decoder: Standard convolutions
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,
                 fd_kernel_num=4, fd_use_simple=False, **kwargs):
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
        
        # Bottleneck with FDConv
        self.bottleneck = FDConvBlock(ch, ch, drop_prob, 
                                      kernel_num=fd_kernel_num, 
                                      use_simple=fd_use_simple)
        
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
        
        # Bottleneck (FDConv)
        output = self.bottleneck(output)
        
        # Decoder with direct skips (no DCN alignment)
        for i, layer in enumerate(self.up_sample_layers):
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            skip_idx = len(skip_features) - 1 - i
            output = torch.cat([output, skip_features[skip_idx]], dim=1)
            output = layer(output)
        
        return self.conv2(output)

