"""
Hybrid Conditional U-Net: Uses CondConv only at the bottleneck.

This model tests whether using CondConv at the most abstract layer (bottleneck)
provides benefits while keeping parameter count lower than full CondUnet.

Design: Regular ConvBlocks in encoder/decoder, CondConvBlock only at bottleneck
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.rec_models.cond_unet_model import CondConvBlock


class ConvBlock(nn.Module):
    """Regular ConvBlock (same as in unet_model.py)."""
    
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )
    
    def forward(self, input):
        return self.layers(input)


class HybridCondUnetModel(nn.Module):
    """
    Hybrid U-Net with CondConv only at bottleneck.
    
    Uses reduced base channels to keep parameter count lower than regular U-Net,
    but adds CondConv at bottleneck for adaptive high-level feature processing.
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, num_experts=8):
        """
        Args:
            in_chans (int): Number of input channels.
            out_chans (int): Number of output channels.
            chans (int): Base number of channels (will be smaller than regular U-Net).
            num_pool_layers (int): Number of down/up-sampling layers.
            drop_prob (float): Dropout probability.
            num_experts (int): Number of experts for bottleneck CondConv.
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.num_experts = num_experts
        
        # Encoder: Regular ConvBlocks
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        
        # Bottleneck: CondConvBlock (the only place with CondConv)
        self.conv = CondConvBlock(ch, ch, drop_prob, num_experts)
        
        # Decoder: Regular ConvBlocks
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        
        # Final output layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
    
    def forward(self, input):
        """Forward pass through hybrid U-Net."""
        stack = []
        output = input
        
        # Encoder
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        
        # Bottleneck (with CondConv)
        output = self.conv(output)
        
        # Decoder
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        
        return self.conv2(output)
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (f'HybridCondUnetModel(in_chans={self.in_chans}, out_chans={self.out_chans}, '
                f'chans={self.chans}, num_pool_layers={self.num_pool_layers}, '
                f'num_experts={self.num_experts})')

