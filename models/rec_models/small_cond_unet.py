"""
Small Conditional U-Net: Uses CondConv everywhere with reduced base channels.

This model tests whether using CondConv throughout the network with fewer base
channels can match or exceed regular U-Net performance with similar or fewer parameters.

Design: CondConvBlocks everywhere, but smaller base channel count
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.rec_models.cond_unet_model import CondConvBlock


class SmallCondUnetModel(nn.Module):
    """
    Small U-Net with CondConv at every layer.
    
    Uses CondConv everywhere but with reduced base channels to keep parameter
    count similar to or lower than regular U-Net. Tests if conditional computation
    can be more parameter-efficient than static large networks.
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, num_experts=8):
        """
        Args:
            in_chans (int): Number of input channels.
            out_chans (int): Number of output channels.
            chans (int): Base number of channels (smaller than regular U-Net).
            num_pool_layers (int): Number of down/up-sampling layers.
            drop_prob (float): Dropout probability.
            num_experts (int): Number of experts for all CondConv layers.
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.num_experts = num_experts
        
        # Encoder: CondConvBlocks
        self.down_sample_layers = nn.ModuleList([
            CondConvBlock(in_chans, chans, drop_prob, num_experts)
        ])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [CondConvBlock(ch, ch * 2, drop_prob, num_experts)]
            ch *= 2
        
        # Bottleneck: CondConvBlock
        self.conv = CondConvBlock(ch, ch, drop_prob, num_experts)
        
        # Decoder: CondConvBlocks
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [CondConvBlock(ch * 2, ch // 2, drop_prob, num_experts)]
            ch //= 2
        self.up_sample_layers += [CondConvBlock(ch * 2, ch, drop_prob, num_experts)]
        
        # Final output layers (regular Conv for efficiency)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
    
    def forward(self, input):
        """Forward pass through small CondUnet."""
        stack = []
        output = input
        
        # Encoder
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        
        # Bottleneck
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
    
    def get_routing_weights(self, input):
        """
        Get routing weights for all CondConv layers (for analysis).
        
        Args:
            input (torch.Tensor): Input tensor
        
        Returns:
            dict: Dictionary mapping layer names to routing weights
        """
        routing_weights = {}
        
        def get_block_routing(block, x, prefix):
            with torch.no_grad():
                w1 = block.conv1.routing(x)
                routing_weights[f'{prefix}_conv1'] = w1.detach().cpu()
                
                x_mid = block.conv1(x)
                x_mid = block.norm1(x_mid)
                x_mid = block.relu1(x_mid)
                
                w2 = block.conv2.routing(x_mid)
                routing_weights[f'{prefix}_conv2'] = w2.detach().cpu()
            
            return x_mid
        
        with torch.no_grad():
            output = input
            
            # Encoder
            for i, layer in enumerate(self.down_sample_layers):
                get_block_routing(layer, output, f'down_{i}')
                output = layer(output)
                output = F.max_pool2d(output, kernel_size=2)
            
            # Bottleneck
            get_block_routing(self.conv, output, 'bottleneck')
        
        return routing_weights
    
    def __repr__(self):
        return (f'SmallCondUnetModel(in_chans={self.in_chans}, out_chans={self.out_chans}, '
                f'chans={self.chans}, num_pool_layers={self.num_pool_layers}, '
                f'num_experts={self.num_experts})')

