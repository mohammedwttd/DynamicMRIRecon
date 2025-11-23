"""
FD-UNet: U-Net with Frequency Dynamic Convolution.

Based on FDConv from "Frequency Dynamic Convolution for Dense Image Prediction" (CVPR 2025)
GitHub: https://github.com/Linwei-Chen/FDConv

Perfect for MRI reconstruction because:
1. MRI data is naturally in frequency domain (k-space)
2. FDConv operates in Fourier domain - natural fit!
3. Super parameter efficient (+3.6M vs +90M for CondConv)
4. State-of-the-art for dense prediction tasks
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.rec_models.fdconv_layer import FDConv, FDConv_Simple


class FDConvBlock(nn.Module):
    """
    Convolutional Block using FDConv.
    
    Two FDConv layers with instance normalization, ReLU, and dropout.
    """
    
    def __init__(self, in_chans, out_chans, drop_prob, kernel_num=64, use_simple=False):
        """
        Args:
            in_chans (int): Number of input channels
            out_chans (int): Number of output channels
            drop_prob (float): Dropout probability
            kernel_num (int): Number of frequency-diverse kernels for FDConv
            use_simple (bool): If True, use simplified FDConv for speed
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.kernel_num = kernel_num
        
        # Choose FDConv variant
        FDConvLayer = FDConv_Simple if use_simple else FDConv
        
        # First FDConv layer
        if use_simple:
            self.conv1 = FDConvLayer(in_chans, out_chans, kernel_size=3, padding=1)
        else:
            self.conv1 = FDConvLayer(in_chans, out_chans, kernel_size=3, padding=1, kernel_num=kernel_num)
        self.norm1 = nn.InstanceNorm2d(out_chans)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout2d(drop_prob)
        
        # Second FDConv layer
        if use_simple:
            self.conv2 = FDConvLayer(out_chans, out_chans, kernel_size=3, padding=1)
        else:
            self.conv2 = FDConvLayer(out_chans, out_chans, kernel_size=3, padding=1, kernel_num=kernel_num)
        self.norm2 = nn.InstanceNorm2d(out_chans)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout2d(drop_prob)
    
    def forward(self, input):
        """
        Args:
            input: [batch, in_chans, height, width]
        
        Returns:
            output: [batch, out_chans, height, width]
        """
        output = self.conv1(input)
        output = self.norm1(output)
        output = self.relu1(output)
        output = self.drop1(output)
        
        output = self.conv2(output)
        output = self.norm2(output)
        output = self.relu2(output)
        output = self.drop2(output)
        
        return output
    
    def __repr__(self):
        return (f'FDConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, '
                f'drop_prob={self.drop_prob}, kernel_num={self.kernel_num})')


class FDUnetModel(nn.Module):
    """
    U-Net with Frequency Dynamic Convolution (FD-UNet).
    
    Replaces standard Conv2d with FDConv for frequency-aware feature extraction.
    Particularly suited for MRI reconstruction where data is naturally in k-space (frequency domain).
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, 
                 kernel_num=64, use_simple=False):
        """
        Args:
            in_chans (int): Number of input channels
            out_chans (int): Number of output channels
            chans (int): Base number of channels
            num_pool_layers (int): Number of down/up-sampling layers
            drop_prob (float): Dropout probability
            kernel_num (int): Number of frequency-diverse kernels (default: 64)
            use_simple (bool): Use simplified FDConv for faster computation
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.kernel_num = kernel_num
        self.use_simple = use_simple
        
        # Encoder (down-sampling path)
        self.down_sample_layers = nn.ModuleList([
            FDConvBlock(in_chans, chans, drop_prob, kernel_num, use_simple)
        ])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [
                FDConvBlock(ch, ch * 2, drop_prob, kernel_num, use_simple)
            ]
            ch *= 2
        
        # Bottleneck
        self.conv = FDConvBlock(ch, ch, drop_prob, kernel_num, use_simple)
        
        # Decoder (up-sampling path)
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [
                FDConvBlock(ch * 2, ch // 2, drop_prob, kernel_num, use_simple)
            ]
            ch //= 2
        self.up_sample_layers += [
            FDConvBlock(ch * 2, ch, drop_prob, kernel_num, use_simple)
        ]
        
        # Final output layers (regular Conv2d for efficiency)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
    
    def forward(self, input):
        """
        Forward pass through FD-UNet.
        
        Args:
            input: [batch, in_chans, height, width]
        
        Returns:
            output: [batch, out_chans, height, width]
        """
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
    
    @staticmethod
    def from_pretrained_unet(unet_model, kernel_num=64):
        """
        Convert a pretrained regular U-Net to FD-UNet by transforming Conv2d weights.
        
        Args:
            unet_model: Pretrained UnetModel
            kernel_num: Number of frequency components
        
        Returns:
            FDUnetModel with converted weights
        """
        from models.rec_models.unet_model import UnetModel
        
        if not isinstance(unet_model, UnetModel):
            raise ValueError("Input must be a UnetModel instance")
        
        # Create FD-UNet with same architecture
        fd_unet = FDUnetModel(
            in_chans=unet_model.in_chans,
            out_chans=unet_model.out_chans,
            chans=unet_model.chans,
            num_pool_layers=unet_model.num_pool_layers,
            drop_prob=unet_model.drop_prob,
            kernel_num=kernel_num
        )
        
        # Convert encoder blocks
        for i, (unet_block, fd_block) in enumerate(zip(unet_model.down_sample_layers, 
                                                        fd_unet.down_sample_layers)):
            # Extract Conv2d layers from Sequential in original U-Net
            conv1 = unet_block.layers[0]  # First Conv2d
            conv2 = unet_block.layers[4]  # Second Conv2d
            
            # Convert to FDConv
            fd_block.conv1 = FDConv.convert_from_conv2d(conv1, kernel_num)
            fd_block.conv2 = FDConv.convert_from_conv2d(conv2, kernel_num)
        
        # Convert bottleneck
        conv1 = unet_model.conv.layers[0]
        conv2 = unet_model.conv.layers[4]
        fd_unet.conv.conv1 = FDConv.convert_from_conv2d(conv1, kernel_num)
        fd_unet.conv.conv2 = FDConv.convert_from_conv2d(conv2, kernel_num)
        
        # Convert decoder blocks
        for i, (unet_block, fd_block) in enumerate(zip(unet_model.up_sample_layers, 
                                                        fd_unet.up_sample_layers)):
            conv1 = unet_block.layers[0]
            conv2 = unet_block.layers[4]
            fd_block.conv1 = FDConv.convert_from_conv2d(conv1, kernel_num)
            fd_block.conv2 = FDConv.convert_from_conv2d(conv2, kernel_num)
        
        # Copy final layers (keep as regular Conv2d)
        fd_unet.conv2.load_state_dict(unet_model.conv2.state_dict())
        
        return fd_unet
    
    def __repr__(self):
        return (f'FDUnetModel(in_chans={self.in_chans}, out_chans={self.out_chans}, '
                f'chans={self.chans}, num_pool_layers={self.num_pool_layers}, '
                f'kernel_num={self.kernel_num}, use_simple={self.use_simple})')

