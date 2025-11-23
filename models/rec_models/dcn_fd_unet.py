"""
DCN-FD U-Net: Lightweight architecture combining Deformable Convolution v2 and Frequency Dynamic Convolution.

Architecture:
- Encoder: Standard convolutions with downsampling
- Skip Aligners: DCNv2 aligns skip connections before concatenation
- Bottleneck: FDConv for global frequency filtering
- Decoder: Standard upsampling + convolutions

Key Innovation: DCNv2 adaptively aligns multi-scale features from skip connections,
while FDConv handles global frequency-domain processing at the bottleneck.

Target: ~1.25M parameters (50% of baseline U-Net)
"""

import torch
from torch import nn
from torch.nn import functional as F
import math

try:
    from torchvision.ops import DeformConv2d
    DCN_AVAILABLE = True
except ImportError:
    DCN_AVAILABLE = False
    print("⚠️  Warning: torchvision.ops.DeformConv2d not available. Install with: pip install torchvision>=0.9.0")

from .fdconv_layer import FDConv, FDConv_Simple


class DCNv2SkipAligner(nn.Module):
    """
    Deformable Convolution v2 for adaptive skip connection alignment.
    
    DCNv2 learns spatial offsets and modulation masks to align features
    from different scales before concatenation.
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if not DCN_AVAILABLE:
            # Fallback to standard convolution if DCN not available
            self.use_dcn = False
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.use_dcn = True
            
            # Offset and mask prediction network
            # Predicts: 2 * kh * kw offsets + kh * kw masks
            self.offset_mask = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 3 * 3 * 3, kernel_size=1)  # 18 offsets + 9 masks
            )
            
            # Deformable convolution
            self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize offset prediction to output zeros (identity deformation)."""
        if self.use_dcn:
            nn.init.constant_(self.offset_mask[-1].weight, 0)
            nn.init.constant_(self.offset_mask[-1].bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        
        Returns:
            Aligned feature map [B, out_channels, H, W]
        """
        if not self.use_dcn:
            # Fallback: standard convolution
            out = self.conv(x)
        else:
            # Predict offsets and modulation masks
            offset_mask = self.offset_mask(x)  # [B, 27, H, W]
            
            # Split into offsets and mask
            offset = offset_mask[:, :18, :, :]  # [B, 18, H, W] (2 * 9 for x,y offsets)
            mask = torch.sigmoid(offset_mask[:, 18:, :, :])  # [B, 9, H, W] (modulation)
            
            # Apply deformable convolution
            out = self.dcn(x, offset, mask)
        
        out = self.norm(out)
        out = self.relu(out)
        
        return out


class StandardConvBlock(nn.Module):
    """Standard convolutional block for encoder/decoder."""
    
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        
        self.layers = nn.Sequential(
            # First standard conv
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_prob),
            
            # Second standard conv
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_prob)
        )
    
    def forward(self, x):
        return self.layers(x)


class FDConvBlock(nn.Module):
    """FDConv block for bottleneck (global frequency filtering)."""
    
    def __init__(self, in_chans, out_chans, drop_prob, kernel_num=4, use_simple=False):
        super().__init__()
        
        FDConvLayer = FDConv_Simple if use_simple else FDConv
        
        # First FDConv
        if use_simple:
            self.conv1 = FDConvLayer(in_chans, out_chans, kernel_size=3, padding=1)
        else:
            self.conv1 = FDConvLayer(in_chans, out_chans, kernel_size=3, padding=1, kernel_num=kernel_num)
        self.norm1 = nn.InstanceNorm2d(out_chans)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(drop_prob)
        
        # Second FDConv
        if use_simple:
            self.conv2 = FDConvLayer(out_chans, out_chans, kernel_size=3, padding=1)
        else:
            self.conv2 = FDConvLayer(out_chans, out_chans, kernel_size=3, padding=1, kernel_num=kernel_num)
        self.norm2 = nn.InstanceNorm2d(out_chans)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d(drop_prob)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        
        return out


class DCNFDUnet(nn.Module):
    """
    DCN-FD U-Net.
    
    Architecture:
    - Encoder: Standard convolutions
    - Skip Aligners: DCNv2 for adaptive feature alignment
    - Bottleneck: FDConv for global frequency filtering
    - Decoder: Standard convolutions
    
    Target: ~2.5M parameters (similar to baseline U-Net)
    
    Args:
        in_chans (int): Number of input channels
        out_chans (int): Number of output channels
        chans (int): Base number of channels (use ~16-20 for lightweight)
        num_pool_layers (int): Number of pooling layers
        drop_prob (float): Dropout probability
        fd_kernel_num (int): Number of frequency kernels in FDConv
        fd_use_simple (bool): Use simplified FDConv
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,
                 fd_kernel_num=4, fd_use_simple=False):
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        
        # ==================== ENCODER ====================
        self.down_sample_layers = nn.ModuleList()
        
        ch = chans
        for i in range(num_pool_layers):
            if i == 0:
                self.down_sample_layers.append(
                    StandardConvBlock(in_chans, ch, drop_prob)
                )
            else:
                self.down_sample_layers.append(
                    StandardConvBlock(ch // 2, ch, drop_prob)
                )
            
            if i < num_pool_layers - 1:
                ch *= 2
        
        # ==================== SKIP ALIGNERS (DCNv2) ====================
        self.skip_aligners = nn.ModuleList()
        
        ch = chans
        for i in range(num_pool_layers):
            # DCNv2 aligns skip features before concatenation
            self.skip_aligners.append(
                DCNv2SkipAligner(ch, ch)
            )
            
            if i < num_pool_layers - 1:
                ch *= 2
        
        # ==================== BOTTLENECK (FDConv) ====================
        self.bottleneck = FDConvBlock(ch, ch, drop_prob, 
                                     kernel_num=fd_kernel_num, 
                                     use_simple=fd_use_simple)
        
        # ==================== DECODER ====================
        self.up_sample_layers = nn.ModuleList()
        
        for i in range(num_pool_layers - 1):
            # After concat: ch * 2 → ch // 2
            self.up_sample_layers.append(
                StandardConvBlock(ch * 2, ch // 2, drop_prob)
            )
            ch //= 2
        
        # Last decoder layer
        self.up_sample_layers.append(
            StandardConvBlock(ch * 2, ch, drop_prob)
        )
        
        # ==================== OUTPUT ====================
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
    
    def forward(self, input):
        """
        Forward pass through DCN-FD U-Net.
        
        Args:
            input (torch.Tensor): Input tensor [B, in_chans, H, W]
        
        Returns:
            torch.Tensor: Reconstructed output [B, out_chans, H, W]
        """
        skip_features = []
        output = input
        
        # ==================== ENCODER PATH ====================
        for layer in self.down_sample_layers:
            output = layer(output)
            skip_features.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        
        # ==================== BOTTLENECK (FDConv) ====================
        output = self.bottleneck(output)
        
        # ==================== DECODER PATH ====================
        for i, layer in enumerate(self.up_sample_layers):
            # Upsample
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            
            # Get corresponding skip feature
            skip_idx = len(skip_features) - 1 - i
            skip = skip_features[skip_idx]
            
            # ALIGN skip feature using DCNv2
            skip_aligned = self.skip_aligners[skip_idx](skip)
            
            # Concatenate aligned skip with upsampled features
            output = torch.cat([output, skip_aligned], dim=1)
            
            # Decode
            output = layer(output)
        
        # ==================== OUTPUT ====================
        return self.conv2(output)
    
    def get_architecture_summary(self):
        """Return a summary of the architecture."""
        return {
            'encoder': 'Standard Conv (regular)',
            'skip_aligners': 'DCNv2 (adaptive alignment)',
            'bottleneck': 'FDConv (frequency filtering)',
            'decoder': 'Standard Conv (regular)',
            'target_params': '~2.5M (similar to baseline)',
            'innovation': 'DCNv2 skip alignment + FDConv global filtering'
        }


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    print("Testing DCN-FD U-Net...")
    print("=" * 80)
    
    # Create lightweight model
    model = DCNFDUnet(
        in_chans=2,
        out_chans=2,
        chans=20,  # Use 20 for ~1.25M params (lightweight)
        num_pool_layers=4,
        drop_prob=0.0,
        fd_kernel_num=4,
        fd_use_simple=False
    )
    
    # Print architecture summary
    summary = model.get_architecture_summary()
    print("\nArchitecture Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nTotal parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Check if it meets target
    target = 1.25e6
    ratio = params / target
    if 0.9 <= ratio <= 1.1:
        print(f"✓ Parameter count within target range (±10%)")
    else:
        print(f"⚠ Parameter count off target by {(ratio-1)*100:.1f}%")
        if ratio > 1.1:
            print(f"  → Reduce 'chans' to {int(20 * 0.9)}-{int(20 * 0.95)}")
        else:
            print(f"  → Increase 'chans' to {int(20 * 1.05)}-{int(20 * 1.1)}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 2, 320, 320)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    
    if DCN_AVAILABLE:
        print(f"\n✓ DCNv2 available - using deformable convolution for skip alignment")
    else:
        print(f"\n⚠ DCNv2 not available - using standard convolution (install torchvision>=0.9.0)")
    
    print("\n✓ Model test passed!")
    print("=" * 80)

