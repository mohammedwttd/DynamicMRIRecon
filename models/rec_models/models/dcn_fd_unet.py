"""
Configurable U-Net with optional DCNv2 skip refiners and FDConv bottleneck.

Variants (all from one model):
- Regular U-Net:     use_dcn=False, use_fdconv=False
- Light + DCN:       use_dcn=True,  use_fdconv=False
- Light + FD:        use_dcn=False, use_fdconv=True
- Light + Both:      use_dcn=True,  use_fdconv=True

References:
- DCNv2: "Deformable ConvNets v2" (Zhu et al., CVPR 2019)
- FDConv: "Frequency Dynamic Convolution" (Chen et al., CVPR 2025)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d

from ..layers.fdconv_layer import FDConv


class DCNv2SkipRefiner(nn.Module):
    """
    DCNv2 Skip Connection Refiner.
    
    Based on: "Deformable ConvNets v2" (Zhu et al., CVPR 2019)
    
    Learns offsets and modulation mask from input features to adaptively
    sample spatial locations for skip connection refinement.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1. Learn the Geometry (Offset + Mask)
        self.offset_mask_conv = nn.Conv2d(in_channels, 27, kernel_size=3, padding=1)
        
        # 2. Apply the Geometry (Deformable Conv)
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 3. Non-Linearity
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.zeros_(self.offset_mask_conv.weight)
        nn.init.zeros_(self.offset_mask_conv.bias)
        # Bias mask to ~1.0 to preserve signal at start (sigmoid(2.0) ≈ 0.88)
        with torch.no_grad():
            self.offset_mask_conv.bias[18:] = 2.0
    
    def forward(self, x):
        # Predict
        out = self.offset_mask_conv(x)
        offset = out[:, :18, :, :]
        mask = torch.sigmoid(out[:, 18:, :, :])
        
        # Deform
        x = self.dcn(x, offset, mask)
        
        # Activate
        return self.relu(self.norm(x))


class StandardConvBlock(nn.Module):
    """Standard convolutional block for encoder/decoder."""
    
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
    """FDConv block for bottleneck (global frequency filtering).
    
    Based on: "Frequency Dynamic Convolution" (Chen et al., CVPR 2025)
    """
    
    def __init__(self, in_chans, out_chans, drop_prob, kernel_num=4):
        super().__init__()
        
        # FDConv layers
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


class ODConvBlock(nn.Module):
    """ODConv block for bottleneck (omni-dimensional dynamic convolution).
    
    Based on: "Omni-Dimensional Dynamic Convolution" (Li et al., ICLR 2022)
    
    Uses kernel_num=1 for efficiency while maintaining dynamic attention
    across channel, filter, and spatial dimensions.
    """
    
    def __init__(self, in_chans, out_chans, drop_prob, kernel_num=1):
        super().__init__()
        
        from ..layers.ODConv_layer import ODConv2d
        
        # ODConv layers with single kernel
        self.conv1 = ODConv2d(
            in_planes=in_chans,
            out_planes=out_chans,
            kernel_size=3,
            padding=1,
            kernel_num=kernel_num,
            reduction=0.0625
        )
        self.conv2 = ODConv2d(
            in_planes=out_chans,
            out_planes=out_chans,
            kernel_size=3,
            padding=1,
            kernel_num=kernel_num,
            reduction=0.0625
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


class LightODUnet(nn.Module):
    """
    U-Net with ODConv (Omni-Dimensional Dynamic Convolution) at the bottleneck.
    
    Architecture:
    - Encoder: Standard ConvBlocks
    - Bottleneck: ODConvBlock (dynamic attention across channel, filter, spatial)
    - Decoder: Standard ConvBlocks
    - Optional: DCNv2 skip refiners
    
    ODConv provides input-dependent attention without the overhead of multiple kernels.
    Using kernel_num=1 keeps it efficient while maintaining dynamic properties.
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,
                 use_dcn=False, od_kernel_num=1):
        super(LightODUnet, self).__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.use_dcn = use_dcn
        self.od_kernel_num = od_kernel_num
        
        print(f"*** LightODUnet: od_kernel_num={od_kernel_num}, use_dcn={use_dcn}")
        
        # ==================== ENCODER (Standard Conv) ====================
        self.down_sample_layers = nn.ModuleList()
        
        ch = chans
        for i in range(num_pool_layers):
            in_ch = in_chans if i == 0 else ch // 2
            self.down_sample_layers.append(StandardConvBlock(in_ch, ch, drop_prob))
            if i < num_pool_layers - 1:
                ch *= 2
        
        # ==================== SKIP REFINERS (optional DCN) ====================
        if use_dcn:
            self.skip_refiners = nn.ModuleList()
            ch_skip = chans
            for i in range(num_pool_layers):
                self.skip_refiners.append(DCNv2SkipRefiner(ch_skip, ch_skip))
                if i < num_pool_layers - 1:
                    ch_skip *= 2
        else:
            self.skip_refiners = None
        
        # ==================== BOTTLENECK (ODConv) ====================
        self.bottleneck = ODConvBlock(ch, ch, drop_prob, kernel_num=4)
        
        # ==================== DECODER (Standard Conv) ====================
        self.up_sample_layers = nn.ModuleList()
        
        for i in range(num_pool_layers - 1):
            self.up_sample_layers.append(StandardConvBlock(ch * 2, ch // 2, drop_prob))
            ch //= 2
        
        # Last decoder layer
        self.up_sample_layers.append(StandardConvBlock(ch * 2, ch, drop_prob))
        
        # ==================== OUTPUT ====================
        self.conv_out = nn.Sequential(
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
        
        # Bottleneck (ODConv)
        output = self.bottleneck(output)
        
        # Decoder
        for i, layer in enumerate(self.up_sample_layers):
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            
            skip_idx = len(skip_features) - 1 - i
            skip = skip_features[skip_idx]
            
            # Refine skip with DCNv2 if enabled
            if self.use_dcn and self.skip_refiners is not None:
                skip = self.skip_refiners[skip_idx](skip)
            
            output = torch.cat([output, skip], dim=1)
            output = layer(output)
        
        return self.conv_out(output)
    
    def get_variant_name(self):
        """Return variant name."""
        variant = f"LightOD (k={self.od_kernel_num})"
        if self.use_dcn:
            variant += " + DCN"
        return variant


class ConfigurableUNet(nn.Module):
    """
    Configurable U-Net with optional DCNv2 and FDConv.
    
    Args:
        in_chans (int): Number of input channels
        out_chans (int): Number of output channels
        chans (int): Base number of channels
        num_pool_layers (int): Number of pooling layers
        drop_prob (float): Dropout probability
        use_dcn (bool): Use DCNv2 for skip connection refinement
        use_fdconv (bool): Use FDConv at bottleneck
        fd_kernel_num (int): Number of frequency kernels (if use_fdconv=True)
    
    Variants:
        - Regular U-Net:  use_dcn=False, use_fdconv=False
        - Light + DCN:    use_dcn=True,  use_fdconv=False
        - Light + FD:     use_dcn=False, use_fdconv=True
        - Light + Both:   use_dcn=True,  use_fdconv=True
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,
                 use_dcn=False, use_fdconv=False, fd_kernel_num=4):
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.use_dcn = use_dcn
        self.use_fdconv = use_fdconv
        self.fd_kernel_num = fd_kernel_num
        
        # ==================== ENCODER ====================
        self.down_sample_layers = nn.ModuleList()
        
        ch = chans
        for i in range(num_pool_layers):
            if i == 0:
                self.down_sample_layers.append(StandardConvBlock(in_chans, ch, drop_prob))
            else:
                self.down_sample_layers.append(StandardConvBlock(ch // 2, ch, drop_prob))
            
            if i < num_pool_layers - 1:
                ch *= 2
        
        # ==================== SKIP REFINERS (only if use_dcn=True) ====================
        if use_dcn:
            self.skip_refiners = nn.ModuleList()
            ch = chans
            for i in range(num_pool_layers):
                self.skip_refiners.append(DCNv2SkipRefiner(ch, ch))
                if i < num_pool_layers - 1:
                    ch *= 2
        else:
            self.skip_refiners = None
            # Advance ch for bottleneck calculation
            ch = chans
            for i in range(num_pool_layers - 1):
                ch *= 2
        
        # ==================== BOTTLENECK ====================
        if use_fdconv:
            self.bottleneck = FDConvBlock(ch, ch, drop_prob, kernel_num=fd_kernel_num)
        else:
            self.bottleneck = StandardConvBlock(ch, ch, drop_prob)
        
        # ==================== DECODER ====================
        self.up_sample_layers = nn.ModuleList()
        
        for i in range(num_pool_layers - 1):
            self.up_sample_layers.append(StandardConvBlock(ch * 2, ch // 2, drop_prob))
            ch //= 2
        
        # Last decoder layer
        self.up_sample_layers.append(StandardConvBlock(ch * 2, ch, drop_prob))
        
        # ==================== OUTPUT ====================
        self.conv_out = nn.Sequential(
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
        
        # Decoder
        for i, layer in enumerate(self.up_sample_layers):
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            
            skip_idx = len(skip_features) - 1 - i
            skip = skip_features[skip_idx]
            
            # Refine skip with DCNv2 (only if use_dcn=True)
            if self.use_dcn:
                skip = self.skip_refiners[skip_idx](skip)
            
            output = torch.cat([output, skip], dim=1)
            output = layer(output)
        
        return self.conv_out(output)
    
    def get_variant_name(self):
        """Return variant name based on configuration."""
        if self.use_dcn and self.use_fdconv:
            return "Light + Both (DCN + FD)"
        elif self.use_dcn:
            return "Light + DCN"
        elif self.use_fdconv:
            return "Light + FD"
        else:
            return "Regular U-Net"


# ==================== FULL FD UNET (FDConv everywhere) ====================

class FullFDUnet(nn.Module):
    """
    Full FDConv U-Net - ALL conv blocks use FDConv.
    
    Architecture:
    - Encoder: FDConv blocks (frequency-aware feature extraction)
    - Bottleneck: FDConv block
    - Decoder: FDConv blocks
    - Optional: DCNv2 skip refiners
    
    This maximizes frequency-domain awareness throughout the network.
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,
                 use_dcn=False, fd_kernel_num=4):
        super(FullFDUnet, self).__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.use_dcn = use_dcn
        self.fd_kernel_num = fd_kernel_num
        
        print(f"*** FullFDUnet: kernel_num={fd_kernel_num}, use_dcn={use_dcn}")
        
        # ==================== ENCODER (all FDConv) ====================
        self.down_sample_layers = nn.ModuleList()
        
        ch = chans
        for i in range(num_pool_layers):
            if i == 0:
                self.down_sample_layers.append(
                    FDConvBlock(in_chans, ch, drop_prob, kernel_num=fd_kernel_num)
                )
            else:
                self.down_sample_layers.append(
                    FDConvBlock(ch // 2, ch, drop_prob, kernel_num=fd_kernel_num)
                )
            
            if i < num_pool_layers - 1:
                ch *= 2
        
        # ==================== SKIP REFINERS (optional DCN) ====================
        if use_dcn:
            self.skip_refiners = nn.ModuleList()
            ch_skip = chans
            for i in range(num_pool_layers):
                self.skip_refiners.append(DCNv2SkipRefiner(ch_skip, ch_skip))
                if i < num_pool_layers - 1:
                    ch_skip *= 2
        else:
            self.skip_refiners = None
            # Advance ch for bottleneck calculation
            ch = chans
            for i in range(num_pool_layers - 1):
                ch *= 2
        
        # ==================== BOTTLENECK (FDConv) ====================
        self.bottleneck = FDConvBlock(ch, ch, drop_prob, kernel_num=fd_kernel_num)
        
        # ==================== DECODER (all FDConv) ====================
        self.up_sample_layers = nn.ModuleList()
        
        for i in range(num_pool_layers - 1):
            self.up_sample_layers.append(
                FDConvBlock(ch * 2, ch // 2, drop_prob, kernel_num=fd_kernel_num)
            )
            ch //= 2
        
        # Last decoder layer
        self.up_sample_layers.append(
            FDConvBlock(ch * 2, ch, drop_prob, kernel_num=fd_kernel_num)
        )
        
        # ==================== OUTPUT ====================
        self.conv_out = nn.Sequential(
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
        
        # Decoder
        for i, layer in enumerate(self.up_sample_layers):
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            
            skip_idx = len(skip_features) - 1 - i
            skip = skip_features[skip_idx]
            
            # Refine skip with DCNv2 (only if use_dcn=True)
            if self.use_dcn and self.skip_refiners is not None:
                skip = self.skip_refiners[skip_idx](skip)
            
            output = torch.cat([output, skip], dim=1)
            output = layer(output)
        
        return self.conv_out(output)
    
    def get_variant_name(self):
        """Return variant name."""
        if self.use_dcn:
            return "FullFD + DCN"
        return "FullFD"


# ==================== HYBRID FD UNET (FDConv at deeper levels only) ====================

class HybridFDUnet(nn.Module):
    """
    Hybrid FDConv U-Net - FDConv only at deeper levels.
    
    Architecture (for num_pool_layers=4):
    - Level 1 (320x320): StandardConv - Preserve edges/details
    - Level 2 (160x160): StandardConv - Local features
    - Level 3 (80x80):   FDConv - Start capturing global context
    - Level 4 (40x40):   FDConv - Global frequency patterns
    - Bottleneck (20x20): FDConv - Maximum global context
    
    Decoder mirrors encoder:
    - Up 1 (40x40):  FDConv
    - Up 2 (80x80):  FDConv  
    - Up 3 (160x160): StandardConv
    - Up 4 (320x320): StandardConv
    
    Rationale:
    - Early layers preserve high-frequency details (edges, textures)
    - Deep layers benefit from frequency-domain awareness for global patterns
    - Optional DCNv2 skip refiners for adaptive alignment
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,
                 use_dcn=False, fd_kernel_num=4, fd_start_level=2):
        """
        Args:
            fd_start_level: Level at which to start using FDConv (0-indexed).
                           Default=2 means levels 0,1 use StandardConv, levels 2+ use FDConv.
        """
        super(HybridFDUnet, self).__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.use_dcn = use_dcn
        self.fd_kernel_num = fd_kernel_num
        self.fd_start_level = fd_start_level
        
        print(f"*** HybridFDUnet: fd_start_level={fd_start_level}, kernel_num={fd_kernel_num}, use_dcn={use_dcn}")
        print(f"    Levels 0-{fd_start_level-1}: StandardConv | Levels {fd_start_level}-{num_pool_layers-1}+bottleneck: FDConv")
        
        # ==================== ENCODER (hybrid) ====================
        self.down_sample_layers = nn.ModuleList()
        
        ch = chans
        for i in range(num_pool_layers):
            in_ch = in_chans if i == 0 else ch // 2
            
            if i < fd_start_level:
                # Early levels: StandardConv (preserve edges)
                self.down_sample_layers.append(
                    StandardConvBlock(in_ch, ch, drop_prob)
                )
            else:
                # Deep levels: FDConv (global frequency context)
                self.down_sample_layers.append(
                    FDConvBlock(in_ch, ch, drop_prob, kernel_num=fd_kernel_num)
                )
            
            if i < num_pool_layers - 1:
                ch *= 2
        
        # ==================== SKIP REFINERS (optional DCN) ====================
        if use_dcn:
            self.skip_refiners = nn.ModuleList()
            ch_skip = chans
            for i in range(num_pool_layers):
                self.skip_refiners.append(DCNv2SkipRefiner(ch_skip, ch_skip))
                if i < num_pool_layers - 1:
                    ch_skip *= 2
        else:
            self.skip_refiners = None
        
        # ==================== BOTTLENECK (FDConv - deepest level) ====================
        self.bottleneck = FDConvBlock(ch, ch, drop_prob, kernel_num=fd_kernel_num)
        
        # ==================== DECODER (hybrid, mirrored) ====================
        self.up_sample_layers = nn.ModuleList()
        
        for i in range(num_pool_layers - 1):
            # Mirror the encoder: deep levels use FDConv, early levels use StandardConv
            decoder_level = num_pool_layers - 1 - i  # Maps to encoder level
            
            if decoder_level >= fd_start_level:
                self.up_sample_layers.append(
                    FDConvBlock(ch * 2, ch // 2, drop_prob, kernel_num=fd_kernel_num)
                )
            else:
                self.up_sample_layers.append(
                    StandardConvBlock(ch * 2, ch // 2, drop_prob)
                )
            ch //= 2
        
        # Last decoder layer (level 0)
        if 0 >= fd_start_level:
            self.up_sample_layers.append(
                FDConvBlock(ch * 2, ch, drop_prob, kernel_num=fd_kernel_num)
            )
        else:
            self.up_sample_layers.append(
                StandardConvBlock(ch * 2, ch, drop_prob)
            )
        
        # ==================== OUTPUT ====================
        self.conv_out = nn.Sequential(
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
        
        # Decoder
        for i, layer in enumerate(self.up_sample_layers):
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            
            skip_idx = len(skip_features) - 1 - i
            skip = skip_features[skip_idx]
            
            # Refine skip with DCNv2 if enabled
            if self.use_dcn and self.skip_refiners is not None:
                skip = self.skip_refiners[skip_idx](skip)
            
            output = torch.cat([output, skip], dim=1)
            output = layer(output)
        
        return self.conv_out(output)
    
    def get_variant_name(self):
        """Return variant name."""
        variant = f"HybridFD (L{self.fd_start_level}+)"
        if self.use_dcn:
            variant += " + DCN"
        return variant


# ==================== CONVENIENCE ALIASES ====================

class DCNFDUnet(ConfigurableUNet):
    """Alias: U-Net with both DCN and FDConv (Light + Both)."""
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,
                 fd_kernel_num=4, fd_use_simple=False):
        super().__init__(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            use_dcn=True,
            use_fdconv=True,
            fd_kernel_num=fd_kernel_num
        )


class SmallDCNFDUnet(ConfigurableUNet):
    """Alias: Smaller U-Net with both DCN and FDConv (~2M params)."""
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,
                 fd_kernel_num=2, fd_use_simple=False):
        super().__init__(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            use_dcn=True,
            use_fdconv=True,
            fd_kernel_num=fd_kernel_num
        )


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 80)
    print("Testing Configurable U-Net Variants")
    print("=" * 80)
    
    configs = [
        {"use_dcn": False, "use_fdconv": False, "chans": 32},  # Regular
        {"use_dcn": True,  "use_fdconv": False, "chans": 32},  # + DCN
        {"use_dcn": False, "use_fdconv": True,  "chans": 32},  # + FD
        {"use_dcn": True,  "use_fdconv": True,  "chans": 32},  # + Both
    ]
    
    for cfg in configs:
        model = ConfigurableUNet(
            in_chans=2,
            out_chans=2,
            chans=cfg["chans"],
            num_pool_layers=4,
            drop_prob=0.0,
            use_dcn=cfg["use_dcn"],
            use_fdconv=cfg["use_fdconv"],
            fd_kernel_num=4
        )
        
        params = count_parameters(model)
        name = model.get_variant_name()
        
        print(f"\n{name}:")
        print(f"  use_dcn={cfg['use_dcn']}, use_fdconv={cfg['use_fdconv']}")
        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
        
        # Test forward pass
        x = torch.randn(1, 2, 320, 320)
        with torch.no_grad():
            y = model(x)
        print(f"  Forward: {x.shape} -> {y.shape} ✓")
    
    print("\n" + "=" * 80)
    print("All variants working!")
    print("=" * 80)
