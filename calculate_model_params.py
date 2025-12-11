#!/usr/bin/env python3
"""
Model Parameter Calculator

Computes and compares parameters for different U-Net variants by calculating
overhead of each component separately.

Uses official FDConv from Chen et al. (CVPR 2025).
"""

import torch
import torch.nn as nn
import math
import sys
sys.path.insert(0, '/home/mohammed-wa/dynamic_mri/DynamicMRIRecon')

from models.rec_models.models.unet_model import UnetModel, ConvBlock


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(count):
    """Format parameter count nicely"""
    if count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    return str(count)


def calculate_dcn_overhead(chans=20, num_pool_layers=4):
    """Calculate ONLY the DCNv2 skip aligner overhead"""
    from torchvision.ops import DeformConv2d
    
    total = 0
    ch = chans
    for i in range(num_pool_layers):
        # Offset predictor: 18 = 2 * 3 * 3 (2D offsets for 3x3 kernel)
        offset = nn.Conv2d(ch, 18, kernel_size=3, padding=1)
        dcn = DeformConv2d(ch, ch, kernel_size=3, padding=1)
        total += count_parameters(offset) + count_parameters(dcn)
        ch *= 2
    
    return total


def calculate_fdconv_overhead(chans=20, num_pool_layers=4, kernel_num=4):
    """Calculate ONLY the official FDConv bottleneck overhead (vs standard conv)
    
    Uses official FDConv from Chen et al. (CVPR 2025) - Fourier Disjoint Weights.
    """
    from models.rec_models.layers.fdconv_layer import FDConv
    
    # Bottleneck channels
    bottleneck_in = chans * (2 ** (num_pool_layers - 1))  # 160
    bottleneck_out = chans * (2 ** num_pool_layers)       # 320
    
    # Standard ConvBlock at bottleneck (what Light U-Net uses)
    std_block = ConvBlock(bottleneck_in, bottleneck_out, drop_prob=0.0)
    std_params = count_parameters(std_block)
    
    # Official FDConv block at bottleneck
    fd_block = nn.Sequential(
        FDConv(
            in_channels=bottleneck_in, 
            out_channels=bottleneck_out, 
            kernel_size=3, 
            padding=1, 
            kernel_num=kernel_num,
            use_fdconv_if_c_gt=16,  # Only activate FDConv if channels > 16
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
        ),
        nn.InstanceNorm2d(bottleneck_out, affine=True),
        nn.ReLU(inplace=True),
        FDConv(
            in_channels=bottleneck_out, 
            out_channels=bottleneck_out, 
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
        ),
        nn.InstanceNorm2d(bottleneck_out, affine=True),
        nn.ReLU(inplace=True),
    )
    fd_params = count_parameters(fd_block)
    
    # Overhead is the difference
    overhead = fd_params - std_params
    
    return overhead, std_params, fd_params


def main():
    print("=" * 80)
    print("MODEL PARAMETER COMPARISON")
    print("Using official FDConv from Chen et al. (CVPR 2025)")
    print("=" * 80)
    print()
    
    # Configuration
    in_chans = 1
    out_chans = 1
    num_pool_layers = 4
    drop_prob = 0.0
    kernel_num = 4  # Number of frequency kernels
    chans_regular = 32
    chans_light = 20
    
    # 1. Regular U-Net (32 channels) - BASELINE
    print("Calculating Regular U-Net (32 channels)...")
    regular_unet = UnetModel(in_chans, out_chans, chans=chans_regular, 
                             num_pool_layers=num_pool_layers, drop_prob=drop_prob)
    regular_params = count_parameters(regular_unet)
    
    # 2. Light U-Net (20 channels)
    print("Calculating Light U-Net (20 channels)...")
    light_unet = UnetModel(in_chans, out_chans, chans=chans_light, 
                           num_pool_layers=num_pool_layers, drop_prob=drop_prob)
    light_params = count_parameters(light_unet)
    
    # 3. DCNv2 skip aligner overhead
    print("Calculating DCNv2 skip aligner overhead...")
    dcn_overhead = calculate_dcn_overhead(chans=chans_light, num_pool_layers=num_pool_layers)
    
    # 4. FDConv bottleneck overhead (official version)
    print("Calculating FDConv (official) bottleneck overhead...")
    fd_overhead, std_bottleneck, fd_bottleneck = calculate_fdconv_overhead(
        chans=chans_light, num_pool_layers=num_pool_layers, kernel_num=kernel_num
    )
    
    print()
    print("=" * 80)
    print("COMPONENT BREAKDOWN")
    print("=" * 80)
    print()
    print(f"Regular U-Net (32 ch):     {format_params(regular_params)}")
    print(f"Light U-Net (20 ch):       {format_params(light_params)}")
    print()
    print(f"DCNv2 skip aligners (4x):  +{format_params(dcn_overhead)}")
    print(f"  (offset conv + DCNv2 at each encoder level)")
    print()
    print(f"Official FDConv bottleneck:")
    print(f"  Standard ConvBlock:      {format_params(std_bottleneck)}")
    print(f"  FDConv block (kn={kernel_num}):     {format_params(fd_bottleneck)}")
    print(f"  Overhead:                +{format_params(fd_overhead)}")
    
    # Calculate model totals
    light_dcn_params = light_params + dcn_overhead
    light_fd_params = light_params + fd_overhead
    light_both_params = light_params + dcn_overhead + fd_overhead
    
    print()
    print("=" * 80)
    print("MODEL TOTALS")
    print("=" * 80)
    print()
    
    results = [
        ('Regular U-Net', chans_regular, 'Standard', 'None', regular_params),
        ('Light U-Net', chans_light, 'Standard', 'None', light_params),
        ('Light + DCN', chans_light, 'Standard', 'Skip: DCNv2', light_dcn_params),
        ('Light + FD', chans_light, 'Standard', 'Neck: FDConv', light_fd_params),
        ('Light + Both (Ours)', chans_light, 'Standard', 'DCNv2 + FD', light_both_params),
    ]
    
    print(f"{'Model Variant':<25} {'Channels':<10} {'Convolution':<12} {'Added Modules':<15} {'Params':<12} {'Ratio':<8}")
    print("-" * 90)
    
    for name, ch, conv, modules, params in results:
        ratio = params / regular_params
        print(f"{name:<25} {ch:<10} {conv:<12} {modules:<15} {format_params(params):<12} {ratio:.2f}x")
    
    print()
    print("=" * 80)
    print("OVERHEAD ANALYSIS")
    print("=" * 80)
    print()
    print(f"Base Light U-Net: {format_params(light_params)}")
    print()
    print(f"Light + DCN:")
    print(f"  Total:    {format_params(light_dcn_params)}")
    print(f"  Overhead: +{format_params(dcn_overhead)} (+{100*dcn_overhead/light_params:.1f}%)")
    print()
    print(f"Light + FD:")
    print(f"  Total:    {format_params(light_fd_params)}")
    print(f"  Overhead: +{format_params(fd_overhead)} (+{100*fd_overhead/light_params:.1f}%)")
    print()
    print(f"Light + Both (Ours):")
    print(f"  Total:    {format_params(light_both_params)}")
    print(f"  Overhead: +{format_params(dcn_overhead + fd_overhead)} (+{100*(dcn_overhead + fd_overhead)/light_params:.1f}%)")
    
    print()
    print("=" * 80)
    print("FDConv KERNEL_NUM SENSITIVITY")
    print("=" * 80)
    print()
    print(f"{'kernel_num':<12} {'FDConv Bottleneck':<20} {'Overhead':<15} {'Total (Light+FD)':<20}")
    print("-" * 70)
    for kn in [2, 4, 8]:
        try:
            oh, std_b, fd_b = calculate_fdconv_overhead(chans=chans_light, num_pool_layers=num_pool_layers, kernel_num=kn)
            total = light_params + oh
            print(f"{kn:<12} {format_params(fd_b):<20} +{format_params(oh):<14} {format_params(total):<20}")
        except Exception as e:
            print(f"{kn:<12} Error: {e}")
    
    print()
    print("=" * 80)
    print("MARKDOWN TABLE (copy-paste ready)")
    print("=" * 80)
    print()
    print("| Model Variant | Channels | Convolution | Added Modules | Params | Ratio |")
    print("|---------------|----------|-------------|---------------|--------|-------|")
    for name, ch, conv, modules, params in results:
        ratio = params / regular_params
        print(f"| {name} | {ch} | {conv} | {modules} | ~{format_params(params)} | {ratio:.2f}x |")
    
    print()
    print("=" * 80)
    print("NOTES")
    print("=" * 80)
    print()
    print("Official FDConv (Chen et al. CVPR 2025):")
    print("  - Uses Fourier Disjoint Weights (FDW) for parameter efficiency")
    print("  - Includes KSM (Kernel Spatial Modulation) for dynamic adjustment")
    print("  - Includes FBM (Frequency Band Modulation) for frequency band control")
    print("  - Falls back to standard Conv2d if channels <= 16")
    print()


if __name__ == '__main__':
    main()
