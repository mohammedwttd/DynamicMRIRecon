"""
Manual FLOPs calculation for models with custom operations.

FLOP calculation libraries (thop, fvcore) don't count:
- FFT/IFFT operations
- Grid sampling (deformable convolutions)
- Custom attention mechanisms

This script manually calculates FLOPs for all operations.
"""

import torch
import numpy as np
from models.rec_models.unet_model import UnetModel
from models.rec_models.fd_unet_model import FDUnetModel
from models.rec_models.hybrid_snake_fd_unet import HybridSnakeFDUnet


def count_conv2d_flops(in_channels, out_channels, kernel_size, input_h, input_w, groups=1):
    """
    Calculate FLOPs for a Conv2d operation.
    
    FLOPs = 2 × in_channels × out_channels × kernel_h × kernel_w × output_h × output_w / groups
    """
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    flops = 2 * (in_channels / groups) * out_channels * kernel_h * kernel_w * input_h * input_w
    return flops


def count_fft_flops(n):
    """
    Calculate FLOPs for FFT/IFFT.
    
    FFT complexity: O(N log N) where N is the total number of points
    For 2D FFT on H×W: 5 × H × W × log2(H × W)
    (Factor of 5 accounts for complex arithmetic)
    """
    if n <= 0:
        return 0
    flops = 5 * n * np.log2(n)
    return flops


def count_grid_sample_flops(batch, channels, height, width, num_points):
    """
    Calculate FLOPs for grid_sample (bilinear interpolation).
    
    Each sampling point requires:
    - 4 lookups (bilinear = 2×2 neighbors)
    - 8 multiplications + 4 additions per channel
    """
    flops_per_point = channels * (8 + 4)  # 8 muls, 4 adds
    total_flops = batch * num_points * flops_per_point
    return total_flops


def count_standard_unet_flops(in_chans=2, out_chans=2, chans=32, num_pool_layers=4):
    """
    Manually count FLOPs for standard U-Net.
    """
    total_flops = 0
    h, w = 320, 320
    
    # Encoder
    ch = chans
    for i in range(num_pool_layers):
        in_ch = in_chans if i == 0 else ch // 2
        
        # ConvBlock: 2 Conv2d(3×3)
        total_flops += count_conv2d_flops(in_ch, ch, 3, h, w)  # conv1
        total_flops += count_conv2d_flops(ch, ch, 3, h, w)     # conv2
        
        # Max pooling (negligible)
        h, w = h // 2, w // 2
        ch *= 2
    
    # Bottleneck
    ch = ch // 2  # Undo last doubling
    total_flops += count_conv2d_flops(ch, ch, 3, h, w)  # conv1
    total_flops += count_conv2d_flops(ch, ch, 3, h, w)  # conv2
    
    # Decoder
    for i in range(num_pool_layers):
        h, w = h * 2, w * 2
        
        # Upsampling (bilinear interpolation)
        total_flops += ch * h * w * 4  # 4 operations per pixel
        
        # ConvBlock: 2 Conv2d(3×3)
        total_flops += count_conv2d_flops(ch * 2, ch // 2, 3, h, w)  # conv1 (concat doubles input)
        total_flops += count_conv2d_flops(ch // 2, ch // 2, 3, h, w)  # conv2
        
        ch = ch // 2
    
    # Final 1×1 convolutions
    total_flops += count_conv2d_flops(ch, ch // 2, 1, h, w)
    total_flops += count_conv2d_flops(ch // 2, out_chans, 1, h, w)
    total_flops += count_conv2d_flops(out_chans, out_chans, 1, h, w)
    
    return total_flops


def count_fdconv_flops(in_channels, out_channels, kernel_size, height, width, kernel_num):
    """
    Manually count FLOPs for FDConv operation.
    
    FDConv involves:
    1. Kernel Spatial Modulation (attention): GAP + Conv1x1
    2. Frequency-domain kernel construction: Weighted sum of kernels
    3. IFFT to get spatial kernel
    4. Standard convolution with constructed kernel
    """
    flops = 0
    
    # 1. KSM: Global Average Pooling + Conv1x1 + Softmax
    flops += in_channels * height * width  # GAP
    flops += 2 * in_channels * kernel_num  # Conv1x1
    flops += kernel_num * 3  # Softmax (exp + sum + div)
    
    # 2. Weighted sum of frequency kernels
    # For each kernel: kernel_num × out_channels × in_channels × kh × kw × 2 (complex)
    kh = kw = kernel_size
    flops += 2 * kernel_num * out_channels * in_channels * kh * kw * 2  # Multiply + add, complex
    
    # 3. IFFT to convert frequency kernel to spatial kernel
    # For each output channel and input channel: IFFT on kh×kw
    num_iffts = out_channels * in_channels
    flops += num_iffts * count_fft_flops(kh * kw)
    
    # 4. Apply the spatial kernel (standard convolution)
    # This is the expensive part: per-batch convolution
    flops += count_conv2d_flops(in_channels, out_channels, kernel_size, height, width)
    
    # 5. FBM scaling
    flops += out_channels * height * width  # Element-wise multiply
    
    return flops


def count_snake_conv_flops(in_channels, out_channels, kernel_size, height, width):
    """
    Manually count FLOPs for Dynamic Snake Convolution.
    
    Snake Conv involves:
    1. Offset learning (2 branches for X and Y)
    2. Grid sampling along X and Y axes
    3. 1D convolution along each axis
    4. Fusion of 3 branches
    """
    flops = 0
    
    # 1. Offset learning for X-axis
    # Depthwise conv 3×3
    flops += count_conv2d_flops(in_channels, in_channels, 3, height, width, groups=in_channels)
    # Conv 1×1
    flops += count_conv2d_flops(in_channels, kernel_size, 1, height, width)
    
    # 2. Offset learning for Y-axis (same)
    flops += count_conv2d_flops(in_channels, in_channels, 3, height, width, groups=in_channels)
    flops += count_conv2d_flops(in_channels, kernel_size, 1, height, width)
    
    # 3. Grid sampling for X-axis
    # Sample kernel_size points per pixel
    num_samples = height * width * kernel_size
    flops += count_grid_sample_flops(1, in_channels, height, width, num_samples)
    
    # 4. Grid sampling for Y-axis
    flops += count_grid_sample_flops(1, in_channels, height, width, num_samples)
    
    # 5. 1D convolution along X-axis (kernel_size points)
    flops += 2 * in_channels * out_channels * kernel_size * height * width
    
    # 6. 1D convolution along Y-axis
    flops += 2 * in_channels * out_channels * kernel_size * height * width
    
    # 7. Standard convolution branch
    flops += count_conv2d_flops(in_channels, out_channels, 3, height, width)
    
    # 8. Fusion (3 branches -> 1)
    flops += count_conv2d_flops(out_channels * 3, out_channels, 1, height, width)
    
    return flops


def estimate_fdunet_flops(kernel_num=4):
    """
    Estimate total FLOPs for FDUnet.
    """
    total_flops = 0
    h, w = 320, 320
    chans = 32
    num_pool_layers = 4
    
    print(f"\nFDUnet (kernel_num={kernel_num}) FLOP Breakdown:")
    print("=" * 80)
    
    # Encoder
    ch = chans
    for i in range(num_pool_layers):
        in_ch = 2 if i == 0 else ch // 2
        
        # FDConvBlock: 2 FDConv layers
        conv1_flops = count_fdconv_flops(in_ch, ch, 3, h, w, kernel_num)
        conv2_flops = count_fdconv_flops(ch, ch, 3, h, w, kernel_num)
        total_flops += conv1_flops + conv2_flops
        
        print(f"Encoder {i+1} ({h}×{w}): {(conv1_flops + conv2_flops)/1e9:.2f} GFLOPs")
        
        h, w = h // 2, w // 2
        ch *= 2
    
    # Bottleneck
    ch = ch // 2
    bottleneck_flops = count_fdconv_flops(ch, ch, 3, h, w, kernel_num) * 2
    total_flops += bottleneck_flops
    print(f"Bottleneck ({h}×{w}): {bottleneck_flops/1e9:.2f} GFLOPs")
    
    # Decoder
    for i in range(num_pool_layers):
        h, w = h * 2, w * 2
        
        # Upsampling
        total_flops += ch * h * w * 4
        
        # FDConvBlock
        conv1_flops = count_fdconv_flops(ch * 2, ch // 2, 3, h, w, kernel_num)
        conv2_flops = count_fdconv_flops(ch // 2, ch // 2, 3, h, w, kernel_num)
        total_flops += conv1_flops + conv2_flops
        
        print(f"Decoder {i+1} ({h}×{w}): {(conv1_flops + conv2_flops)/1e9:.2f} GFLOPs")
        
        ch = ch // 2
    
    # Final convolutions
    final_flops = count_conv2d_flops(ch, ch // 2, 1, h, w)
    final_flops += count_conv2d_flops(ch // 2, 2, 1, h, w)
    final_flops += count_conv2d_flops(2, 2, 1, h, w)
    total_flops += final_flops
    print(f"Final layers: {final_flops/1e9:.2f} GFLOPs")
    
    print("=" * 80)
    print(f"Total: {total_flops/1e9:.2f} GFLOPs")
    
    return total_flops


def estimate_hybrid_snake_fd_flops(snake_layers=2, fd_kernel_num=4):
    """
    Estimate total FLOPs for HybridSnakeFDUnet.
    """
    total_flops = 0
    h, w = 320, 320
    chans = 32
    num_pool_layers = 4
    snake_kernel_size = 9
    
    print(f"\nHybridSnakeFDUnet (snake_layers={snake_layers}, fd_kernel_num={fd_kernel_num}) FLOP Breakdown:")
    print("=" * 80)
    
    # Encoder
    ch = chans
    for i in range(num_pool_layers):
        in_ch = 2 if i == 0 else ch // 2
        
        if i < snake_layers:
            # Snake Conv block
            conv1_flops = count_snake_conv_flops(in_ch, ch, snake_kernel_size, h, w)
            conv2_flops = count_conv2d_flops(ch, ch, 3, h, w)
            layer_type = "Snake"
        else:
            # Standard Conv block
            conv1_flops = count_conv2d_flops(in_ch, ch, 3, h, w)
            conv2_flops = count_conv2d_flops(ch, ch, 3, h, w)
            layer_type = "Standard"
        
        total_flops += conv1_flops + conv2_flops
        print(f"Encoder {i+1} ({h}×{w}, {layer_type}): {(conv1_flops + conv2_flops)/1e9:.2f} GFLOPs")
        
        h, w = h // 2, w // 2
        ch *= 2
    
    # Bottleneck (FDConv)
    ch = ch // 2
    bottleneck_flops = count_fdconv_flops(ch, ch, 3, h, w, fd_kernel_num) * 2
    total_flops += bottleneck_flops
    print(f"Bottleneck ({h}×{w}, FDConv): {bottleneck_flops/1e9:.2f} GFLOPs")
    
    # Decoder (FDConv)
    for i in range(num_pool_layers):
        h, w = h * 2, w * 2
        
        # Upsampling
        total_flops += ch * h * w * 4
        
        # FDConvBlock
        conv1_flops = count_fdconv_flops(ch * 2, ch // 2, 3, h, w, fd_kernel_num)
        conv2_flops = count_fdconv_flops(ch // 2, ch // 2, 3, h, w, fd_kernel_num)
        total_flops += conv1_flops + conv2_flops
        
        print(f"Decoder {i+1} ({h}×{w}, FDConv): {(conv1_flops + conv2_flops)/1e9:.2f} GFLOPs")
        
        ch = ch // 2
    
    # Final convolutions
    final_flops = count_conv2d_flops(ch, ch // 2, 1, h, w)
    final_flops += count_conv2d_flops(ch // 2, 2, 1, h, w)
    final_flops += count_conv2d_flops(2, 2, 1, h, w)
    total_flops += final_flops
    print(f"Final layers: {final_flops/1e9:.2f} GFLOPs")
    
    print("=" * 80)
    print(f"Total: {total_flops/1e9:.2f} GFLOPs")
    
    return total_flops


def main():
    """Compare FLOPs across all models."""
    
    print("\n" + "=" * 80)
    print("MANUAL FLOP CALCULATION (Including Custom Operations)")
    print("=" * 80)
    print("Standard FLOP libraries (thop, fvcore) DON'T count:")
    print("  • FFT/IFFT operations (used in FDConv)")
    print("  • Grid sampling (used in Snake Conv)")
    print("  • Custom attention mechanisms")
    print("\nThis script manually calculates ALL operations.")
    print("=" * 80)
    
    # Baseline U-Net
    unet_flops = count_standard_unet_flops()
    print(f"\nStandard U-Net: {unet_flops/1e9:.2f} GFLOPs")
    
    # FDUnet with different kernel_num
    fdunet_4_flops = estimate_fdunet_flops(kernel_num=4)
    fdunet_8_flops = estimate_fdunet_flops(kernel_num=8)
    
    # HybridSnakeFDUnet
    hybrid_flops = estimate_hybrid_snake_fd_flops(snake_layers=2, fd_kernel_num=4)
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<40} {'FLOPs':<15} {'Relative to U-Net':<20}")
    print("-" * 80)
    print(f"{'U-Net (baseline)':<40} {unet_flops/1e9:>7.2f} GFLOPs   {'1.00×':<20}")
    print(f"{'FDUnet (kernel_num=4)':<40} {fdunet_4_flops/1e9:>7.2f} GFLOPs   {fdunet_4_flops/unet_flops:>4.2f}×")
    print(f"{'FDUnet (kernel_num=8)':<40} {fdunet_8_flops/1e9:>7.2f} GFLOPs   {fdunet_8_flops/unet_flops:>4.2f}×")
    print(f"{'HybridSnakeFDUnet (snake=2, fd=4)':<40} {hybrid_flops/1e9:>7.2f} GFLOPs   {hybrid_flops/unet_flops:>4.2f}×")
    print("=" * 80)
    
    print("\nNOTE:")
    print("  • These are THEORETICAL estimates based on operation counts")
    print("  • Actual runtime depends on hardware optimization (CUDA kernels, etc.)")
    print("  • FFT/IFFT are highly optimized on GPUs (cuFFT)")
    print("  • Grid sampling benefits from hardware interpolation units")
    print()


if __name__ == '__main__':
    main()

