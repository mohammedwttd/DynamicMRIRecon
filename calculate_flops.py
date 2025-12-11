"""
Calculate FLOPs (Floating Point Operations) and MACs (Multiply-Accumulate Operations) for all U-Net variants.

Usage:
    python3 calculate_flops.py

Note: Install required packages first:
    pip install ptflops
    # or
    pip install fvcore
"""

import torch
import sys
from models.rec_models.models.unet_model import UnetModel
from models.rec_models.models.dynamic_unet_model import DynamicUnetModel
from models.rec_models.models.cond_unet_model import CondUnetModel
from models.rec_models.models.hybrid_cond_unet import HybridCondUnetModel
from models.rec_models.models.small_cond_unet import SmallCondUnetModel
from models.rec_models.models.fd_unet_model import FDUnetModel
from models.rec_models.models.hybrid_snake_fd_unet import HybridSnakeFDUnet


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops_ptflops(model, input_shape=(1, 2, 320, 320)):
    """
    Calculate FLOPs using ptflops library.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
    
    Returns:
        macs: Multiply-Accumulate Operations (MACs)
        params: Number of parameters
    """
    try:
        from ptflops import get_model_complexity_info
        
        # ptflops expects (C, H, W) without batch dimension
        macs, params = get_model_complexity_info(
            model, 
            input_shape[1:],  # Remove batch dimension
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        return macs, params
    except ImportError:
        print("⚠️  ptflops not installed. Install with: pip install ptflops")
        return None, None
    except Exception as e:
        print(f"⚠️  ptflops failed: {e}")
        return None, None


def calculate_flops_fvcore(model, input_shape=(1, 2, 320, 320)):
    """
    Calculate FLOPs using fvcore library (Facebook's tool).
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
    
    Returns:
        flops: Floating Point Operations
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        
        model.eval()
        inputs = torch.randn(input_shape)
        flops = FlopCountAnalysis(model, inputs)
        total_flops = flops.total()
        return total_flops
    except ImportError:
        print("⚠️  fvcore not installed. Install with: pip install fvcore")
        return None
    except Exception as e:
        print(f"⚠️  fvcore failed: {e}")
        return None


def calculate_flops_thop(model, input_shape=(1, 2, 320, 320)):
    """
    Calculate FLOPs using thop library.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
    
    Returns:
        macs: Multiply-Accumulate Operations
        params: Number of parameters
    """
    try:
        from thop import profile, clever_format
        
        model.eval()
        inputs = torch.randn(input_shape)
        macs, params = profile(model, inputs=(inputs,), verbose=False)
        return macs, params
    except ImportError:
        print("⚠️  thop not installed. Install with: pip install thop")
        return None, None
    except Exception as e:
        print(f"⚠️  thop failed: {e}")
        return None, None


def format_number(num):
    """Format large numbers in human-readable format."""
    if num is None:
        return "N/A"
    
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def main():
    """Calculate FLOPs for all models."""
    
    # Standard configuration
    in_chans = 2
    out_chans = 2
    chans = 32
    num_pool_layers = 4
    drop_prob = 0.0
    input_shape = (1, 2, 320, 320)  # Batch=1, Channels=2, H=320, W=320
    
    models_config = {
        'Unet': {
            'model': UnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob),
            'description': 'Baseline U-Net'
        },
        'DynamicUnet': {
            'model': DynamicUnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob, swap_frequency=10),
            'description': 'Channel-swapping U-Net'
        },
        'CondUnet': {
            'model': CondUnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob, num_experts=8),
            'description': 'Full CondConv U-Net (8 experts)'
        },
        'HybridCondUnet': {
            'model': HybridCondUnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob, num_experts=8),
            'description': 'CondConv at bottleneck only'
        },
        'SmallCondUnet': {
            'model': SmallCondUnetModel(in_chans, out_chans, int(chans * 0.55), num_pool_layers, drop_prob, num_experts=8),
            'description': 'Full CondConv, fewer channels'
        },
        'FDUnet (kernel_num=4)': {
            'model': FDUnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob, kernel_num=4, use_simple=False),
            'description': 'Frequency Dynamic Conv (lightweight)'
        },
        'FDUnet (kernel_num=8)': {
            'model': FDUnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob, kernel_num=8, use_simple=False),
            'description': 'Frequency Dynamic Conv (moderate)'
        },
        'HybridSnakeFDUnet (2 layers)': {
            'model': HybridSnakeFDUnet(in_chans, out_chans, chans, num_pool_layers, drop_prob, 
                                       snake_layers=2, snake_kernel_size=9, fd_kernel_num=4, fd_use_simple=False),
            'description': 'Snake (encoder 1-2) + FDConv (rest)'
        },
        'HybridSnakeFDUnet (1 layer)': {
            'model': HybridSnakeFDUnet(in_chans, out_chans, chans, num_pool_layers, drop_prob, 
                                       snake_layers=1, snake_kernel_size=9, fd_kernel_num=4, fd_use_simple=False),
            'description': 'Snake (encoder 1) + FDConv (rest)'
        },
    }
    
    print("\n" + "=" * 120)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 120)
    print(f"Input shape: {input_shape} (Batch, Channels, Height, Width)")
    print(f"Resolution: {input_shape[2]}×{input_shape[3]} = {input_shape[2] * input_shape[3]:,} pixels")
    print("=" * 120)
    print()
    
    # Try different FLOP calculation methods
    methods = []
    try:
        import ptflops
        methods.append('ptflops')
    except ImportError:
        pass
    
    try:
        import fvcore
        methods.append('fvcore')
    except ImportError:
        pass
    
    try:
        import thop
        methods.append('thop')
    except ImportError:
        pass
    
    if not methods:
        print("⚠️  No FLOP calculation library found!")
        print("   Install one with:")
        print("   - pip install ptflops")
        print("   - pip install fvcore")
        print("   - pip install thop")
        print()
        print("Showing parameter counts only:\n")
    
    # Calculate for each model
    results = []
    
    for model_name, config in models_config.items():
        model = config['model']
        description = config['description']
        
        print(f"Analyzing: {model_name}")
        print(f"  Description: {description}")
        
        # Count parameters
        params = count_parameters(model)
        
        # Try to calculate FLOPs
        flops = None
        macs = None
        
        if 'thop' in methods:
            macs_thop, params_thop = calculate_flops_thop(model, input_shape)
            if macs_thop is not None:
                macs = macs_thop
                flops = macs_thop * 2  # MACs ≈ FLOPs/2 (multiply + add)
        
        if flops is None and 'fvcore' in methods:
            flops_fvcore = calculate_flops_fvcore(model, input_shape)
            if flops_fvcore is not None:
                flops = flops_fvcore
                macs = flops_fvcore / 2
        
        if flops is None and 'ptflops' in methods:
            macs_ptflops, params_ptflops = calculate_flops_ptflops(model, input_shape)
            if macs_ptflops is not None:
                macs = macs_ptflops
                flops = macs_ptflops * 2
        
        results.append({
            'name': model_name,
            'description': description,
            'params': params,
            'flops': flops,
            'macs': macs
        })
        
        print(f"  Parameters: {format_number(params)} ({params:,})")
        if flops is not None:
            print(f"  FLOPs: {format_number(flops)} ({flops:,})")
            print(f"  MACs: {format_number(macs)} ({macs:,})")
        print()
    
    # Print summary table
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    print(f"{'Model':<35} {'Description':<35} {'Parameters':<15} {'FLOPs':<15} {'MACs':<15}")
    print("-" * 120)
    
    baseline_params = results[0]['params']
    baseline_flops = results[0]['flops']
    
    for result in results:
        params_str = format_number(result['params'])
        params_ratio = f"({result['params'] / baseline_params:.2f}×)" if result['params'] != baseline_params else "(1.00×)"
        
        if result['flops'] is not None:
            flops_str = format_number(result['flops'])
            flops_ratio = f"({result['flops'] / baseline_flops:.2f}×)" if baseline_flops and result['flops'] != baseline_flops else "(1.00×)"
            macs_str = format_number(result['macs'])
        else:
            flops_str = "N/A"
            flops_ratio = ""
            macs_str = "N/A"
        
        print(f"{result['name']:<35} {result['description']:<35} {params_str:<8} {params_ratio:<6} {flops_str:<8} {flops_ratio:<6} {macs_str:<8}")
    
    print("=" * 120)
    print()
    
    # Additional notes
    print("NOTES:")
    print("  • FLOPs = Floating Point Operations (multiply OR add)")
    print("  • MACs = Multiply-Accumulate Operations (multiply AND add)")
    print("  • Relationship: FLOPs ≈ 2 × MACs")
    print("  • Lower is better for inference speed")
    print("  • These are per-forward-pass values")
    print()
    
    if not methods:
        print("\n💡 TIP: Install a FLOP calculation library for full analysis:")
        print("   pip install thop  # Recommended, most compatible")
        print()


if __name__ == '__main__':
    main()

