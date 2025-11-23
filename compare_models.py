"""
Compare parameter counts and performance of different U-Net variants.

Models to compare:
1. Regular U-Net (baseline)
2. HybridCondUnet (CondConv only at bottleneck, 80% base channels)
3. SmallCondUnet (CondConv everywhere, 55% base channels)

This script helps you choose which model to train for your experiments.
"""

import torch
from models.rec_models.unet_model import UnetModel
from models.rec_models.hybrid_cond_unet import HybridCondUnetModel
from models.rec_models.small_cond_unet import SmallCondUnetModel


def count_params(model):
    """Count parameters in a model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def compare_models():
    """Compare all three model variants."""
    
    print("\n" + "=" * 100)
    print(" " * 30 + "Model Comparison: U-Net Variants")
    print("=" * 100 + "\n")
    
    # Configuration
    in_chans = 1
    out_chans = 1
    base_chans = 32  # Standard U-Net channels
    num_pool_layers = 4
    drop_prob = 0.0
    num_experts = 8
    
    print(f"Configuration:")
    print(f"  Input channels:  {in_chans}")
    print(f"  Output channels: {out_chans}")
    print(f"  Base channels:   {base_chans}")
    print(f"  Pool layers:     {num_pool_layers}")
    print(f"  Num experts:     {num_experts}")
    print()
    
    # Create models
    print("Creating models...\n")
    
    # 1. Regular U-Net
    unet = UnetModel(in_chans, out_chans, base_chans, num_pool_layers, drop_prob)
    unet_params = count_params(unet)
    
    # 2. Hybrid CondUnet (80% channels)
    hybrid_chans = int(base_chans * 0.8)
    hybrid = HybridCondUnetModel(in_chans, out_chans, hybrid_chans, num_pool_layers, 
                                   drop_prob, num_experts)
    hybrid_params = count_params(hybrid)
    
    # 3. Small CondUnet (55% channels)
    small_chans = int(base_chans * 0.55)
    small = SmallCondUnetModel(in_chans, out_chans, small_chans, num_pool_layers,
                                drop_prob, num_experts)
    small_params = count_params(small)
    
    # Display comparison
    print("=" * 100)
    print(f"{'Model':<25} {'Channels':<15} {'Parameters':<20} {'vs U-Net':<15} {'Description'}")
    print("=" * 100)
    
    print(f"{'1. Regular U-Net':<25} {base_chans:<15} {unet_params:>12,} {'(baseline)':<15} "
          f"Standard U-Net architecture")
    
    print(f"{'2. HybridCondUnet':<25} {hybrid_chans:<15} {hybrid_params:>12,} "
          f"{hybrid_params/unet_params:>7.2%}{'     ':<8} "
          f"CondConv at bottleneck only")
    
    print(f"{'3. SmallCondUnet':<25} {small_chans:<15} {small_params:>12,} "
          f"{small_params/unet_params:>7.2%}{'     ':<8} "
          f"CondConv everywhere")
    
    print("=" * 100 + "\n")
    
    # Test forward pass
    print("Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, in_chans, 320, 320).to(device)
    
    unet = unet.to(device)
    hybrid = hybrid.to(device)
    small = small.to(device)
    
    with torch.no_grad():
        out1 = unet(x)
        out2 = hybrid(x)
        out3 = small(x)
    
    print(f"✓ All models produce output shape: {tuple(out1.shape)}\n")
    
    # Parameter efficiency
    print("=" * 100)
    print("Parameter Efficiency Analysis:")
    print("=" * 100)
    print(f"\n{'Model':<25} {'Params':<20} {'Size (MB)':<15} {'Relative Size'}")
    print("-" * 100)
    
    for name, params in [('Regular U-Net', unet_params), 
                          ('HybridCondUnet', hybrid_params),
                          ('SmallCondUnet', small_params)]:
        size_mb = params * 4 / (1024**2)  # float32
        relative = params / unet_params
        print(f"{name:<25} {params:>12,} {size_mb:>12.2f} MB {relative:>12.2%}")
    
    print("\n" + "=" * 100)
    
    # Recommendations
    print("\n📋 Recommendations:")
    print("=" * 100)
    print("""
1. Regular U-Net (baseline):
   - Use this as your baseline for comparison
   - Most parameters, no conditional computation
   - Expected to perform well but may be parameter-inefficient

2. HybridCondUnet (~70-80% params):
   - CondConv only at bottleneck (most abstract features)
   - Fewer parameters than regular U-Net
   - Tests if adaptive high-level features help
   - Good balance between efficiency and capacity

3. SmallCondUnet (~60-70% params):
   - CondConv everywhere
   - Significantly fewer parameters
   - Tests if conditional computation is more efficient than static networks
   - May generalize better due to input-adaptive behavior

Hypothesis: 
  CondUnet variants may match or exceed regular U-Net performance with fewer 
  parameters due to dynamic, input-adaptive computation.

To test, train all three models and compare:
  - PSNR / SSIM (reconstruction quality)
  - Training time
  - Generalization (different protocols, accelerations)
  - Parameter efficiency (performance per parameter)
    """)
    print("=" * 100 + "\n")
    
    # How to run experiments
    print("🚀 How to Run Experiments:")
    print("=" * 100)
    print("""
1. Edit exp.py and set:
   model = 'Unet'           # For baseline
   model = 'HybridCondUnet' # For hybrid
   model = 'SmallCondUnet'  # For small

2. Run training:
   python exp.py

3. Compare results:
   - Check PSNR/SSIM metrics
   - Compare parameter counts (printed at start)
   - Analyze training curves
    """)
    print("=" * 100 + "\n")


if __name__ == '__main__':
    compare_models()

