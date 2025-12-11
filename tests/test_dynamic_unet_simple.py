"""
Simple sanity test for the Dynamic U-Net with channel swapping.

This script demonstrates:
1. Creating a dynamic U-Net
2. Running forward passes
3. Automatic channel swapping every 10 batches
4. Tracking parameter count (should remain approximately constant)
"""

import torch
import torch.nn as nn
from models.rec_models.models.dynamic_unet_model import DynamicUnetModel


def test_dynamic_unet():
    """Test the dynamic U-Net with channel swapping."""
    
    print("=" * 80)
    print("Dynamic U-Net Sanity Test - Simplified Channel Swapping")
    print("=" * 80)
    
    # Configuration
    in_chans = 2
    out_chans = 2
    chans = 16  # Start with small network for easy testing
    num_pool_layers = 3
    drop_prob = 0.0
    swap_frequency = 10  # Swap every 10 batches
    
    # Create model
    print(f"\nCreating Dynamic U-Net:")
    print(f"  - Input channels: {in_chans}")
    print(f"  - Output channels: {out_chans}")
    print(f"  - Starting channels: {chans}")
    print(f"  - Depth: {num_pool_layers}")
    print(f"  - Swap frequency: every {swap_frequency} batches")
    
    model = DynamicUnetModel(
        in_chans=in_chans,
        out_chans=out_chans,
        chans=chans,
        num_pool_layers=num_pool_layers,
        drop_prob=drop_prob,
        swap_frequency=swap_frequency
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  - Device: {device}")
    
    # Initial network size
    print("\n" + "=" * 80)
    print("Initial Network Architecture:")
    print("=" * 80)
    sizes = model.get_network_size()
    for layer_name, layer_sizes in sizes.items():
        print(f"{layer_name:15s}: conv1_out={layer_sizes['conv1_out']:3d}, "
              f"conv2_out={layer_sizes['conv2_out']:3d}")
    
    initial_params = model.count_parameters()
    print(f"\nInitial parameter count: {initial_params:,}")
    
    # Create dummy data
    batch_size = 4
    height, width = 128, 128
    
    # Loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Simulate training for multiple batches
    num_batches = 35  # Will trigger 3 swaps (at batches 10, 20, 30)
    
    print("\n" + "=" * 80)
    print(f"Simulating {num_batches} training batches")
    print("=" * 80)
    
    param_history = [initial_params]
    
    for batch_idx in range(1, num_batches + 1):
        # Create random input and target
        input_data = torch.randn(batch_size, in_chans, height, width, device=device)
        target_data = torch.randn(batch_size, out_chans, height, width, device=device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track parameters before swap
        params_before = model.count_parameters()
        
        # Check if swap should happen (and perform it)
        model.maybe_swap_channels()
        
        # Track parameters after swap
        params_after = model.count_parameters()
        param_history.append(params_after)
        
        # Print batch info
        if batch_idx % swap_frequency == 0:
            print(f"\nBatch {batch_idx:3d}: Loss = {loss.item():.6f}")
            print(f"  Parameters before swap: {params_before:,}")
            print(f"  Parameters after swap:  {params_after:,}")
            print(f"  Change: {params_after - params_before:+,}")
        elif batch_idx % 5 == 0:
            print(f"Batch {batch_idx:3d}: Loss = {loss.item():.6f}, Params = {params_after:,}")
    
    # Final network size
    print("\n" + "=" * 80)
    print("Final Network Architecture:")
    print("=" * 80)
    sizes = model.get_network_size()
    for layer_name, layer_sizes in sizes.items():
        print(f"{layer_name:15s}: conv1_out={layer_sizes['conv1_out']:3d}, "
              f"conv2_out={layer_sizes['conv2_out']:3d}")
    
    final_params = model.count_parameters()
    print(f"\nFinal parameter count: {final_params:,}")
    print(f"Change from initial: {final_params - initial_params:+,} "
          f"({100 * (final_params - initial_params) / initial_params:+.2f}%)")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Parameter Count History:")
    print("=" * 80)
    print(f"  Initial:  {param_history[0]:,}")
    print(f"  Min:      {min(param_history):,}")
    print(f"  Max:      {max(param_history):,}")
    print(f"  Final:    {param_history[-1]:,}")
    print(f"  Std Dev:  {torch.tensor(param_history).float().std().item():,.1f}")
    
    # Check that parameters stayed relatively constant
    param_change_pct = 100 * (final_params - initial_params) / initial_params
    if abs(param_change_pct) < 5:
        print(f"\n✅ SUCCESS: Parameter count remained approximately constant!")
        print(f"   (Changed by only {param_change_pct:.2f}%)")
    else:
        print(f"\n⚠️  WARNING: Parameter count changed significantly by {param_change_pct:.2f}%")
    
    # Test forward pass still works
    print("\n" + "=" * 80)
    print("Final Forward Pass Test:")
    print("=" * 80)
    with torch.no_grad():
        test_input = torch.randn(2, in_chans, height, width, device=device)
        test_output = model(test_input)
        print(f"  Input shape:  {tuple(test_input.shape)}")
        print(f"  Output shape: {tuple(test_output.shape)}")
        print(f"  Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
        print(f"\n✅ Model still works after {num_batches} batches and multiple channel swaps!")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    test_dynamic_unet()

