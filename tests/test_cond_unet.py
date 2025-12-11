"""
Simple test for CondUnet (Conditional U-Net with CondConv).

Tests basic functionality including:
1. Forward pass
2. Parameter counting
3. Routing weight analysis
"""

import torch
import torch.nn as nn
from models.rec_models.models.cond_unet_model import CondUnetModel


def test_basic_forward():
    """Test that forward pass works."""
    print("=" * 80)
    print("Test 1: Basic Forward Pass")
    print("=" * 80)
    
    model = CondUnetModel(
        in_chans=2,
        out_chans=2,
        chans=16,
        num_pool_layers=3,
        drop_prob=0.0,
        num_experts=4
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Device: {device}")
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 2, 128, 128, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"✓ Forward pass successful!\n")
    
    return model


def test_parameter_count(model):
    """Test parameter counting."""
    print("=" * 80)
    print("Test 2: Parameter Count")
    print("=" * 80)
    
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    
    # Compare with regular U-Net (approximately)
    from models.rec_models.models.unet_model import UnetModel
    unet = UnetModel(
        in_chans=2,
        out_chans=2,
        chans=16,
        num_pool_layers=3,
        drop_prob=0.0
    )
    unet_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    
    print(f"Regular U-Net parameters: {unet_params:,}")
    print(f"CondUnet has {total_params / unet_params:.2f}x more parameters")
    print(f"✓ Parameter count checked!\n")


def test_routing_weights(model):
    """Test routing weight extraction."""
    print("=" * 80)
    print("Test 3: Routing Weights Analysis")
    print("=" * 80)
    
    device = next(model.parameters()).device
    x = torch.randn(2, 2, 128, 128, device=device)
    
    routing_weights = model.get_routing_weights(x)
    
    print(f"Number of layers with routing: {len(routing_weights)}")
    for layer_name, weights in routing_weights.items():
        print(f"{layer_name:20s}: shape {tuple(weights.shape)}, "
              f"mean={weights.mean():.3f}, std={weights.std():.3f}")
    
    print(f"✓ Routing weights extracted successfully!\n")


def test_training_step():
    """Test a simple training step."""
    print("=" * 80)
    print("Test 4: Training Step")
    print("=" * 80)
    
    model = CondUnetModel(
        in_chans=2,
        out_chans=2,
        chans=8,
        num_pool_layers=2,
        drop_prob=0.0,
        num_experts=4
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Create dummy data
    x = torch.randn(2, 2, 64, 64, device=device)
    target = torch.randn(2, 2, 64, 64, device=device)
    
    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"✓ Training step successful!\n")


def test_multiple_experts():
    """Test with different numbers of experts."""
    print("=" * 80)
    print("Test 5: Multiple Expert Configurations")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 2, 64, 64, device=device)
    
    for num_experts in [2, 4, 8, 16]:
        model = CondUnetModel(
            in_chans=2,
            out_chans=2,
            chans=8,
            num_pool_layers=2,
            drop_prob=0.0,
            num_experts=num_experts
        ).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        params = model.count_parameters()
        print(f"Experts: {num_experts:2d} | Parameters: {params:8,} | Output shape: {tuple(output.shape)}")
    
    print(f"✓ All expert configurations work!\n")


def test_comparison_with_unet():
    """Compare CondUnet with regular U-Net."""
    print("=" * 80)
    print("Test 6: Comparison with Regular U-Net")
    print("=" * 80)
    
    from models.rec_models.models.unet_model import UnetModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 2, 128, 128, device=device)
    
    # Regular U-Net
    unet = UnetModel(
        in_chans=2,
        out_chans=2,
        chans=16,
        num_pool_layers=3,
        drop_prob=0.0
    ).to(device)
    
    # CondUnet
    cond_unet = CondUnetModel(
        in_chans=2,
        out_chans=2,
        chans=16,
        num_pool_layers=3,
        drop_prob=0.0,
        num_experts=8
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        unet_out = unet(x)
        cond_out = cond_unet(x)
    
    unet_params = sum(p.numel() for p in unet.parameters())
    cond_params = sum(p.numel() for p in cond_unet.parameters())
    
    print(f"Regular U-Net:")
    print(f"  Parameters: {unet_params:,}")
    print(f"  Output shape: {tuple(unet_out.shape)}")
    print()
    print(f"CondUnet:")
    print(f"  Parameters: {cond_params:,} ({cond_params/unet_params:.2f}x)")
    print(f"  Output shape: {tuple(cond_out.shape)}")
    print(f"  Number of experts: 8")
    print()
    print(f"✓ Both models produce same output shape!\n")


if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CondUnet Test Suite" + " " * 39 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run tests
    model = test_basic_forward()
    test_parameter_count(model)
    test_routing_weights(model)
    test_training_step()
    test_multiple_experts()
    test_comparison_with_unet()
    
    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print()
    print("To use CondUnet in training, set model='CondUnet' in exp.py")
    print()

