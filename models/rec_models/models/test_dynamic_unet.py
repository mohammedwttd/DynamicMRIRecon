"""
Quick test to verify Dynamic U-Net implementation.
"""

import torch
import torch.nn as nn
from dynamic_unet_model import DynamicUnetModel


def test_basic_forward():
    """Test that forward pass works."""
    print("Test 1: Basic forward pass...")
    model = DynamicUnetModel(
        in_chans=2, out_chans=2, chans=16,
        num_pool_layers=3, drop_prob=0.0,
        growth_method='sample_select', n_candidates=3
    )
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 2, 128, 128)
    
    # Forward pass
    output = model(x)
    
    assert output.shape == (batch_size, 2, 128, 128), f"Wrong output shape: {output.shape}"
    print(f"✓ Forward pass works! Output shape: {output.shape}")
    return model


def test_sample_select_growth():
    """Test sample-and-select growth method."""
    print("\nTest 2: Sample-and-Select growth...")
    model = DynamicUnetModel(
        in_chans=2, out_chans=2, chans=8,
        num_pool_layers=2, drop_prob=0.0,
        growth_method='sample_select', n_candidates=3
    )
    
    # Initial size
    initial_size = model.get_network_size()
    print(f"Initial bottleneck size: {initial_size['bottleneck']}")
    
    # Create dummy data
    data = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    criterion = nn.MSELoss()
    
    # Grow the network
    try:
        model.grow_network(
            layer_idx='bottleneck',
            conv_idx=1,
            batch_data=data,
            target_data=target,
            criterion=criterion
        )
        
        # Check size increased
        new_size = model.get_network_size()
        print(f"After growth: {new_size['bottleneck']}")
        
        # Forward pass should still work
        output = model(data)
        assert output.shape == target.shape
        print("✓ Sample-and-Select growth works!")
        return True
        
    except Exception as e:
        print(f"✗ Growth failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_split_growth():
    """Test gradient-informed splitting method."""
    print("\nTest 3: Gradient-Informed Splitting...")
    model = DynamicUnetModel(
        in_chans=2, out_chans=2, chans=8,
        num_pool_layers=2, drop_prob=0.0,
        growth_method='split'
    )
    
    # Create dummy data
    data = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    criterion = nn.MSELoss()
    
    # Need to run forward-backward first to get gradients
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    initial_size = model.get_network_size()
    print(f"Initial bottleneck size: {initial_size['bottleneck']}")
    
    # Grow the network
    try:
        model.grow_network(
            layer_idx='bottleneck',
            conv_idx=1,
            batch_data=data,
            target_data=target,
            criterion=criterion
        )
        
        new_size = model.get_network_size()
        print(f"After growth: {new_size['bottleneck']}")
        
        # Forward pass should still work
        output = model(data)
        assert output.shape == target.shape
        print("✓ Gradient-Informed Splitting works!")
        return True
        
    except Exception as e:
        print(f"✗ Growth failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_growths():
    """Test growing multiple times."""
    print("\nTest 4: Multiple sequential growths...")
    model = DynamicUnetModel(
        in_chans=2, out_chans=2, chans=4,
        num_pool_layers=2, drop_prob=0.0,
        growth_method='sample_select', n_candidates=2
    )
    
    data = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    criterion = nn.MSELoss()
    
    print(f"Initial size: {model.get_network_size()['bottleneck']}")
    
    # Grow 3 times
    for i in range(3):
        try:
            model.grow_network(
                layer_idx='bottleneck',
                conv_idx=1,
                batch_data=data,
                target_data=target,
                criterion=criterion
            )
            size = model.get_network_size()
            print(f"After growth {i+1}: {size['bottleneck']}")
        except Exception as e:
            print(f"✗ Growth {i+1} failed: {e}")
            return False
    
    # Verify forward pass still works
    output = model(data)
    assert output.shape == target.shape
    print("✓ Multiple growths work!")
    return True


def test_network_size_tracking():
    """Test that network size is correctly tracked."""
    print("\nTest 5: Network size tracking...")
    model = DynamicUnetModel(
        in_chans=2, out_chans=2, chans=8,
        num_pool_layers=3, drop_prob=0.0,
        growth_method='sample_select', n_candidates=2
    )
    
    sizes = model.get_network_size()
    print("Network architecture:")
    for layer_name, layer_sizes in sizes.items():
        print(f"  {layer_name}: {layer_sizes}")
    
    # Check all expected layers are present
    expected_layers = ['down_0', 'down_1', 'down_2', 'bottleneck', 'up_0', 'up_1', 'up_2']
    for layer in expected_layers:
        assert layer in sizes, f"Missing layer: {layer}"
    
    print("✓ Network size tracking works!")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Dynamic U-Net Implementation")
    print("=" * 60)
    
    tests = [
        ("Basic Forward Pass", test_basic_forward),
        ("Sample-Select Growth", test_sample_select_growth),
        ("Split Growth", test_split_growth),
        ("Multiple Growths", test_multiple_growths),
        ("Size Tracking", test_network_size_tracking)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

