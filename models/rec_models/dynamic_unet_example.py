"""
Example usage of Dynamic U-Net with growth during training.

This shows how to integrate the growth mechanism into your training loop.
"""

import torch
import torch.nn as nn
from dynamic_unet_model import DynamicUnetModel


def train_with_dynamic_growth():
    """
    Example training loop with periodic network growth.
    """
    # Initialize model
    model = DynamicUnetModel(
        in_chans=2,           # For complex MRI data
        out_chans=2,
        chans=32,             # Starting with 32 channels
        num_pool_layers=4,
        drop_prob=0.0,
        growth_method='sample_select',  # or 'split'
        n_candidates=10       # Number of candidates to try
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training parameters
    num_epochs = 100
    grow_every_n_epochs = 10  # Grow network every 10 epochs
    layers_to_grow = ['bottleneck', 'down_3', 'up_0']  # Which layers to grow
    
    print(f"Initial network size: {model.get_network_size()}")
    
    for epoch in range(num_epochs):
        # Regular training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Periodic growth
        if (epoch + 1) % grow_every_n_epochs == 0:
            print(f"\n=== Growing network at epoch {epoch + 1} ===")
            
            # Get a batch for gradient calculation
            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
            
            # Grow selected layers
            for layer_id in layers_to_grow:
                try:
                    model.grow_network(
                        layer_idx=layer_id,
                        conv_idx=1,  # Grow the second conv in the block
                        batch_data=data,
                        target_data=target,
                        criterion=criterion
                    )
                except Exception as e:
                    print(f"Failed to grow {layer_id}: {e}")
            
            # Reset optimizer to include new parameters
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            print(f"New network size: {model.get_network_size()}\n")


def train_with_gradient_splitting():
    """
    Example using gradient-informed splitting method.
    """
    model = DynamicUnetModel(
        in_chans=2,
        out_chans=2,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        growth_method='split'  # Use splitting method
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Grow every N steps
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"Growing network at epoch {epoch}, batch {batch_idx}")
                
                # Must have gradients for splitting method
                model.grow_network(
                    layer_idx='bottleneck',
                    conv_idx=1,
                    batch_data=data,
                    target_data=target,
                    criterion=criterion
                )
                
                # Reset optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def adaptive_growth_schedule():
    """
    More sophisticated: grow when loss plateaus.
    """
    model = DynamicUnetModel(
        in_chans=2, out_chans=2, chans=32,
        num_pool_layers=4, drop_prob=0.0,
        growth_method='sample_select', n_candidates=10
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Track loss for plateau detection
    loss_history = []
    plateau_threshold = 5  # epochs without improvement
    plateau_counter = 0
    best_loss = float('inf')
    
    for epoch in range(100):
        epoch_loss = 0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Check for plateau
        if avg_loss < best_loss * 0.99:  # 1% improvement threshold
            best_loss = avg_loss
            plateau_counter = 0
        else:
            plateau_counter += 1
        
        # Grow when plateau detected
        if plateau_counter >= plateau_threshold:
            print(f"\n=== Loss plateau detected at epoch {epoch} ===")
            print(f"Growing network... Current loss: {avg_loss:.6f}")
            
            # Get a batch
            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
            
            # Grow bottleneck (most impactful)
            model.grow_network(
                layer_idx='bottleneck',
                conv_idx=1,
                batch_data=data,
                target_data=target,
                criterion=criterion
            )
            
            # Reset
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            plateau_counter = 0
            best_loss = avg_loss
            
            print(f"Network size: {model.get_network_size()}\n")


# Integration with your existing exp.py
def integrate_with_existing_training(args):
    """
    How to integrate with your existing training code in exp.py
    """
    from models.rec_models.dynamic_unet_model import DynamicUnetModel
    
    # Replace regular UNet with Dynamic UNet
    model = DynamicUnetModel(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.num_chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        growth_method=args.growth_method if hasattr(args, 'growth_method') else 'sample_select',
        n_candidates=args.n_candidates if hasattr(args, 'n_candidates') else 10
    )
    
    # In your training loop, add growth logic
    # Example: grow every N epochs
    if epoch > 0 and epoch % args.growth_interval == 0:
        # Get a batch for gradient calculation
        sample_batch = next(iter(train_loader))
        
        model.grow_network(
            layer_idx='bottleneck',  # Or cycle through different layers
            conv_idx=1,
            batch_data=sample_batch[0].to(args.device),
            target_data=sample_batch[1].to(args.device),
            criterion=criterion
        )
        
        # Important: recreate optimizer to include new parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Log the growth
        print(f"Network grown at epoch {epoch}")
        print(f"New size: {model.get_network_size()}")


if __name__ == '__main__':
    print("Dynamic U-Net Example Usage")
    print("=" * 50)
    print("\nThis file shows how to use the Dynamic U-Net.")
    print("See the functions above for integration examples:")
    print("  - train_with_dynamic_growth()")
    print("  - train_with_gradient_splitting()")
    print("  - adaptive_growth_schedule()")
    print("  - integrate_with_existing_training()")

