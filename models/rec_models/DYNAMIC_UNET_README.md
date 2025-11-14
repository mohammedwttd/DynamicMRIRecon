# Dynamic U-Net for MRI Reconstruction

A U-Net implementation that can **dynamically grow during training** using GradMax-inspired methods.

## Overview

Traditional neural networks have a fixed architecture throughout training. Dynamic U-Net allows you to:
- Start with a smaller network
- Progressively add capacity where it's needed most
- Choose filters based on gradient information
- Potentially achieve better performance with efficient growth

## Growth Methods

### 1. Sample-and-Select (Recommended for beginners)

**How it works:**
1. Generate N random candidate filters
2. Add each temporarily (with "silent" addition - zero weights in next layer)
3. Run a forward-backward pass
4. Measure gradient norm for each candidate
5. Keep the candidate with highest gradient norm

**Advantages:**
- Simple to understand and implement
- Adds diversity (explores random initializations)
- Works well in practice

**Usage:**
```python
model = DynamicUnetModel(
    in_chans=2, out_chans=2, chans=32,
    num_pool_layers=4, drop_prob=0.0,
    growth_method='sample_select',
    n_candidates=10  # Try 10 random candidates
)
```

### 2. Gradient-Informed Splitting

**How it works:**
1. Find the filter with the largest gradient norm
2. Duplicate it
3. Add small noise to break symmetry
4. Halve the weights in the next layer to maintain output

**Advantages:**
- More principled (builds on what's already working)
- Faster (no need to test N candidates)
- Maintains network output initially

**Usage:**
```python
model = DynamicUnetModel(
    in_chans=2, out_chans=2, chans=32,
    num_pool_layers=4, drop_prob=0.0,
    growth_method='split'
)
```

## Integration with Training

### Basic Example

```python
from models.rec_models.dynamic_unet_model import DynamicUnetModel

# Initialize
model = DynamicUnetModel(
    in_chans=2, out_chans=2, chans=32,
    num_pool_layers=4, drop_prob=0.0,
    growth_method='sample_select',
    n_candidates=10
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    # Regular training
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Grow every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Get a sample batch
        data, target = next(iter(train_loader))
        
        # Grow the bottleneck layer
        model.grow_network(
            layer_idx='bottleneck',
            conv_idx=1,  # Second conv in the block
            batch_data=data,
            target_data=target,
            criterion=criterion
        )
        
        # IMPORTANT: Recreate optimizer to include new parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        print(f"Network size: {model.get_network_size()}")
```

### Growth Schedules

#### 1. Fixed Interval Growth
Grow every N epochs:
```python
if (epoch + 1) % growth_interval == 0:
    model.grow_network(...)
```

#### 2. Adaptive Growth (Loss Plateau)
Grow when training stalls:
```python
if loss hasn't improved for N epochs:
    model.grow_network(...)
```

#### 3. Progressive Growth
Grow different layers at different times:
```python
epoch_schedule = {
    10: 'bottleneck',
    20: 'down_3',
    30: 'up_0',
    40: 'bottleneck',
    # ...
}

if epoch in epoch_schedule:
    layer_to_grow = epoch_schedule[epoch]
    model.grow_network(layer_idx=layer_to_grow, ...)
```

## Layer Identifiers

When calling `grow_network()`, specify which layer to grow:

- `'bottleneck'` - The middle layer (most impactful)
- `'down_0'`, `'down_1'`, `'down_2'`, `'down_3'` - Encoder layers
- `'up_0'`, `'up_1'`, `'up_2'`, `'up_3'` - Decoder layers

Each layer has two convolutions (`conv_idx=0` or `conv_idx=1`).

## Monitoring Growth

Track network size over time:
```python
sizes = model.get_network_size()
print(sizes)
# Output:
# {
#   'down_0': {'conv1_out': 32, 'conv2_out': 32},
#   'down_1': {'conv1_out': 64, 'conv2_out': 64},
#   ...
#   'bottleneck': {'conv1_out': 256, 'conv2_out': 257},  # <- Grown!
#   ...
# }
```

## Best Practices

1. **Start Small**: Begin with fewer channels (e.g., 16 or 32) to see clear growth
2. **Grow Strategically**: Bottleneck first, then deeper layers
3. **Reset Optimizer**: Always recreate the optimizer after growth
4. **Don't Grow Too Often**: Every 5-10 epochs is reasonable
5. **Monitor Performance**: Log network size and validation metrics

## Command-Line Arguments

Add these to your `exp.py`:

```python
parser.add_argument('--model-type', type=str, default='unet',
                    choices=['unet', 'dynamic_unet'])
parser.add_argument('--growth-method', type=str, default='sample_select',
                    choices=['sample_select', 'split'])
parser.add_argument('--n-candidates', type=int, default=10,
                    help='Number of candidates for sample_select')
parser.add_argument('--growth-interval', type=int, default=10,
                    help='Grow network every N epochs')
parser.add_argument('--layers-to-grow', type=str, nargs='+',
                    default=['bottleneck'],
                    help='Which layers to grow')
```

Then use:
```bash
python exp.py --model-type dynamic_unet --growth-method sample_select \
              --growth-interval 10 --layers-to-grow bottleneck down_3
```

## Debugging Tips

1. **"No gradients available"**: Make sure to run forward-backward pass before growing with 'split' method
2. **CUDA out of memory**: Network grows, so adjust batch size or growth frequency
3. **Loss spikes after growth**: Expected initially; should stabilize within a few iterations
4. **Growth doesn't help**: Try different layers, or network may already be large enough

## Mathematical Background

### Sample-and-Select
Approximates:
$$\arg\max_{w \in \mathcal{W}} \|\nabla_w L(W \cup \{w\})\|$$

by sampling $N$ candidates and selecting best.

### Gradient-Informed Splitting
Finds filter $i$ with max gradient:
$$i^* = \arg\max_i \|\nabla_{W_i} L\|$$

Then duplicates with noise: $W_{new} = W_{i^*} + \epsilon$

## References

- **GradMax**: Evci et al., "Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win"
- **Firefly Algorithm**: Progressive neural networks with dynamic growth

## Troubleshooting

**Q: Network doesn't grow / error during growth**
- Check that layer_idx and conv_idx are valid
- Ensure batch_data and target_data are on correct device
- Verify gradients exist (for 'split' method)

**Q: Performance worse than fixed U-Net**
- Try different growth schedules
- Start with even smaller initial network
- Grow less frequently
- Use 'sample_select' with more candidates (e.g., 20)

**Q: How to resume training with grown network?**
- Save/load model state dict normally
- Network size is automatically preserved
- No special handling needed

## Example Output

```
Initial network size: {'bottleneck': {'conv1_out': 256, 'conv2_out': 256}}
Epoch 10: Loss 0.0123
=== Growing network at epoch 10 ===
Growing bottleneck conv1: Selected candidate 7
New network size: {'bottleneck': {'conv1_out': 257, 'conv2_out': 256}}
Epoch 20: Loss 0.0098
=== Growing network at epoch 20 ===
Growing bottleneck conv1: Selected candidate 3
New network size: {'bottleneck': {'conv1_out': 258, 'conv2_out': 256}}
...
```

## Future Enhancements

- [ ] Automatic layer selection (grow layer with highest gradient)
- [ ] Pruning (remove low-magnitude filters)
- [ ] Multi-filter growth (add K filters at once)
- [ ] Layer-wise learning rates for new filters
- [ ] Better next-layer handling for deeper architectures

