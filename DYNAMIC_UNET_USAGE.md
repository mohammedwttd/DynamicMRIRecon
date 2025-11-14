# Using Dynamic U-Net with Your Training Pipeline

The Dynamic U-Net is now fully integrated! You can use it just like the regular U-Net.

## Quick Start

### Option 1: Using exp.py (Recommended)

Simply change the model in `exp.py`:

```python
model = 'DynamicUnet'  # Changed from 'Unet'

# Optional: Customize growth parameters
growth_method = 'sample_select'  # or 'split'
n_candidates = 10
growth_interval = 10  # Grow every 10 epochs
layers_to_grow = ['bottleneck']  # or ['bottleneck', 'down_3', 'up_0']
```

Then run:
```bash
python exp.py
```

### Option 2: Direct train.py

```bash
python train.py \
  --model=DynamicUnet \
  --growth-method=sample_select \
  --n-candidates=10 \
  --growth-interval=10 \
  --layers-to-grow bottleneck \
  --num-chans=16 \
  --num-epochs=30 \
  # ... other parameters as usual
```

## Configuration Parameters

### Required
- `--model=DynamicUnet` - Use Dynamic U-Net instead of regular U-Net

### Optional (Dynamic U-Net specific)
- `--growth-method` - `sample_select` (default) or `split`
  - `sample_select`: Tests N random candidates, picks best by gradient
  - `split`: Duplicates high-gradient filters with noise
  
- `--n-candidates` - Number of candidate filters to try (default: 10)
  - Only used for `sample_select` method
  - Higher = more thorough but slower growth
  
- `--growth-interval` - Grow every N epochs (default: 10)
  - Set to 0 to disable growth
  - Smaller = more frequent growth
  
- `--layers-to-grow` - Which layers to grow (default: `['bottleneck']`)
  - Options: `bottleneck`, `down_0`, `down_1`, `down_2`, `down_3`, `up_0`, `up_1`, `up_2`, `up_3`
  - Example: `--layers-to-grow bottleneck down_3 up_0`

## Examples

### Example 1: Basic Dynamic U-Net (start small, grow gradually)
```python
# In exp.py:
model = 'DynamicUnet'
drop_prob = 0.0
# Keep num_chans default (32) or reduce to start smaller

growth_method = 'sample_select'
n_candidates = 10
growth_interval = 10
layers_to_grow = ['bottleneck']

num_epochs = 50
```

### Example 2: Aggressive Growth (multiple layers)
```python
# In exp.py:
model = 'DynamicUnet'

growth_method = 'sample_select'
n_candidates = 15  # More candidates
growth_interval = 5  # Grow more frequently
layers_to_grow = ['bottleneck', 'down_3', 'down_2', 'up_0']  # Multiple layers

num_epochs = 50
```

### Example 3: Fast Growth with Splitting
```python
# In exp.py:
model = 'DynamicUnet'

growth_method = 'split'  # Faster method
n_candidates = 10  # Not used for split, but keep for compatibility
growth_interval = 5
layers_to_grow = ['bottleneck', 'down_3']

num_epochs = 40
```

## What to Expect

### During Training

You'll see output like:
```
Epoch = [   9/ 30] TrainLoss = 0.0123 DevLoss = 0.0145
============================================================
Growing network at epoch 10
============================================================
  ✓ Grew layer: bottleneck

New network size:
  bottleneck: {'conv1_out': 32, 'conv2_out': 33}
============================================================
Optimizer reset after network growth at epoch 9
Epoch = [  10/ 30] TrainLoss = 0.0119 DevLoss = 0.0142
```

### Performance Characteristics

- **Memory**: Grows gradually as network expands
- **Speed**: Small overhead during growth epochs (~5-30s depending on n_candidates)
- **Checkpoints**: Automatically save grown architecture
- **Resuming**: Works normally, network size preserved in checkpoint

## Tips for Best Results

1. **Start Smaller**: Use fewer initial channels (e.g., `num_chans=16` instead of 32)
   - This allows more room for growth to make a difference
   
2. **Bottleneck First**: Start by growing just the bottleneck
   - It's the most impactful layer
   - See results before adding more layers
   
3. **Monitor Growth**: Check the network size output
   - Make sure filters are actually being added
   - Verify growth is happening when expected
   
4. **Adjust Frequency**: 
   - Too frequent (every 2-3 epochs): May not converge well
   - Too rare (every 30 epochs): May miss opportunities
   - Sweet spot: Every 5-15 epochs
   
5. **Compare Baselines**: Run both regular U-Net and Dynamic U-Net
   - Same total epochs
   - Compare final PSNR/SSIM
   - Check parameter count

## Troubleshooting

### Growth isn't happening
- Check that `model = 'DynamicUnet'` (exact spelling)
- Verify `growth_interval` is reasonable (< num_epochs)
- Look for growth messages in log output

### Out of memory errors
- Reduce `n_candidates` (try 5 instead of 10)
- Increase `growth_interval` (grow less frequently)
- Reduce number of `layers_to_grow`
- Start with even fewer initial `num_chans`

### Loss spikes after growth
- Normal! Network needs a few iterations to adapt
- Usually recovers within 1-2 epochs
- If doesn't recover, try growing less frequently

### Performance worse than regular U-Net
- You may be starting too large - try `num_chans=16`
- Try growing less frequently
- Experiment with `split` method instead of `sample_select`

## Comparison: Dynamic U-Net vs Regular U-Net

| Aspect | Regular U-Net | Dynamic U-Net |
|--------|--------------|---------------|
| Architecture | Fixed | Grows during training |
| Parameter count | Constant | Increases gradually |
| Initial size | Full size | Can start smaller |
| Training time | Baseline | +5-10% overhead |
| Memory | Constant | Gradually increases |
| Adaptability | None | Adds capacity where needed |

## Advanced: Programmatic Control

If you want more control in your code:

```python
# In train.py or custom training loop:
from models.subsampling_model import Subsampling_Model

model = Subsampling_Model(
    # ... standard params ...
    type='DynamicUnet',
    growth_method='sample_select',
    n_candidates=10,
    growth_interval=10,
    layers_to_grow=['bottleneck', 'down_3']
)

# During training:
if model.type == 'DynamicUnet':
    # Manual growth
    growth_occurred = model.grow_network_if_needed(
        epoch=current_epoch,
        batch_data=sample_input,
        target_data=sample_target,
        criterion=torch.nn.MSELoss()
    )
    
    if growth_occurred:
        # Reset optimizer to include new parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Check network size
    sizes = model.get_network_size()
    print(f"Current size: {sizes}")
```

## Questions?

Check the detailed documentation:
- `DYNAMIC_UNET_README.md` - Technical details
- `dynamic_unet_example.py` - Code examples
- `test_dynamic_unet.py` - Unit tests

Happy training! 🚀

