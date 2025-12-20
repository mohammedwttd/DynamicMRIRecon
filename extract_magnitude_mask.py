#!/usr/bin/env python3
"""
Extract magnitude-based mask from a trained UNet.
Prunes the lowest 60% weights globally.

Usage:
    python extract_magnitude_mask.py --checkpoint path/to/model.pt --sparsity 0.6
"""

import argparse
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.rec_models.models.unet_model import UnetModel as Unet


def extract_magnitude_masks(model, sparsity=0.6, exclude_layers=None):
    """
    Create masks by keeping top (1-sparsity) weights by magnitude.
    
    Args:
        model: Trained model
        sparsity: Fraction of weights to prune (0.6 = remove lowest 60%)
        exclude_layers: Layer patterns to skip
    
    Returns:
        Dictionary of binary masks
    """
    exclude_layers = exclude_layers or []
    
    def is_excluded(name):
        for pattern in exclude_layers:
            if pattern in name:
                return True
        return False
    
    def classify_layer(name):
        name_lower = name.lower()
        if 'down' in name_lower or 'encoder' in name_lower:
            return 'encoder'
        elif 'up' in name_lower or 'decoder' in name_lower:
            return 'decoder'
        return 'bottleneck'
    
    # Collect all weight magnitudes
    layer_info = {}
    all_weights = []
    
    print(f"\n{'='*80}")
    print(f"Magnitude Pruning: Target {sparsity*100:.0f}% sparsity")
    print(f"{'='*80}")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if is_excluded(name):
                print(f"[SKIP]  {name}: {module.weight.numel():,} params")
                continue
            
            layer_info[name] = {
                'module': module,
                'type': classify_layer(name),
                'numel': module.weight.numel(),
            }
            all_weights.append(module.weight.data.abs().flatten())
            print(f"[PRUNE] {name}: {module.weight.numel():,} params")
    
    # Concatenate all weights
    all_weights = torch.cat(all_weights)
    total_params = len(all_weights)
    
    # Find global threshold for target density
    target_density = 1 - sparsity
    k = int(total_params * target_density)
    
    if k > 0:
        threshold = torch.topk(all_weights, k, largest=True).values[-1].item()
    else:
        threshold = float('inf')
    
    print(f"\nGlobal magnitude threshold: {threshold:.6f}")
    print(f"Keeping top {k:,} / {total_params:,} weights ({target_density*100:.0f}%)")
    
    # Create masks
    masks = {}
    
    print(f"\n{'='*80}")
    print(f"Layer-wise Statistics")
    print(f"{'='*80}")
    print(f"{'Type':<12} {'Layer Name':<40} {'Active':>10} {'Total':>10} {'Density':>8}")
    print(f"{'-'*80}")
    
    encoder_active, encoder_total = 0, 0
    decoder_active, decoder_total = 0, 0
    bottleneck_active, bottleneck_total = 0, 0
    
    for name, info in layer_info.items():
        module = info['module']
        weight_mag = module.weight.data.abs()
        
        # Create mask: 1 where |weight| >= threshold
        mask = (weight_mag >= threshold).float()
        masks[name] = mask
        
        active = int(mask.sum().item())
        total = mask.numel()
        density = active / total * 100
        layer_type = info['type']
        
        print(f"{layer_type.upper():<12} {name:<40} {active:>10,} {total:>10,} {density:>7.1f}%")
        
        if layer_type == 'encoder':
            encoder_active += active
            encoder_total += total
        elif layer_type == 'decoder':
            decoder_active += active
            decoder_total += total
        else:
            bottleneck_active += active
            bottleneck_total += total
    
    print(f"{'-'*80}")
    
    total_active = encoder_active + decoder_active + bottleneck_active
    total_all = encoder_total + decoder_total + bottleneck_total
    
    print(f"\nDensity by Region:")
    if encoder_total > 0:
        print(f"  Encoder:    {encoder_active:>10,} / {encoder_total:>10,} ({encoder_active/encoder_total*100:.1f}%)")
    if decoder_total > 0:
        print(f"  Decoder:    {decoder_active:>10,} / {decoder_total:>10,} ({decoder_active/decoder_total*100:.1f}%)")
    if bottleneck_total > 0:
        print(f"  Bottleneck: {bottleneck_active:>10,} / {bottleneck_total:>10,} ({bottleneck_active/bottleneck_total*100:.1f}%)")
    print(f"  TOTAL:      {total_active:>10,} / {total_all:>10,} ({total_active/total_all*100:.1f}%)")
    print(f"{'='*80}\n")
    
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--sparsity', type=float, default=0.6, help='Target sparsity (0.6 = 60% zeros)')
    parser.add_argument('--num-chans', type=int, default=32, help='Base channels (32 or 48)')
    parser.add_argument('--num-pools', type=int, default=4)
    parser.add_argument('--output', type=str, default='magnitude_masks.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    # Create model
    model = Unet(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=0.0
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove various prefixes
        new_k = k.replace('module.', '')
        new_k = new_k.replace('reconstruction_model.', '')
        new_state_dict[new_k] = v
    state_dict = new_state_dict
    
    # Try to load (may need to filter keys)
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    
    if len(filtered_state) == 0:
        print("Available keys in checkpoint:")
        for k in list(state_dict.keys())[:10]:
            print(f"  {k}")
        print("...")
        raise ValueError("Could not match checkpoint keys to model")
    
    model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded {len(filtered_state)} parameters")
    
    # Extract masks
    exclude = []  # No exclusions - apply to all layers
    masks = extract_magnitude_masks(model, sparsity=args.sparsity, exclude_layers=exclude)
    
    # Save
    save_dict = {
        'masks': {k: v.cpu() for k, v in masks.items()},
        'sparsity': args.sparsity,
        'method': 'magnitude',
        'checkpoint': args.checkpoint,
    }
    torch.save(save_dict, args.output)
    print(f"✓ Masks saved to: {args.output}")


if __name__ == '__main__':
    main()

