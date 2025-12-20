#!/usr/bin/env python3
"""
Find sparse masks for a U-Net model before training using SNIP or PUN-IT.

Uses the local UnetModel (from models.rec_models.models.unet_model), not fastmri.

Usage:
    # SNIP on FastMRI (default):
    python find_punit_mask.py --method snip --sparsity 0.6 --data-path ../data
    
    # SNIP on ImageNet:
    python find_punit_mask.py --method snip --dataset imagenet --data-path /path/to/imagenet --sparsity 0.6
    
    # PUN-IT (original):
    python find_punit_mask.py --method punit --sparsity 0.6 --num-steps 500 --data-path ../data
    
This will:
1. Create a fresh U-Net model (local UnetModel)
2. Find optimal sparse masks
3. Save masks to a file for later use in training
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.rec_models.layers.punit_mask import PUNITMaskFinder, SNIPMaskFinder
from models.rec_models.models.unet_model import UnetModel
import numpy as np


class SimpleTransform:
    """Simple transform for mask finding (FastMRI)."""
    def __init__(self, resolution=320):
        self.resolution = resolution
    
    def __call__(self, kspace, target, attrs, fname, slice_idx):
        from data import transforms
        import fastmri
        
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        
        # RSS returns [H, W, 2] (complex format with real/imag)
        rss_img = fastmri.rss(image)  # [H, W, 2]
        
        # Convert complex to magnitude: sqrt(real^2 + imag^2)
        if rss_img.dim() == 3 and rss_img.shape[-1] == 2:
            input_img = torch.sqrt(rss_img[..., 0]**2 + rss_img[..., 1]**2)  # [H, W]
        elif rss_img.dim() == 2:
            input_img = rss_img  # Already [H, W]
        else:
            input_img = rss_img
        
        # Add channel dimension: [H, W] -> [1, H, W]
        if input_img.dim() == 2:
            input_img = input_img.unsqueeze(0)
        
        # Target processing
        target = transforms.to_tensor(target)
        if target.dim() == 2:
            target = target.unsqueeze(0)  # [H, W] -> [1, H, W]
        # Handle complex target
        if target.dim() == 3 and target.shape[-1] == 2:
            target = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)
            target = target.unsqueeze(0)
        if target.shape[-1] != self.resolution or target.shape[-2] != self.resolution:
            target = transforms.center_crop(target, (self.resolution, self.resolution))
        
        # Normalize to [0, 1]
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        return input_img, target


def create_fastmri_dataloader(data_path, batch_size=4, num_samples=200):
    """Create a small dataloader for mask finding (FastMRI)."""
    from torch.utils.data import Subset
    from data.mri_data import SliceData
    
    train_path = os.path.join(data_path, 'multicoil_train')
    
    dataset = SliceData(
        root=train_path,
        transform=SimpleTransform(resolution=320),
        sample_rate=1.0
    )
    
    # Use subset for faster mask finding
    if num_samples and len(dataset) > num_samples:
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return loader


class ImageNetReconDataset(torch.utils.data.Dataset):
    """Wrapper to convert ImageNet (image, label) to (image, image) for reconstruction."""
    def __init__(self, imagenet_dataset):
        self.dataset = imagenet_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Ignore class label
        return image, image  # Use image as both input and target


def create_imagenet_dataloader(data_path, batch_size=4, num_samples=200, resolution=224):
    """Create ImageNet dataloader for mask finding."""
    from torch.utils.data import Subset
    from torchvision import datasets, transforms as T
    
    # ImageNet transforms
    transform = T.Compose([
        T.Resize(resolution),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Try train folder first, then val
    train_path = os.path.join(data_path, 'train')
    if not os.path.exists(train_path):
        train_path = os.path.join(data_path, 'val')
    if not os.path.exists(train_path):
        train_path = data_path  # Direct path to ImageFolder
    
    print(f"Loading ImageNet from: {train_path}")
    base_dataset = datasets.ImageFolder(train_path, transform=transform)
    
    # Wrap to return (image, image) instead of (image, label)
    dataset = ImageNetReconDataset(base_dataset)
    
    # Use subset for faster mask finding
    if num_samples and len(dataset) > num_samples:
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return loader


def create_dataloader(data_path, dataset_type='fastmri', batch_size=4, num_samples=200, resolution=320):
    """Create dataloader based on dataset type."""
    if dataset_type == 'imagenet':
        return create_imagenet_dataloader(data_path, batch_size, num_samples, resolution=224)
    else:
        return create_fastmri_dataloader(data_path, batch_size, num_samples)


def main():
    parser = argparse.ArgumentParser(description='Find PUN-IT masks')
    
    # Model args
    parser.add_argument('--in-chans', type=int, default=None, help='Input channels (auto: 1 for MRI, 3 for ImageNet)')
    parser.add_argument('--out-chans', type=int, default=None, help='Output channels (auto: 1 for MRI, 3 for ImageNet)')
    parser.add_argument('--num-chans', type=int, default=32, help='Base channels (32=3.35M, 48=7.75M)')
    parser.add_argument('--num-pools', type=int, default=4)
    parser.add_argument('--drop-prob', type=float, default=0.0)
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='fastmri', choices=['fastmri', 'imagenet'],
                        help='Dataset: fastmri (1-channel MRI) or imagenet (3-channel RGB)')
    
    # Method selection
    parser.add_argument('--method', type=str, default='snip', choices=['snip', 'punit'],
                        help='Method: snip (recommended) or punit')
    
    # Common args
    parser.add_argument('--sparsity', type=float, default=0.6, help='Target sparsity (0.6 = 60% zeros)')
    
    # SNIP args
    parser.add_argument('--num-batches', type=int, default=10, help='Batches for SNIP gradient computation')
    
    # PUN-IT args (only used if --method punit)
    parser.add_argument('--num-steps', type=int, default=100, help='Optimization steps (PUN-IT only)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (PUN-IT only)')
    parser.add_argument('--sparsity-weight', type=float, default=10.0, help='Sparsity loss weight (PUN-IT only)')
    parser.add_argument('--temperature', type=float, default=5.0, help='Initial Gumbel-Softmax temperature (PUN-IT only)')
    parser.add_argument('--init-noise', type=float, default=0.5, help='Noise std to add to initial logits (breaks symmetry)')
    
    # Data args
    parser.add_argument('--data-path', type=str, default='../data', help='Path to data')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-samples', type=int, default=200, help='Samples to use for mask finding')
    
    # Output
    parser.add_argument('--output', type=str, default='punit_masks.pt', help='Output file for masks')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Auto-set channels based on dataset
    if args.in_chans is None:
        args.in_chans = 3 if args.dataset == 'imagenet' else 1
    if args.out_chans is None:
        args.out_chans = 3 if args.dataset == 'imagenet' else 1
    
    print("="*80)
    print(f"Mask Finding using {args.method.upper()}")
    print("="*80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: UnetModel (local) with {args.num_chans} base channels")
    print(f"Input/Output channels: {args.in_chans}/{args.out_chans}")
    print(f"Target sparsity: {args.sparsity*100:.0f}%")
    print(f"Method: {args.method}")
    print("="*80)
    
    # Create model (local small UNet, not fastmri)
    model = UnetModel(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create dataloader
    print("\nLoading data...")
    dataloader = create_dataloader(
        args.data_path,
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    print(f"Using {len(dataloader)} batches for mask finding")
    
    exclude_layers = []  # No exclusions - apply to all layers
    
    if args.method == 'snip':
        # SNIP: Single-shot Network Pruning (RECOMMENDED)
        finder = SNIPMaskFinder(
            model=model,
            sparsity=args.sparsity,
            exclude_layers=exclude_layers,
            device=args.device
        )
        
        print("\nComputing SNIP importance scores...")
        masks = finder.find_masks(
            dataloader=dataloader,
            criterion=nn.MSELoss(),
            num_batches=args.num_batches,
        )
    else:
        # PUN-IT: Probability optimization
        finder = PUNITMaskFinder(
            model=model,
            sparsity=args.sparsity,
            temperature=args.temperature,
            exclude_layers=exclude_layers,
            device=args.device,
            init_noise=args.init_noise,  # Breaks symmetry in initial probabilities
        )
        
        print("\nStarting PUN-IT optimization...")
        masks = finder.find_masks(
            dataloader=dataloader,
            criterion=nn.MSELoss(),
            num_steps=args.num_steps,
            lr=args.lr,
            sparsity_weight=args.sparsity_weight,
            print_freq=50
        )
    
    # Save masks
    save_dict = {
        'masks': {k: v.cpu() for k, v in masks.items()},
        'sparsity': args.sparsity,
        'method': args.method,
        'dataset': args.dataset,
        'in_chans': args.in_chans,
        'out_chans': args.out_chans,
        'num_chans': args.num_chans,
        'num_pools': args.num_pools,
    }
    torch.save(save_dict, args.output)
    print(f"\n✓ Masks saved to: {args.output}")
    
    # Print summary
    active = sum(m.sum().item() for m in masks.values())
    total = sum(m.numel() for m in masks.values())
    print(f"\nFinal Summary:")
    print(f"  Active parameters: {int(active):,} / {int(total):,}")
    print(f"  Actual sparsity: {(1 - active/total)*100:.1f}%")
    print(f"  Target sparsity: {args.sparsity*100:.0f}%")


if __name__ == '__main__':
    main()

