#!/usr/bin/env python3
"""
Test script to evaluate RigL models with different mask initialization methods
on FastMRI Brain or M4Raw test set (1000 samples).

Models tested:
- RigL40: SNIP static, Random dynamic, PUNIT static, PUNIT dynamic
- RigL60: SNIP static, Random dynamic, PUNIT static, PUNIT dynamic

Excludes magnitude-based initialization (still running).

Usage:
    python tests/test_rigl_mask_init_1000samples.py --data-path ../data --device cuda
    python tests/test_rigl_mask_init_1000samples.py --data-path ../data --device cuda --dataset m4raw
"""

import argparse
import pathlib
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastmri
from data import transforms
from data.mri_data import SliceData
from models.subsampling_model import Subsampling_Model
from common.evaluate import psnr, ssim


def normalize(img):
    """Normalize image to [0, 1] range."""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def resolve_checkpoint_dir(checkpoint_dir: str) -> pathlib.Path:
    """
    Backward/forward compatible checkpoint directory resolution.
    New training runs may append: *_train-<dataset>_params-<N>
    """
    p = pathlib.Path(checkpoint_dir)
    if p.exists():
        return p
    parent = p.parent
    base = p.name
    matches = sorted(parent.glob(base + "_train-*_params-*"))
    if matches:
        return matches[-1]
    return p


class DataTransform:
    """Transform for test data."""
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice_idx):
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        target = normalize(transforms.to_tensor(target))
        mean = std = 0

        if target.shape[1] != self.resolution:
            target = transforms.center_crop(target, (self.resolution, self.resolution))
        return fastmri.rss(image), target, mean, std, attrs['norm'].astype(np.float32)


def create_test_loader(data_path, resolution=320, batch_size=1, num_workers=8, max_samples=None):
    """Create test data loader."""
    test_data = SliceData(
        root=data_path / 'multicoil_test',
        transform=DataTransform(resolution),
        sample_rate=1
    )
    
    if max_samples is not None and len(test_data) > max_samples:
        indices = list(range(max_samples))
        test_data = Subset(test_data, indices)
    
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    return test_loader, len(test_data)


# ==================== M4Raw Dataset Support ====================

class M4RawSliceData(Dataset):
    """
    M4Raw dataset for evaluation.
    
    Expected layout:
      {data_path}/m4raw/multicoil_val/*.h5
    Each file should have:
      - kspace: (slices, coils, H, W) complex
      - reconstruction_rss: (slices, H, W)
    """
    def __init__(self, root, transform, sample_rate=1, acquisition_filter=None, max_samples=None):
        self.transform = transform
        self.examples = []
        
        files = sorted(pathlib.Path(root).glob('*.h5'))
        
        if sample_rate < 1:
            import random
            random.seed(42)
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        
        print(f"  Scanning {len(files)} M4Raw files (validating each slice)...")
        
        reached_limit = False
        skipped = 0
        
        for fname in files:
            if reached_limit:
                break
            try:
                with h5py.File(fname, 'r') as data:
                    if acquisition_filter:
                        acq = data.attrs.get('acquisition', '')
                        if acquisition_filter.upper() not in str(acq).upper():
                            continue
                    
                    if 'kspace' not in data or 'reconstruction_rss' not in data:
                        continue
                    
                    kspace = data['kspace']
                    num_slices = kspace.shape[0]
                    
                    # Skip edge slices and validate each slice
                    for slice_idx in range(2, max(3, num_slices - 2)):
                        try:
                            _ = data['kspace'][slice_idx]
                            _ = data['reconstruction_rss'][slice_idx]
                            self.examples.append((fname, slice_idx))
                            if max_samples is not None and len(self.examples) >= max_samples:
                                reached_limit = True
                                break
                        except OSError:
                            skipped += 1
                            continue
            except Exception:
                continue
        
        if skipped > 0:
            print(f"  Skipped {skipped} corrupted slices")
        print(f"  Found {len(self.examples)} valid slices")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        fname, slice_idx = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice_idx]  # (coils, H, W)
            target = data['reconstruction_rss'][slice_idx]  # (H, W)
            attrs = {
                'norm': data.attrs.get('max', 1.0),
                'acquisition': data.attrs.get('acquisition', 'unknown'),
            }
            return self.transform(kspace, target, attrs, fname.name, slice_idx)


class M4RawTransform:
    """
    Transform for M4Raw data.
    M4Raw is typically 256x256; we k-space zero-pad to 320x320.
    
    Input: k-space zero-padding -> sinc interpolation (upsampling)
    Target: bilinear interpolation to match input resolution
    """
    def __init__(self, target_resolution=320):
        self.target_resolution = target_resolution
    
    def __call__(self, kspace, target, attrs, fname, slice_idx):
        kspace = transforms.to_tensor(kspace)  # (coils, H, W, 2)
        orig_h, orig_w = kspace.shape[1], kspace.shape[2]
        
        # Zero-pad in k-space to upsample (sinc interpolation)
        if orig_h < self.target_resolution or orig_w < self.target_resolution:
            pad_h = (self.target_resolution - orig_h) // 2
            pad_w = (self.target_resolution - orig_w) // 2
            kspace_padded = torch.zeros(
                kspace.shape[0], self.target_resolution, self.target_resolution, 2,
                dtype=kspace.dtype
            )
            kspace_padded[:, pad_h:pad_h + orig_h, pad_w:pad_w + orig_w, :] = kspace
            kspace = kspace_padded
        
        # Compute RSS image from k-space
        image = transforms.ifft2_regular(kspace)  # (coils, H, W, 2)
        image = transforms.complex_center_crop(image, (self.target_resolution, self.target_resolution))
        rss_image = fastmri.rss(image)  # (H, W, 2) complex
        
        # Process target - use bilinear interpolation to match sinc-interpolated input
        target = torch.tensor(target).float()
        if target.shape[0] != self.target_resolution or target.shape[1] != self.target_resolution:
            # Bilinear interpolation to resize target
            target = target.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            target = F.interpolate(
                target,
                size=(self.target_resolution, self.target_resolution),
                mode='bilinear',
                align_corners=False
            )
            target = target.squeeze(0)  # (1, H, W)
        else:
            target = target.unsqueeze(0)  # (1, H, W)
        
        # Normalize
        target = normalize(target)
        
        mean = std = 0
        norm = attrs.get('norm', 1.0)
        if isinstance(norm, (int, float)):
            norm = np.float32(norm)
        elif isinstance(norm, np.ndarray):
            norm = np.float32(norm.item() if norm.size == 1 else 1.0)
        else:
            norm = np.float32(1.0)
        
        return rss_image, target, mean, std, norm


def create_m4raw_loader(data_path, resolution=320, batch_size=1, num_workers=8, max_samples=None):
    """Create M4Raw data loader."""
    m4raw_path = data_path / 'm4raw' / 'multicoil_val'
    
    if not m4raw_path.exists():
        print(f"Warning: M4Raw path not found: {m4raw_path}")
        return None, 0
    
    m4raw_data = M4RawSliceData(
        root=m4raw_path,
        transform=M4RawTransform(target_resolution=resolution),
        sample_rate=1,
        max_samples=max_samples
    )
    
    # Use fewer workers for M4Raw to avoid H5 file access issues
    effective_workers = min(num_workers, 4)
    
    m4raw_loader = DataLoader(
        dataset=m4raw_data,
        batch_size=batch_size,
        num_workers=effective_workers,
        pin_memory=True,
        shuffle=False
    )
    
    return m4raw_loader, len(m4raw_data)


def load_model(checkpoint_path, device, apply_masks=True):
    """Load a trained model from checkpoint."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    
    if checkpoint_path.is_dir():
        if (checkpoint_path / 'best_model.pt').exists():
            checkpoint_path = checkpoint_path / 'best_model.pt'
        elif (checkpoint_path / 'model.pt').exists():
            checkpoint_path = checkpoint_path / 'model.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get('args', None)
    
    # Build model using saved args
    model = Subsampling_Model(
        in_chans=getattr(saved_args, 'in_chans', 1),
        out_chans=getattr(saved_args, 'out_chans', 1),
        chans=getattr(saved_args, 'num_chans', 32),
        num_pool_layers=getattr(saved_args, 'num_pools', 4),
        drop_prob=getattr(saved_args, 'drop_prob', 0),
        decimation_rate=getattr(saved_args, 'decimation_rate', 4),
        res=getattr(saved_args, 'resolution', 320),
        trajectory_learning=getattr(saved_args, 'trajectory_learning', 0),
        initialization=getattr(saved_args, 'initialization', 'cartesian'),
        n_shots=getattr(saved_args, 'n_shots', 16),
        interp_gap=getattr(saved_args, 'interp_gap', 50),
        SNR=getattr(saved_args, 'SNR', False),
        type=getattr(saved_args, 'model', 'Unet'),
        img_size=(getattr(saved_args, 'resolution', 320), getattr(saved_args, 'resolution', 320)),
        acceleration=getattr(saved_args, 'acceleration', 4),
        center_fraction=getattr(saved_args, 'center_fraction', 0.08),
    ).to(device)
    
    # Load weights
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    # Apply masks if present
    masks_applied = False
    if apply_masks and 'rigl_scheduler' in checkpoint and 'masks' in checkpoint['rigl_scheduler']:
        masks = checkpoint['rigl_scheduler']['masks']
        for name, mask in masks.items():
            # Navigate to the module
            parts = name.replace('module.', '').split('.')
            module = model
            for part in parts:
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    module = None
                    break
            
            if module is not None and hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data *= mask.to(device)
        masks_applied = True
    
    return model, saved_args, masks_applied


def evaluate_model(model, data_loader, device, model_name="Model"):
    """Evaluate model on test data."""
    model.eval()
    psnr_list = []
    ssim_list = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            input_data, target, mean, std, norm = data
            input_data = input_data.to(device)
            target = target.to(device)
            
            output = model(input_data.unsqueeze(1))
            
            resolution = target.shape[-1]
            recons = output.cpu().squeeze(1).view(target.shape)
            target_cpu = target.cpu()
            
            if output.shape != target.shape:
                target_cpu = target_cpu.view_as(output.cpu())
            
            recons = recons.squeeze().view(-1, resolution, resolution)
            target_cpu = target_cpu.view(-1, resolution, resolution)
            
            psnr_val = psnr(target_cpu.numpy(), recons.numpy())
            ssim_val = ssim(target_cpu.numpy(), recons.numpy())
            
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            
            if (batch_idx + 1) % 200 == 0:
                print(f"  [{model_name}] Processed {batch_idx + 1}/{len(data_loader)} samples...")
    
    eval_time = time.time() - start_time
    
    return {
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'eval_time': eval_time,
        'num_samples': len(psnr_list)
    }


def count_nonzero_params(model):
    """Count non-zero parameters in reconstruction model."""
    total = 0
    nonzero = 0
    for name, param in model.named_parameters():
        if 'reconstruction_model' in name:
            total += param.numel()
            nonzero += (param != 0).sum().item()
    return total, nonzero


def print_results_table(results, title="RESULTS"):
    """Print results in a formatted table."""
    print(f"\n{'=' * 100}")
    print(f"{title}")
    print(f"{'=' * 100}")
    print(f"{'Model':<55} {'PSNR (dB)':<18} {'SSIM':<18} {'Sparsity':<12} {'Time (s)':<10}")
    print(f"{'-' * 100}")
    
    # Sort by PSNR descending
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('psnr_mean', 0), reverse=True)
    
    for model_name, metrics in sorted_results:
        psnr_str = f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}"
        ssim_str = f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"
        sparsity_str = f"{metrics.get('sparsity', 0):.1f}%"
        time_str = f"{metrics.get('eval_time', 0):.1f}"
        print(f"{model_name:<55} {psnr_str:<18} {ssim_str:<18} {sparsity_str:<12} {time_str:<10}")
    
    print(f"{'=' * 100}\n")


# ==================== MODELS TO TEST ====================
# Excluding magnitude-based (still running)

MODELS_TO_TEST = OrderedDict([
    # ==================== RigL40 (40% Sparsity) ====================
    ('RigL40_SNIP_Static', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL40_50_mask-snip_40_single_batch_static_train-fastmri_params-3348867',
        'description': 'RigL40 with SNIP initialization, static masks',
        'sparsity': 40,
        'init': 'SNIP',
        'mode': 'Static'
    }),
    ('RigL40_Random_Dynamic', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL40_50_dynamic_train-fastmri_params-3348867',
        'description': 'RigL40 with random initialization, dynamic RigL',
        'sparsity': 40,
        'init': 'Random',
        'mode': 'Dynamic'
    }),
    ('RigL40_PUNIT_Static', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL40_50_mask-punit_4_static_train-fastmri_params-3348867',
        'description': 'RigL40 with PUN-IT initialization, static masks',
        'sparsity': 40,
        'init': 'PUN-IT',
        'mode': 'Static'
    }),
    ('RigL40_PUNIT_Dynamic', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL40_50_mask-punit_4_dynamic_train-fastmri_params-3348867',
        'description': 'RigL40 with PUN-IT initialization, dynamic RigL',
        'sparsity': 40,
        'init': 'PUN-IT',
        'mode': 'Dynamic'
    }),
    
    # ==================== RigL60 (60% Sparsity) ====================
    ('RigL60_SNIP_Static', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL60_50_mask-snip_60_single_batch_static_train-fastmri_params-3348867',
        'description': 'RigL60 with SNIP initialization, static masks',
        'sparsity': 60,
        'init': 'SNIP',
        'mode': 'Static'
    }),
    ('RigL60_Random_Dynamic', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL60_50_dynamic_train-fastmri_params-3348867',
        'description': 'RigL60 with random initialization, dynamic RigL',
        'sparsity': 60,
        'init': 'Random',
        'mode': 'Dynamic'
    }),
    ('RigL60_PUNIT_Static', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL60_50_mask-punit_6_static_train-fastmri_params-3348867',
        'description': 'RigL60 with PUN-IT initialization, static masks',
        'sparsity': 60,
        'init': 'PUN-IT',
        'mode': 'Static'
    }),
    ('RigL60_PUNIT_Dynamic', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL60_50_mask-punit_6_dynamic_train-fastmri_params-3348867',
        'description': 'RigL60 with PUN-IT initialization, dynamic RigL',
        'sparsity': 60,
        'init': 'PUN-IT',
        'mode': 'Dynamic'
    }),
])


def main():
    parser = argparse.ArgumentParser(description='Test RigL models with different mask initializations')
    parser.add_argument('--data-path', type=pathlib.Path, default=pathlib.Path('../data'),
                        help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Maximum number of samples to test (default: 1000)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file for results (optional)')
    parser.add_argument('--dataset', type=str, default='fastmri', choices=['fastmri', 'm4raw'],
                        help='Dataset to test on (default: fastmri)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset_name = "FastMRI Brain" if args.dataset == 'fastmri' else "M4Raw"
    
    print(f"\n{'=' * 80}")
    print(f"RigL Mask Initialization Comparison Test")
    print(f"{'=' * 80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {dataset_name}")
    print(f"Max samples: {args.max_samples}")
    print(f"Data path: {args.data_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")
    
    # Load test data based on dataset choice
    if args.dataset == 'fastmri':
        print(f"Loading FastMRI Brain test data...")
        test_loader, num_samples = create_test_loader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples
        )
    else:
        print(f"Loading M4Raw test data...")
        test_loader, num_samples = create_m4raw_loader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples
        )
        if test_loader is None:
            print("Failed to load M4Raw data. Exiting.")
            return
    
    print(f"Loaded {num_samples} test samples\n")
    
    # Store results
    results = {}
    
    # Test each model
    for model_name, model_info in MODELS_TO_TEST.items():
        checkpoint_dir = resolve_checkpoint_dir(model_info['checkpoint'])
        checkpoint_path = checkpoint_dir / 'best_model.pt' if checkpoint_dir.is_dir() else checkpoint_dir
        
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / 'model.pt'
        
        if not checkpoint_path.exists():
            print(f"\n⚠️  Checkpoint not found for {model_name}: {checkpoint_dir}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Testing: {model_name}")
        print(f"  Description: {model_info['description']}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"{'=' * 80}")
        
        try:
            # Load model
            print("Loading model...")
            model, saved_args, masks_applied = load_model(checkpoint_path, device, apply_masks=True)
            
            # Count parameters
            total_params, nonzero_params = count_nonzero_params(model)
            actual_sparsity = 100 * (1 - nonzero_params / total_params) if total_params > 0 else 0
            print(f"  Parameters: {nonzero_params:,}/{total_params:,} non-zero ({actual_sparsity:.1f}% sparse)")
            print(f"  Masks applied: {masks_applied}")
            
            # Evaluate
            print("Evaluating...")
            metrics = evaluate_model(model, test_loader, device, model_name)
            
            # Store results
            metrics['total_params'] = total_params
            metrics['nonzero_params'] = nonzero_params
            metrics['sparsity'] = actual_sparsity
            metrics['masks_applied'] = masks_applied
            metrics['init'] = model_info['init']
            metrics['mode'] = model_info['mode']
            metrics['target_sparsity'] = model_info['sparsity']
            results[model_name] = metrics
            
            print(f"\n✓ Results for {model_name}:")
            print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
            print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
            print(f"  Eval time: {metrics['eval_time']:.1f}s")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final results
    if results:
        # Group by sparsity level
        rigl40_results = {k: v for k, v in results.items() if 'RigL40' in k}
        rigl60_results = {k: v for k, v in results.items() if 'RigL60' in k}
        
        if rigl40_results:
            print_results_table(rigl40_results, f"RIGL40 (40% SPARSITY) - {dataset_name.upper()} TEST RESULTS")
        
        if rigl60_results:
            print_results_table(rigl60_results, f"RIGL60 (60% SPARSITY) - {dataset_name.upper()} TEST RESULTS")
        
        # Print summary comparison
        print(f"\n{'=' * 80}")
        print("SUMMARY: Best Initialization Method per Sparsity Level")
        print(f"{'=' * 80}")
        
        if rigl40_results:
            best_40 = max(rigl40_results.items(), key=lambda x: x[1]['psnr_mean'])
            print(f"\n40% Sparsity Winner: {best_40[0]}")
            print(f"  PSNR: {best_40[1]['psnr_mean']:.2f} dB, SSIM: {best_40[1]['ssim_mean']:.4f}")
        
        if rigl60_results:
            best_60 = max(rigl60_results.items(), key=lambda x: x[1]['psnr_mean'])
            print(f"\n60% Sparsity Winner: {best_60[0]}")
            print(f"  PSNR: {best_60[1]['psnr_mean']:.2f} dB, SSIM: {best_60[1]['ssim_mean']:.4f}")
        
        print(f"\n{'=' * 80}\n")
        
        # Save results to file if specified
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                serializable_results = {}
                for model_name, metrics in results.items():
                    serializable_results[model_name] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metrics.items()
                    }
                json.dump(serializable_results, f, indent=2)
            print(f"Results saved to: {args.output_file}")
    else:
        print("\n⚠️  No models were successfully tested!")


if __name__ == '__main__':
    main()

