#!/usr/bin/env python3
"""
Test script to evaluate RigL80, RigL90, and Unet (50 epochs, 7.76M params) 
on FastMRI and M4Raw datasets with 1500 samples each.

Usage:
    python tests/test_rigl80_90_unet_50epochs.py --data-path ../data --device cuda
"""

import argparse
import pathlib
import os
import sys
import time
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
        from torch.utils.data import Subset
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

        files = list(pathlib.Path(root).glob('*.h5'))

        if sample_rate < 1:
            import random
            random.seed(42)
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        reached_limit = False
        skipped = 0

        for fname in sorted(files):
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

                    # Skip a couple edge slices (often noisy/empty) and validate reads
                    for slice_idx in range(2, max(2, num_slices - 2)):
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
            print(f"Skipped {skipped} corrupted M4Raw slices")
        if max_samples is not None and len(self.examples) >= max_samples:
            print(f"Reached max_samples limit: {max_samples}")

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
    Returns:
      - input: (H, W, 2) complex (RSS)
      - target: (1, H, W) magnitude
    """
    def __init__(self, target_resolution=320):
        self.target_resolution = target_resolution

    def __call__(self, kspace, target, attrs, fname, slice_idx):
        kspace = transforms.to_tensor(kspace)  # (coils, H, W, 2)
        orig_h, orig_w = kspace.shape[1], kspace.shape[2]

        # Zero-pad in k-space to upsample
        if orig_h < self.target_resolution or orig_w < self.target_resolution:
            pad_h = (self.target_resolution - orig_h) // 2
            pad_w = (self.target_resolution - orig_w) // 2
            kspace_padded = torch.zeros(
                kspace.shape[0], self.target_resolution, self.target_resolution, 2,
                dtype=kspace.dtype
            )
            kspace_padded[:, pad_h:pad_h + orig_h, pad_w:pad_w + orig_w, :] = kspace
            kspace = kspace_padded

        image = transforms.ifft2_regular(kspace)  # (coils, H, W, 2)
        image = transforms.complex_center_crop(image, (self.target_resolution, self.target_resolution))
        rss_image = fastmri.rss(image)  # (H, W, 2) complex for this codebase

        # Target in image domain -> resize to 320 if needed
        target = torch.tensor(target).float()
        if target.shape[0] != self.target_resolution or target.shape[1] != self.target_resolution:
            target = target.unsqueeze(0).unsqueeze(0)
            target = F.interpolate(
                target,
                size=(self.target_resolution, self.target_resolution),
                mode='bilinear',
                align_corners=False
            )
            target = target.squeeze(0)  # (1, H, W)
        else:
            target = target.unsqueeze(0)  # (1, H, W)

        target = normalize(target)

        mean = std = 0
        norm = np.float32(attrs.get('norm', 1.0)) if isinstance(attrs.get('norm', 1.0), (int, float)) else np.float32(1.0)
        return rss_image, target, mean, std, norm


def create_m4raw_loader(data_path, resolution=320, batch_size=1, num_workers=8, acquisition_filter=None, max_samples=None):
    """Create M4Raw data loader."""
    m4raw_path = data_path / 'm4raw' / 'multicoil_val'
    if not m4raw_path.exists():
        print(f"Warning: M4Raw path not found: {m4raw_path}")
        return None, 0

    m4raw_data = M4RawSliceData(
        root=m4raw_path,
        transform=M4RawTransform(target_resolution=resolution),
        sample_rate=1,
        acquisition_filter=acquisition_filter,
        max_samples=max_samples
    )

    m4raw_loader = DataLoader(
        dataset=m4raw_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    return m4raw_loader, len(m4raw_data)


def build_model(args, device):
    """Build the model from saved args."""
    model = Subsampling_Model(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        res=args.resolution,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization,
        SNR=args.SNR,
        n_shots=args.n_shots,
        interp_gap=getattr(args, 'interp_gap', 10),
        type=args.model,
        img_size=getattr(args, 'img_size', [320, 320]),
        window_size=getattr(args, 'window_size', 10),
        embed_dim=getattr(args, 'embed_dim', 66),
        num_blocks=getattr(args, 'num_blocks', 1),
        sample_per_shot=getattr(args, 'sample_per_shot', 3001),
        epsilon=getattr(args, 'epsilon', 0),
        noise_p=0,
        std=getattr(args, 'std', 0),
        acceleration=getattr(args, 'acceleration', 4),
        center_fraction=getattr(args, 'center_fraction', 0.08),
        noise=getattr(args, 'noise_behaviour', 'constant'),
        epochs=getattr(args, 'num_epochs', 50),
        swap_frequency=getattr(args, 'swap_frequency', 10),
        num_experts=getattr(args, 'num_experts', 8),
        fd_kernel_num=getattr(args, 'fd_kernel_num', 64),
        fd_use_simple=getattr(args, 'fd_use_simple', False),
        snake_layers=getattr(args, 'snake_layers', 2),
        snake_kernel_size=getattr(args, 'snake_kernel_size', 9),
        use_dcn=getattr(args, 'use_dcn', False),
        use_fdconv=getattr(args, 'use_fdconv', False),
    ).to(device)
    
    return model


def apply_rigl_masks(model, masks, device):
    """Apply RigL masks to model weights to enforce sparsity."""
    applied_count = 0
    total_zeros = 0
    total_params = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
            
            if mask_key in masks:
                mask = masks[mask_key].to(device)
                param.data.mul_(mask)
                applied_count += 1
                total_zeros += (mask == 0).sum().item()
                total_params += mask.numel()
    
    if applied_count > 0:
        sparsity = 100 * total_zeros / total_params
        print(f"  Applied {applied_count} RigL masks: {total_zeros:,}/{total_params:,} weights zeroed ({sparsity:.2f}% sparse)")
    
    return applied_count


def load_model(checkpoint_path, device, apply_masks=True):
    """Load model from checkpoint and optionally apply RigL masks."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    model = build_model(args, device)
    
    state_dict = checkpoint['model']
    
    # Filter out thop profiling keys
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if 'total_ops' not in k and 'total_params' not in k
    }
    
    if list(filtered_state_dict.keys())[0].startswith('module.'):
        model = torch.nn.DataParallel(model)
    
    model.load_state_dict(filtered_state_dict)
    
    # Apply RigL masks if they exist
    rigl_scheduler = checkpoint.get('rigl_scheduler')
    masks_applied = 0
    if apply_masks and rigl_scheduler is not None and 'masks' in rigl_scheduler:
        masks = rigl_scheduler['masks']
        masks_applied = apply_rigl_masks(model, masks, device)
    
    model.eval()
    
    return model, args, masks_applied


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
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  [{model_name}] Processed {batch_idx + 1}/{len(data_loader)} batches...")
    
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


# Define models to test: All numbered RigL models (10-90) and Unet (50 epochs, 7.76M params)
MODELS_TO_TEST = OrderedDict([
    ('Unet_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_Unet_50_train-fastmri_params-7756737',
        'description': 'Dense U-Net baseline (32 chans, ~7.76M params, 50 epochs)',
    }),
    ('RigL10_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL10_50_train-fastmri_params-7756737',
        'description': 'RigL 10% sparsity (32 chans, ~7.76M total, 50 epochs)',
    }),
    ('RigL20_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL20_50_train-fastmri_params-7756737',
        'description': 'RigL 20% sparsity (32 chans, ~7.76M total, 50 epochs)',
    }),
    ('RigL30_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL30_50_train-fastmri_params-7756737',
        'description': 'RigL 30% sparsity (32 chans, ~7.76M total, 50 epochs)',
    }),
    ('RigL40_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL40_50_train-fastmri_params-7756737',
        'description': 'RigL 40% sparsity (32 chans, ~7.76M total, 50 epochs)',
    }),
    ('RigL50_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL50_50_train-fastmri_params-7756737',
        'description': 'RigL 50% sparsity (32 chans, ~7.76M total, 50 epochs)',
    }),
    ('RigL60_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL60_50_train-fastmri_params-7756737',
        'description': 'RigL 60% sparsity (32 chans, ~7.76M total, 50 epochs)',
    }),
    ('RigL70_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL70_50_train-fastmri_params-7756737',
        'description': 'RigL 70% sparsity (32 chans, ~7.76M total, 50 epochs)',
    }),
    ('RigL80_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL80_50_train-fastmri_params-7756737',
        'description': 'RigL 80% sparsity (32 chans, ~7.76M total, 50 epochs)',
    }),
    ('RigL90_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL90_50_train-fastmri_params-7756737',
        'description': 'RigL 90% sparsity (32 chans, ~7.76M total, 50 epochs)',
    }),
])


def print_results_table(results, models_order, title="TEST SET RESULTS"):
    """Print results in a formatted table."""
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(f"{'Model':<18} | {'PSNR (dB)':<20} | {'SSIM':<20} | {'Non-zero Params':<18} | {'Sparsity':<10}")
    print("-" * 110)
    
    for model_name in models_order:
        if model_name not in results:
            continue
        metrics = results[model_name]
        psnr_str = f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}"
        ssim_str = f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"
        params_str = f"{metrics['nonzero_params']/1e6:.2f}M"
        sparsity_str = f"{metrics['sparsity']:.1f}%"
        print(f"{model_name:<18} | {psnr_str:<20} | {ssim_str:<20} | {params_str:<18} | {sparsity_str:<10}")
    
    print("=" * 110)


def evaluate_on_dataset(models_to_test, data_loader, device, dataset_name="Test"):
    """Evaluate all models on a given dataset."""
    results = {}
    
    for model_name, model_info in models_to_test.items():
        checkpoint_dir = resolve_checkpoint_dir(model_info['checkpoint'])
        checkpoint_path = checkpoint_dir / 'best_model.pt'
        
        if not checkpoint_path.exists():
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}. Skipping {model_name}.")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Testing: {model_name} on {dataset_name}")
        print(f"Description: {model_info['description']}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'=' * 80}")
        
        try:
            print("Loading model...")
            model, saved_args, masks_applied = load_model(checkpoint_path, device, apply_masks=True)
            
            total_params, nonzero_params = count_nonzero_params(model)
            sparsity = 100 * (1 - nonzero_params / total_params) if total_params > 0 else 0
            print(f"  Parameters: {nonzero_params:,}/{total_params:,} non-zero ({sparsity:.2f}% sparse)")
            
            print("Evaluating...")
            metrics = evaluate_model(model, data_loader, device, model_name)
            
            metrics['total_params'] = total_params
            metrics['nonzero_params'] = nonzero_params
            metrics['sparsity'] = sparsity
            metrics['masks_applied'] = masks_applied
            results[model_name] = metrics
            
            print(f"\nResults for {model_name} ({dataset_name}):")
            print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
            print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
            print(f"  Eval time: {metrics['eval_time']:.1f}s")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError testing {model_name} on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test RigL80, RigL90, and Unet (50 epochs) on FastMRI and M4Raw')
    parser.add_argument('--data-path', type=pathlib.Path, 
                        default=pathlib.Path('../data'),
                        help='Path to the dataset directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--fastmri-max-samples', type=int, default=1500,
                        help='Maximum number of FastMRI samples to evaluate (default: 1500)')
    parser.add_argument('--m4raw-max-samples', type=int, default=1500,
                        help='Maximum number of M4Raw samples to evaluate (default: 1500)')
    parser.add_argument('--fastmri-only', action='store_true', default=False,
                        help='Only evaluate on FastMRI (skip M4Raw)')
    parser.add_argument('--m4raw-only', action='store_true', default=False,
                        help='Only evaluate on M4Raw (skip FastMRI)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 80}")
    print("ALL RIGL MODELS (10-90) + UNET (50 EPOCHS) EVALUATION")
    print(f"{'=' * 80}")
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nModels to evaluate:")
    for name, info in MODELS_TO_TEST.items():
        print(f"  - {name}: {info['description']}")
    
    all_results = {}
    models_order = list(MODELS_TO_TEST.keys())
    
    # ==================== FastMRI Test Set Evaluation ====================
    if not args.m4raw_only:
        print("\n" + "=" * 80)
        print(f"FASTMRI TEST SET EVALUATION ({args.fastmri_max_samples} samples)")
        print("=" * 80)
        
        print(f"\nLoading FastMRI test data from: {args.data_path / 'multicoil_test'}")
        test_loader, num_samples = create_test_loader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.fastmri_max_samples
        )
        print(f"Total test samples: {num_samples}")
        
        fastmri_results = evaluate_on_dataset(MODELS_TO_TEST, test_loader, device, "FastMRI Test")
        all_results['fastmri'] = fastmri_results
        
        if fastmri_results:
            print_results_table(fastmri_results, models_order, 
                              f"ALL RIGL (10-90) + UNET (50 EPOCHS) - FASTMRI TEST RESULTS ({num_samples} samples)")
    
    # ==================== M4Raw Evaluation ====================
    if not args.fastmri_only:
        print("\n" + "=" * 80)
        print(f"M4RAW DOMAIN SHIFT EVALUATION ({args.m4raw_max_samples} samples)")
        print("=" * 80)
        
        m4raw_loader, m4raw_samples = create_m4raw_loader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.m4raw_max_samples
        )
        
        if m4raw_loader is not None and m4raw_samples > 0:
            print(f"Total M4Raw samples: {m4raw_samples}")
            
            m4raw_results = evaluate_on_dataset(MODELS_TO_TEST, m4raw_loader, device, "M4Raw")
            all_results['m4raw'] = m4raw_results
            
            if m4raw_results:
                print_results_table(m4raw_results, models_order,
                                  f"ALL RIGL (10-90) + UNET (50 EPOCHS) - M4RAW RESULTS ({m4raw_samples} samples)")
        else:
            print("Warning: M4Raw dataset not available or empty.")
    
    # ==================== Combined Summary ====================
    if 'fastmri' in all_results and 'm4raw' in all_results:
        print("\n" + "=" * 110)
        print("COMBINED SUMMARY - FASTMRI vs M4RAW")
        print("=" * 110)
        print(f"{'Model':<18} | {'FastMRI PSNR':<15} | {'M4Raw PSNR':<15} | {'FastMRI SSIM':<15} | {'M4Raw SSIM':<15}")
        print("-" * 110)
        
        for model_name in models_order:
            fm = all_results['fastmri'].get(model_name, {})
            m4 = all_results['m4raw'].get(model_name, {})
            
            fm_psnr = f"{fm.get('psnr_mean', 0):.2f}" if fm else "N/A"
            m4_psnr = f"{m4.get('psnr_mean', 0):.2f}" if m4 else "N/A"
            fm_ssim = f"{fm.get('ssim_mean', 0):.4f}" if fm else "N/A"
            m4_ssim = f"{m4.get('ssim_mean', 0):.4f}" if m4 else "N/A"
            
            print(f"{model_name:<18} | {fm_psnr:<15} | {m4_psnr:<15} | {fm_ssim:<15} | {m4_ssim:<15}")
        
        print("=" * 110)
    
    # Save results
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    main()

