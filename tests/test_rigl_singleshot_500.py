#!/usr/bin/env python3
"""
Single-shot evaluation of RigL models on FastMRI Brain and M4Raw (500 samples each).

Models:
- RigL50 (10 epochs) - early training checkpoint
- RigL95 (50 epochs) - high sparsity
- RigL99 (50 epochs) - extreme sparsity

Usage:
    python tests/test_rigl_singleshot_500.py --data-path ../data --device cuda
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
from torch.utils.data import DataLoader, Dataset, Subset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastmri
from data import transforms
from data.mri_data import SliceData
from models.subsampling_model import Subsampling_Model
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def psnr(gt, pred):
    """Compute PSNR for 2D images."""
    return sk_psnr(gt, pred, data_range=gt.max() - gt.min())


def ssim(gt, pred):
    """Compute SSIM for 2D images."""
    return sk_ssim(gt, pred, data_range=gt.max() - gt.min())


def normalize(img):
    """Normalize image to [0, 1] range."""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


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


class M4RawSliceData(Dataset):
    """M4Raw dataset for domain shift evaluation."""
    def __init__(self, root, transform, max_samples=None):
        self.transform = transform
        self.examples = []
        
        if not root.exists():
            print(f"Warning: M4Raw path not found: {root}")
            return
            
        files = sorted(list(root.glob('*.h5')))
        for fname in files:
            with h5py.File(fname, 'r') as f:
                kspace = f['kspace']
                num_slices = kspace.shape[0]
                for slice_idx in range(num_slices):
                    self.examples.append((fname, slice_idx))
                    if max_samples and len(self.examples) >= max_samples:
                        return
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        fname, slice_idx = self.examples[idx]
        with h5py.File(fname, 'r') as f:
            kspace = f['kspace'][slice_idx]
            target = f['reconstruction_rss'][slice_idx] if 'reconstruction_rss' in f else None
            attrs = {'norm': f.attrs.get('norm', 1.0) if 'norm' in f.attrs else 1.0}
        return self.transform(kspace, target, attrs, fname.name, slice_idx)


class M4RawTransform:
    """Transform for M4Raw data with zero-padding to 320x320."""
    def __init__(self, target_resolution=320):
        self.target_resolution = target_resolution
    
    def __call__(self, kspace, target, attrs, fname, slice_idx):
        kspace = transforms.to_tensor(kspace)
        orig_h, orig_w = kspace.shape[1], kspace.shape[2]
        
        # Zero-pad k-space to target resolution
        if orig_h < self.target_resolution or orig_w < self.target_resolution:
            pad_h = max(0, self.target_resolution - orig_h)
            pad_w = max(0, self.target_resolution - orig_w)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            kspace = F.pad(kspace.permute(0, 3, 1, 2), 
                          (pad_left, pad_right, pad_top, pad_bottom), 
                          mode='constant', value=0).permute(0, 2, 3, 1)
        
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.target_resolution, self.target_resolution))
        rss_image = fastmri.rss(image)
        
        if target is not None:
            target = normalize(transforms.to_tensor(target))
            if target.shape[0] < self.target_resolution or target.shape[1] < self.target_resolution:
                pad_h = max(0, self.target_resolution - target.shape[0])
                pad_w = max(0, self.target_resolution - target.shape[1])
                target = F.pad(target.unsqueeze(0), (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2)).squeeze(0)
        else:
            target = rss_image[..., 0]
        
        norm = np.float32(attrs.get('norm', 1.0)) if isinstance(attrs.get('norm', 1.0), (int, float)) else np.float32(1.0)
        return rss_image, target, 0, 0, norm


def create_test_loader(data_path, resolution=320, batch_size=1, num_workers=4, max_samples=500):
    """Create FastMRI Brain test data loader."""
    test_data = SliceData(
        root=data_path / 'multicoil_test',
        transform=DataTransform(resolution),
        sample_rate=1
    )
    
    if max_samples and len(test_data) > max_samples:
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


def create_m4raw_loader(data_path, resolution=320, batch_size=1, num_workers=4, max_samples=500):
    """Create M4Raw data loader."""
    m4raw_path = data_path / 'm4raw' / 'multicoil_val'
    
    if not m4raw_path.exists():
        print(f"Warning: M4Raw path not found: {m4raw_path}")
        return None, 0
    
    m4raw_data = M4RawSliceData(
        root=m4raw_path,
        transform=M4RawTransform(target_resolution=resolution),
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


def load_model(checkpoint_path, device, apply_masks=True):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get('args', None)
    
    if saved_args is None:
        raise ValueError(f"No args found in checkpoint: {checkpoint_path}")
    
    model = Subsampling_Model(
        in_chans=saved_args.in_chans,
        out_chans=saved_args.out_chans,
        chans=saved_args.num_chans,
        num_pool_layers=4,
        drop_prob=0.0,
        decimation_rate=saved_args.decimation_rate,
        res=saved_args.resolution,
        trajectory_learning=saved_args.trajectory_learning,
        initialization=saved_args.initialization,
        n_shots=saved_args.n_shots,
        interp_gap=saved_args.interp_gap,
        type=saved_args.model,
        acceleration=saved_args.acceleration,
        center_fraction=saved_args.center_fraction
    )
    
    # Handle DataParallel checkpoints (remove 'module.' prefix)
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        if k.startswith('module.'):
            new_key = k[7:]  # Remove 'module.'
        else:
            new_key = k
        # Skip thop profiling keys (total_ops, total_params)
        if 'total_ops' in new_key or 'total_params' in new_key:
            continue
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    # Apply RigL masks if available
    masks_applied = 0
    if apply_masks and 'rigl_scheduler' in checkpoint:
        raw_masks = checkpoint['rigl_scheduler'].get('masks', {})
        # Convert mask keys (remove 'module.' prefix if present)
        masks = {}
        for k, v in raw_masks.items():
            if k.startswith('module.'):
                new_key = k[7:]
            else:
                new_key = k
            masks[new_key] = v
        
        if masks:
            for name, param in model.named_parameters():
                # Masks are stored with layer name (without .weight/.bias suffix)
                # Try matching with and without the suffix
                layer_name = name.rsplit('.', 1)[0] if name.endswith(('.weight', '.bias')) else name
                
                if layer_name in masks and name.endswith('.weight'):
                    mask = masks[layer_name].to(device)
                    param.data.mul_(mask)
                    masks_applied += 1
    
    return model, saved_args, masks_applied


def count_nonzero_params(model):
    """Count non-zero parameters."""
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += (param != 0).sum().item()
    return total, nonzero


def evaluate_model(model, data_loader, device, model_name="Model"):
    """Standard evaluation without single-shot adaptation."""
    model.eval()
    psnr_list = []
    ssim_list = []
    resolution = 320
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_img, target, mean, std, norm = batch
            input_img = input_img.to(device)
            
            output = model(input_img)
            output = output.cpu()
            
            # Normalize output
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            
            # Ensure both are 2D (H, W) for single sample
            # Target shape handling
            if target.dim() == 4:  # (B, C, H, W)
                target = target.squeeze(0).squeeze(0)
            elif target.dim() == 3:  # (B, H, W) or (C, H, W)
                target = target.squeeze(0)
            
            # Output shape handling - model returns (B*H*W) flattened or (B, H, W)
            if output.dim() == 1:  # Flattened
                output = output.view(resolution, resolution)
            elif output.dim() == 3:  # (B, H, W)
                output = output.squeeze(0)
            elif output.dim() == 2 and output.shape[0] != resolution:
                # Might be (B, H*W)
                output = output.view(resolution, resolution)
            
            # Debug first batch
            if batch_idx == 0:
                print(f"    Target shape: {target.shape}, Output shape: {output.shape}")
            
            # Calculate metrics
            psnr_val = psnr(target.numpy(), output.numpy())
            ssim_val = ssim(target.numpy(), output.numpy())
            
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  [{model_name}] Processed {batch_idx + 1}/{len(data_loader)}...")
    
    return {
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'num_samples': len(psnr_list),
    }


def single_shot_evaluate(model, data_loader, device, masks, criterion, model_name="Model",
                         num_steps=20, lr=1e-4):
    """Single-shot adaptation: adapt on first sample, evaluate on rest."""
    model.eval()
    
    # Get first sample for adaptation
    data_iter = iter(data_loader)
    adapt_batch = next(data_iter)
    adapt_input, adapt_target, _, _, _ = adapt_batch
    adapt_input = adapt_input.to(device)
    adapt_target = adapt_target.to(device)
    
    # Compute loss before adaptation
    resolution = 320
    with torch.no_grad():
        adapt_output = model(adapt_input)
        # Reshape target to match output
        if adapt_target.dim() == 4:  # (B, C, H, W)
            adapt_target = adapt_target.squeeze(1)  # (B, H, W)
        elif adapt_target.dim() == 3 and adapt_target.shape[0] != 1:
            adapt_target = adapt_target.unsqueeze(0)  # Add batch dim
        # Reshape output if needed
        if adapt_output.dim() == 1:
            adapt_output = adapt_output.view(1, resolution, resolution)
        loss_before = criterion(adapt_output, adapt_target).item()
    
    # Create optimizer for masked parameters only
    # Masks are stored with layer name (without .weight/.bias suffix)
    params_to_adapt = []
    param_to_mask = {}  # Map param name to mask key
    for name, param in model.named_parameters():
        layer_name = name.rsplit('.', 1)[0] if name.endswith(('.weight', '.bias')) else name
        if layer_name in masks and name.endswith('.weight'):
            param.requires_grad = True
            params_to_adapt.append(param)
            param_to_mask[name] = layer_name
        else:
            param.requires_grad = False
    
    if not params_to_adapt:
        print(f"  Warning: No masked parameters found for adaptation")
        # Fall back to standard evaluation
        model.eval()
        return evaluate_model(model, data_loader, device, model_name)
    
    optimizer = torch.optim.Adam(params_to_adapt, lr=lr)
    
    # Adaptation loop
    model.train()
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        output = model(adapt_input)
        # Reshape output to match target
        if output.dim() == 1:
            output = output.view(1, resolution, resolution)
        loss = criterion(output, adapt_target)
        loss.backward()
        
        # Only update masked (pruned) positions
        for name, param in model.named_parameters():
            if name in param_to_mask and param.grad is not None:
                mask_key = param_to_mask[name]
                mask = masks[mask_key].to(device)
                # Gradient only for pruned positions (where mask is 0)
                param.grad.mul_(1 - mask)
        
        optimizer.step()
        losses.append(loss.item())
    
    loss_after = losses[-1] if losses else loss_before
    
    # Evaluate on remaining samples
    model.eval()
    psnr_list = []
    ssim_list = []
    resolution = 320
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            input_img, target, mean, std, norm = batch
            input_img = input_img.to(device)
            
            output = model(input_img)
            output = output.cpu()
            
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            
            # Target shape handling
            if target.dim() == 4:  # (B, C, H, W)
                target = target.squeeze(0).squeeze(0)
            elif target.dim() == 3:  # (B, H, W) or (C, H, W)
                target = target.squeeze(0)
            
            # Output shape handling
            if output.dim() == 1:  # Flattened
                output = output.view(resolution, resolution)
            elif output.dim() == 3:  # (B, H, W)
                output = output.squeeze(0)
            elif output.dim() == 2 and output.shape[0] != resolution:
                output = output.view(resolution, resolution)
            
            psnr_val = psnr(target.numpy(), output.numpy())
            ssim_val = ssim(target.numpy(), output.numpy())
            
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  [{model_name}] Processed {batch_idx + 1}/{len(data_loader)-1}...")
    
    return {
        'loss_before': loss_before,
        'loss_after': loss_after,
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'num_samples': len(psnr_list),
        'adaptation_steps': num_steps,
    }


# Models to test
MODELS_TO_TEST = OrderedDict([
    ('RigL50_10ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL50_10',
        'description': 'RigL 50% sparsity (10 epochs training)',
    }),
    ('RigL95_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL95_50',
        'description': 'RigL 95% sparsity (50 epochs training)',
    }),
    ('RigL99_50ep', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL99_50',
        'description': 'RigL 99% sparsity (50 epochs training)',
    }),
])


def print_results_table(results, title="RESULTS"):
    """Print results in a formatted table."""
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(f"{'Model':<18} | {'PSNR (dB)':<20} | {'SSIM':<20} | {'Non-zero Params':<18} | {'Sparsity':<10}")
    print("-" * 110)
    
    for model_name in MODELS_TO_TEST.keys():
        if model_name not in results:
            continue
        metrics = results[model_name]
        psnr_str = f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}"
        ssim_str = f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"
        params_str = f"{metrics.get('nonzero_params', 0)/1e6:.2f}M"
        sparsity_str = f"{metrics.get('sparsity', 0):.1f}%"
        print(f"{model_name:<18} | {psnr_str:<20} | {ssim_str:<20} | {params_str:<18} | {sparsity_str:<10}")
    
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description='Single-shot evaluation of RigL models')
    parser.add_argument('--data-path', type=pathlib.Path, default=pathlib.Path('../data'),
                        help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Maximum samples per dataset (default: 500)')
    parser.add_argument('--single-shot', action='store_true', default=True,
                        help='Enable single-shot adaptation')
    parser.add_argument('--single-shot-steps', type=int, default=20,
                        help='Number of adaptation steps')
    parser.add_argument('--single-shot-lr', type=float, default=1e-4,
                        help='Learning rate for adaptation')
    parser.add_argument('--no-single-shot', action='store_true', default=False,
                        help='Disable single-shot (standard eval only)')
    
    args = parser.parse_args()
    
    if args.no_single_shot:
        args.single_shot = False
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("SINGLE-SHOT EVALUATION: RigL50 (10ep), RigL95 (50ep), RigL99 (50ep)")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Single-shot adaptation: {args.single_shot}")
    if args.single_shot:
        print(f"  Steps: {args.single_shot_steps}, LR: {args.single_shot_lr}")
    
    brain_results = {}
    m4raw_results = {}
    
    # ==================== FastMRI Brain Evaluation ====================
    print(f"\n{'=' * 80}")
    print(f"FASTMRI BRAIN TEST SET ({args.max_samples} samples)")
    print(f"{'=' * 80}")
    
    brain_loader, brain_samples = create_test_loader(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )
    print(f"Loaded {brain_samples} samples")
    
    for model_name, model_info in MODELS_TO_TEST.items():
        checkpoint_path = pathlib.Path(model_info['checkpoint']) / 'best_model.pt'
        
        if not checkpoint_path.exists():
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Testing: {model_name}")
        print(f"  {model_info['description']}")
        print(f"{'=' * 60}")
        
        model, saved_args, masks_applied = load_model(checkpoint_path, device, apply_masks=True)
        total_params, nonzero_params = count_nonzero_params(model)
        sparsity = 100 * (1 - nonzero_params / total_params)
        
        print(f"  Params: {nonzero_params:,}/{total_params:,} ({sparsity:.1f}% sparse)")
        print(f"  Masks applied: {masks_applied}")
        
        # Load checkpoint for masks (handle module. prefix)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_masks = checkpoint.get('rigl_scheduler', {}).get('masks', {})
        masks = {}
        for k, v in raw_masks.items():
            new_key = k[7:] if k.startswith('module.') else k
            masks[new_key] = v
        
        if args.single_shot and masks:
            print(f"  Running single-shot adaptation...")
            metrics = single_shot_evaluate(
                model, brain_loader, device, masks,
                criterion=torch.nn.L1Loss(),
                model_name=model_name,
                num_steps=args.single_shot_steps,
                lr=args.single_shot_lr
            )
            print(f"  Loss: {metrics.get('loss_before', 0):.6f} → {metrics.get('loss_after', 0):.6f}")
        else:
            print(f"  Running standard evaluation...")
            metrics = evaluate_model(model, brain_loader, device, model_name)
        
        metrics['nonzero_params'] = nonzero_params
        metrics['total_params'] = total_params
        metrics['sparsity'] = sparsity
        brain_results[model_name] = metrics
        
        print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        
        # Reload model for M4Raw (reset adaptation)
        del model
        torch.cuda.empty_cache()
    
    # ==================== M4Raw Evaluation ====================
    print(f"\n{'=' * 80}")
    print(f"M4RAW DOMAIN SHIFT ({args.max_samples} samples)")
    print(f"{'=' * 80}")
    
    m4raw_loader, m4raw_samples = create_m4raw_loader(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )
    
    if m4raw_loader is None or m4raw_samples == 0:
        print("M4Raw dataset not available, skipping...")
    else:
        print(f"Loaded {m4raw_samples} samples")
        
        for model_name, model_info in MODELS_TO_TEST.items():
            checkpoint_path = pathlib.Path(model_info['checkpoint']) / 'best_model.pt'
            
            if not checkpoint_path.exists():
                continue
            
            print(f"\n{'=' * 60}")
            print(f"Testing: {model_name} on M4Raw")
            print(f"{'=' * 60}")
            
            model, saved_args, masks_applied = load_model(checkpoint_path, device, apply_masks=True)
            total_params, nonzero_params = count_nonzero_params(model)
            sparsity = 100 * (1 - nonzero_params / total_params)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            raw_masks = checkpoint.get('rigl_scheduler', {}).get('masks', {})
            masks = {}
            for k, v in raw_masks.items():
                new_key = k[7:] if k.startswith('module.') else k
                masks[new_key] = v
            
            if args.single_shot and masks:
                print(f"  Running single-shot adaptation...")
                metrics = single_shot_evaluate(
                    model, m4raw_loader, device, masks,
                    criterion=torch.nn.L1Loss(),
                    model_name=model_name,
                    num_steps=args.single_shot_steps,
                    lr=args.single_shot_lr
                )
                print(f"  Loss: {metrics.get('loss_before', 0):.6f} → {metrics.get('loss_after', 0):.6f}")
            else:
                print(f"  Running standard evaluation...")
                metrics = evaluate_model(model, m4raw_loader, device, model_name)
            
            metrics['nonzero_params'] = nonzero_params
            metrics['total_params'] = total_params
            metrics['sparsity'] = sparsity
            m4raw_results[model_name] = metrics
            
            print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
            print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
            
            del model
            torch.cuda.empty_cache()
    
    # ==================== Summary ====================
    if brain_results:
        print_results_table(brain_results, f"FASTMRI BRAIN RESULTS ({args.max_samples} samples, single-shot={args.single_shot})")
    
    if m4raw_results:
        print_results_table(m4raw_results, f"M4RAW RESULTS ({args.max_samples} samples, single-shot={args.single_shot})")
    
    # Comparison table
    if brain_results and m4raw_results:
        print("\n" + "=" * 130)
        print("DOMAIN SHIFT COMPARISON: FastMRI Brain vs M4Raw")
        print("=" * 130)
        print(f"{'Model':<18} | {'Brain PSNR':<18} | {'M4Raw PSNR':<18} | {'Δ PSNR':<10} | {'Brain SSIM':<18} | {'M4Raw SSIM':<18} | {'Δ SSIM':<10}")
        print("-" * 130)
        
        for model_name in MODELS_TO_TEST.keys():
            if model_name not in brain_results or model_name not in m4raw_results:
                continue
            
            b = brain_results[model_name]
            m = m4raw_results[model_name]
            
            psnr_diff = m['psnr_mean'] - b['psnr_mean']
            ssim_diff = m['ssim_mean'] - b['ssim_mean']
            
            b_psnr = f"{b['psnr_mean']:.2f} ± {b['psnr_std']:.2f}"
            m_psnr = f"{m['psnr_mean']:.2f} ± {m['psnr_std']:.2f}"
            b_ssim = f"{b['ssim_mean']:.4f} ± {b['ssim_std']:.4f}"
            m_ssim = f"{m['ssim_mean']:.4f} ± {m['ssim_std']:.4f}"
            
            print(f"{model_name:<18} | {b_psnr:<18} | {m_psnr:<18} | {psnr_diff:+.2f} dB   | {b_ssim:<18} | {m_ssim:<18} | {ssim_diff:+.4f}")
        
        print("=" * 130)


if __name__ == '__main__':
    main()

