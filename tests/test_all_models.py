#!/usr/bin/env python3
"""
Test script to evaluate all trained models on the test set.

This script loads each trained model checkpoint and evaluates it on the 
multicoil_test dataset, reporting PSNR and SSIM metrics.

Supports domain shift evaluation using M4Raw dataset.

Usage:
    python test_all_models.py [--data-path PATH] [--use-best] [--device DEVICE]
    python test_all_models.py --m4raw  # Include M4Raw domain shift evaluation
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
    """
    M4Raw dataset for domain shift evaluation.
    
    M4Raw has different data format:
    - Resolution: 256x256 (will be padded to 320x320)
    - kspace shape: (slices, coils, H, W)
    - Different acquisition types: T1, T2, FLAIR
    """
    def __init__(self, root, transform, sample_rate=1, acquisition_filter=None, max_samples=None):
        """
        Args:
            root: Path to M4Raw multicoil_val directory
            transform: Transform to apply
            sample_rate: Fraction of data to use
            acquisition_filter: Filter by acquisition type (e.g., 'T1', 'T2', 'FLAIR')
            max_samples: Maximum number of samples to use (None = all)
        """
        self.transform = transform
        self.examples = []
        self.max_samples = max_samples
        
        files = list(pathlib.Path(root).glob('*.h5'))
        
        if sample_rate < 1:
            import random
            random.seed(42)
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        
        skipped = 0
        done = False
        for fname in sorted(files):
            if done:
                break
            try:
                with h5py.File(fname, 'r') as data:
                    # Check acquisition type if filter specified
                    if acquisition_filter:
                        acq = data.attrs.get('acquisition', '')
                        if acquisition_filter.upper() not in acq.upper():
                            continue
                    
                    kspace = data['kspace']
                    num_slices = kspace.shape[0]
                    
                    # Skip edge slices and validate each slice
                    for slice_idx in range(2, num_slices - 2):
                        try:
                            # Try reading the slice to check if it's corrupted
                            _ = data['kspace'][slice_idx]
                            _ = data['reconstruction_rss'][slice_idx]
                            self.examples.append((fname, slice_idx))
                            
                            # Check if we've reached max_samples
                            if max_samples and len(self.examples) >= max_samples:
                                done = True
                                break
                        except OSError:
                            skipped += 1
                            continue
            except Exception as e:
                print(f"Warning: Could not read {fname}: {e}")
                continue
        
        if skipped > 0:
            print(f"Skipped {skipped} corrupted slices")
        if max_samples:
            print(f"Limited to {len(self.examples)} samples (max_samples={max_samples})")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        fname, slice_idx = self.examples[i]
        
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice_idx]  # Shape: (coils, H, W)
            target = data['reconstruction_rss'][slice_idx]  # Shape: (H, W)
            
            # Create attrs-like dict with norm value
            attrs = {
                'norm': data.attrs.get('max', 1.0),
                'acquisition': data.attrs.get('acquisition', 'unknown')
            }
            
            return self.transform(kspace, target, attrs, fname.name, slice_idx)


class M4RawTransform:
    """
    Transform for M4Raw data.
    
    M4Raw has 256x256 resolution, we need to handle this for models trained on 320x320.
    Uses zero-padding in k-space to upsample to 320x320.
    """
    def __init__(self, target_resolution=320):
        self.target_resolution = target_resolution
    
    def __call__(self, kspace, target, attrs, fname, slice_idx):
        # kspace shape: (coils, H, W) - M4Raw format
        # Convert to tensor
        kspace = transforms.to_tensor(kspace)  # (coils, H, W, 2)
        
        orig_h, orig_w = kspace.shape[1], kspace.shape[2]
        
        # Zero-pad k-space to target resolution for upsampling
        if orig_h < self.target_resolution or orig_w < self.target_resolution:
            pad_h = (self.target_resolution - orig_h) // 2
            pad_w = (self.target_resolution - orig_w) // 2
            
            # Pad k-space (zero-padding in frequency domain = interpolation in image domain)
            kspace_padded = torch.zeros(
                kspace.shape[0], self.target_resolution, self.target_resolution, 2,
                dtype=kspace.dtype
            )
            kspace_padded[:, pad_h:pad_h+orig_h, pad_w:pad_w+orig_w, :] = kspace
            kspace = kspace_padded
        
        # Compute image from k-space
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.target_resolution, self.target_resolution))
        
        # RSS combination
        rss_image = fastmri.rss(image)
        
        # Process target
        target = torch.tensor(target).float()
        
        # Resize target to match model input resolution
        if target.shape[0] != self.target_resolution or target.shape[1] != self.target_resolution:
            target = target.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            target = F.interpolate(target, size=(self.target_resolution, self.target_resolution), 
                                   mode='bilinear', align_corners=False)
            target = target.squeeze(0)  # Remove batch dim, keep channel
        else:
            target = target.unsqueeze(0)  # Add channel dim
        
        # Normalize
        target = normalize(target)
        
        mean = std = 0
        norm = np.float32(attrs.get('norm', 1.0)) if isinstance(attrs.get('norm', 1.0), (int, float)) else np.float32(1.0)
        
        return rss_image, target, mean, std, norm


def create_test_loader(data_path, resolution=320, batch_size=1, num_workers=8):
    """Create test data loader."""
    test_data = SliceData(
        root=data_path / 'multicoil_test',
        transform=DataTransform(resolution),
        sample_rate=1
    )
    
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    return test_loader, len(test_data)


def create_m4raw_loader(data_path, resolution=320, batch_size=1, num_workers=8, acquisition_filter=None, max_samples=None):
    """Create M4Raw data loader for domain shift evaluation."""
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
        epochs=getattr(args, 'num_epochs', 30),
        # Dynamic U-Net parameters
        swap_frequency=getattr(args, 'swap_frequency', 10),
        # CondUnet parameters
        num_experts=getattr(args, 'num_experts', 8),
        # FDUnet parameters
        fd_kernel_num=getattr(args, 'fd_kernel_num', 64),
        fd_use_simple=getattr(args, 'fd_use_simple', False),
        # HybridSnakeFDUnet parameters
        snake_layers=getattr(args, 'snake_layers', 2),
        snake_kernel_size=getattr(args, 'snake_kernel_size', 9),
        # ConfigurableUNet parameters (LightUNet, LightDCN, LightFD, LightDCNFD)
        use_dcn=getattr(args, 'use_dcn', False),
        use_fdconv=getattr(args, 'use_fdconv', False),
    ).to(device)
    
    return model


def load_model(checkpoint_path, device, apply_rigl_masks=True):
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        apply_rigl_masks: If True, apply RigL masks to ensure pruned weights are 0
    
    Returns:
        model: Loaded model
        args: Saved arguments
        mask_info: Dict with mask application info (masks_applied, sparsity, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    model = build_model(args, device)
    
    # Handle DataParallel state dict
    state_dict = checkpoint['model']
    
    # Filter out thop profiling keys (total_ops, total_params) that were added during training
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if 'total_ops' not in k and 'total_params' not in k
    }
    
    # Check if model was saved with DataParallel
    is_dataparallel = list(filtered_state_dict.keys())[0].startswith('module.')
    if is_dataparallel:
        # Model was saved with DataParallel, wrap current model
        model = torch.nn.DataParallel(model)
    
    model.load_state_dict(filtered_state_dict)
    model.eval()
    
    # Apply RigL masks if available
    mask_info = {
        'masks_applied': 0,
        'total_params': 0,
        'nonzero_params': 0,
        'sparsity': 0.0,
        'has_rigl': False
    }
    
    if apply_rigl_masks and 'rigl_scheduler' in checkpoint:
        mask_info['has_rigl'] = True
        raw_masks = checkpoint['rigl_scheduler'].get('masks', {})
        
        if raw_masks:
            # Convert mask keys (handle module. prefix)
            masks = {}
            for k, v in raw_masks.items():
                # Remove 'module.' prefix if model is not DataParallel
                # Or keep it if model is DataParallel
                if is_dataparallel and not k.startswith('module.'):
                    new_key = 'module.' + k
                elif not is_dataparallel and k.startswith('module.'):
                    new_key = k[7:]  # Remove 'module.'
                else:
                    new_key = k
                masks[new_key] = v
            
            # Apply masks to model parameters
            for name, param in model.named_parameters():
                # Masks are stored with layer name (without .weight/.bias suffix)
                layer_name = name.rsplit('.', 1)[0] if name.endswith(('.weight', '.bias')) else name
                
                if layer_name in masks and name.endswith('.weight'):
                    mask = masks[layer_name].to(device)
                    param.data.mul_(mask)
                    mask_info['masks_applied'] += 1
    
    # Count parameters
    mask_info['total_params'], mask_info['nonzero_params'] = count_parameters(model)
    if mask_info['total_params'] > 0:
        mask_info['sparsity'] = 100 * (1 - mask_info['nonzero_params'] / mask_info['total_params'])
    
    return model, args, mask_info


def count_parameters(model):
    """Count total and non-zero parameters."""
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += (param != 0).sum().item()
    return total, nonzero


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
            
            # Forward pass
            output = model(input_data.unsqueeze(1))
            
            # Reshape for metric calculation
            resolution = target.shape[-1]
            recons = output.cpu().squeeze(1).view(target.shape)
            target_cpu = target.cpu()
            
            if output.shape != target.shape:
                target_cpu = target_cpu.view_as(output.cpu())
            
            recons = recons.squeeze().view(-1, resolution, resolution)
            target_cpu = target_cpu.view(-1, resolution, resolution)
            
            # Calculate metrics
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


def single_shot_evaluate(model, data_loader, device, masks, criterion, model_name="Model",
                         num_steps=20, lr=1e-3, save_dir=None):
    """
    Single-shot adaptation using FrozenMaskTrainer: adapt on first sample using pruned weights.
    
    Uses FrozenMaskTrainer which handles:
    - Xavier reinitialization of pruned weights
    - Freezing active (learned) weights during adaptation
    - Tracking and restoring best weights
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run on
        masks: Dict of RigL masks (layer_name -> mask tensor)
        criterion: Loss function for adaptation
        model_name: Name for logging
        num_steps: Number of adaptation steps on first sample
        lr: Learning rate for adaptation
        save_dir: Directory to save best adapted model (None = don't save)
    
    Returns:
        Dict with metrics including loss_before, loss_after, psnr, ssim
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train import FrozenMaskTrainer
    
    resolution = 320
    
    # Get first sample for adaptation
    data_iter = iter(data_loader)
    adapt_batch = next(data_iter)
    adapt_input, adapt_target, _, _, _ = adapt_batch
    adapt_input = adapt_input.to(device)
    adapt_target = adapt_target.to(device)
    
    # Prepare input shape (add channel dim if needed)
    adapt_input_model = adapt_input.unsqueeze(1)  # (B, 1, H, W, 2)
    
    # Reshape target to match output
    if adapt_target.dim() == 4:  # (B, C, H, W)
        adapt_target_reshaped = adapt_target.squeeze(1)  # (B, H, W)
    elif adapt_target.dim() == 3 and adapt_target.shape[0] != 1:
        adapt_target_reshaped = adapt_target.unsqueeze(0)  # Add batch dim
    else:
        adapt_target_reshaped = adapt_target
    
    # Compute loss BEFORE adaptation (with zeros in pruned positions)
    model.eval()
    with torch.no_grad():
        adapt_output = model(adapt_input_model)
        if adapt_output.dim() == 1:
            adapt_output = adapt_output.view(1, resolution, resolution)
        loss_with_zeros = criterion(adapt_output, adapt_target_reshaped).item()
    
    print(f"\n  [{model_name}] Single-shot adaptation using FrozenMaskTrainer...")
    print(f"  Loss with zeros (before reinit): {loss_with_zeros:.6f}")
    
    # Create FrozenMaskTrainer (handles Xavier reinit automatically)
    trainer = FrozenMaskTrainer(model, masks, device=device, reinit='xavier')
    
    # Compute loss AFTER Xavier reinit
    model.eval()
    with torch.no_grad():
        adapt_output = model(adapt_input_model)
        if adapt_output.dim() == 1:
            adapt_output = adapt_output.view(1, resolution, resolution)
        loss_after_reinit = criterion(adapt_output, adapt_target_reshaped).item()
    print(f"  Loss after Xavier reinit: {loss_after_reinit:.6f}")
    
    # Single-shot adaptation using trainer
    losses, _ = trainer.single_shot_adapt(
        adapt_input_model, adapt_target_reshaped, criterion, 
        num_steps=num_steps, lr=lr
    )
    
    loss_before = loss_after_reinit  # Use post-reinit as baseline
    loss_after = losses[-1] if losses else loss_before
    best_loss = min(losses) if losses else loss_before
    best_step = losses.index(best_loss) + 1 if losses else 0
    
    # Print adaptation summary
    print(f"\n  ════════════════════════════════════════════════════════")
    print(f"  SINGLE-SHOT ADAPTATION COMPLETE")
    print(f"  ════════════════════════════════════════════════════════")
    print(f"  Loss (with zeros):     {loss_with_zeros:.6f}")
    print(f"  Loss (after reinit):   {loss_after_reinit:.6f}")
    print(f"  Loss (after adapt):    {best_loss:.6f} (step {best_step}/{num_steps})")
    if loss_after_reinit > 0:
        improvement_pct = 100 * (loss_after_reinit - best_loss) / loss_after_reinit
        print(f"  Improvement:           {loss_after_reinit - best_loss:.6f} ({improvement_pct:.1f}%)")
    print(f"  ════════════════════════════════════════════════════════")
    print(f"  Now evaluating on remaining samples...\n")
    
    # Save adapted model if requested
    saved_paths = {}
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save full adapted model
        model_path = os.path.join(save_dir, f"{model_name}_singleshot_adapted.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_step': best_step,
            'best_loss': best_loss,
            'loss_with_zeros': loss_with_zeros,
            'loss_after_reinit': loss_after_reinit,
            'adaptation_steps': num_steps,
            'learning_rate': lr,
        }, model_path)
        saved_paths['model'] = model_path
        print(f"  💾 Saved adapted model to: {model_path}")
    
    # Evaluate on remaining samples
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    psnr_list = []
    ssim_list = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            input_data, target, mean, std, norm = batch
            input_data = input_data.to(device)
            target = target.to(device)
            
            output = model(input_data.unsqueeze(1))
            
            # Reshape for metric calculation
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
                print(f"  [{model_name}] Processed {batch_idx + 1}/{len(data_loader)-1}...")
    
    eval_time = time.time() - start_time
    
    return {
        'loss_with_zeros': loss_with_zeros,
        'loss_after_reinit': loss_after_reinit,
        'loss_before': loss_before,
        'loss_after': loss_after,
        'best_loss': best_loss,
        'best_step': best_step,
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'eval_time': eval_time,
        'num_samples': len(psnr_list),
        'adaptation_steps': num_steps,
        'single_shot': True,
        'saved_paths': saved_paths if saved_paths else None
    }


# Define the trained models from slurm logs
TRAINED_MODELS = OrderedDict([
    # ═══════════════════════════════════════════════════════════════════════════════
    # BASELINE MODELS
    # ═══════════════════════════════════════════════════════════════════════════════
    ('Unet', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_Unet_50',
        'description': 'Standard UNet baseline (32 base channels)',
        'params': '3.35M',
        'flops': '24.32 GFLOPs'
    }),
    ('LightUNet', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_LightUNet_50',
        'description': 'Baseline UNet with 20 base channels (no DCN, no FDConv)',
        'params': '1.31M',
        'flops': '9.54 GFLOPs'
    }),
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RigL MODELS (Dynamic Sparse Training)
    # ═══════════════════════════════════════════════════════════════════════════════
    ('RigL10', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL10_50',
        'description': 'RigL 10% sparsity',
        'params': '3.35M (90% dense)',
    }),
    ('RigL20', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL20_50',
        'description': 'RigL 20% sparsity',
        'params': '3.35M (80% dense)',
    }),
    ('RigL30', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL30_50',
        'description': 'RigL 30% sparsity',
        'params': '3.35M (70% dense)',
    }),
    ('RigL40', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL40_50',
        'description': 'RigL 40% sparsity',
        'params': '3.35M (60% dense)',
    }),
    ('RigL50', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL50_50',
        'description': 'RigL 50% sparsity',
        'params': '3.35M (50% dense)',
    }),
    ('RigL60', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL60_50',
        'description': 'RigL 60% sparsity',
        'params': '3.35M (40% dense)',
    }),
    ('RigL70', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL70_50',
        'description': 'RigL 70% sparsity',
        'params': '3.35M (30% dense)',
    }),
    ('RigL80', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL80_50',
        'description': 'RigL 80% sparsity',
        'params': '3.35M (20% dense)',
    }),
    ('RigL95', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL95_50',
        'description': 'RigL 95% sparsity (extreme)',
        'params': '3.35M (5% dense)',
    }),
    ('RigL99', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL99_50',
        'description': 'RigL 99% sparsity (ultra extreme)',
        'params': '3.35M (1% dense)',
    }),
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # STATIC SPARSE MODELS (Channel Reduction)
    # ═══════════════════════════════════════════════════════════════════════════════
    ('StaticSparseLight', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseLight_50',
        'description': 'Static channel reduction (~30% param reduction)',
    }),
    ('StaticSparseMedium', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseMedium_50',
        'description': 'Static channel reduction (~50% param reduction)',
    }),
    ('StaticSparseHeavy', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseHeavy_50',
        'description': 'Static channel reduction (~70% param reduction)',
    }),
    ('StaticSparse48Wide', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparse48Wide_50',
        'description': '48 base channels, slim bottleneck (~2.04M params)',
    }),
    ('StaticSparseUltraLight', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseUltraLight_50',
        'description': '32 base, extreme bottleneck compression (~0.5M)',
    }),
    ('StaticSparseFlat32', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseFlat32_50',
        'description': 'Uniform 32 channels throughout (~300K)',
    }),
    ('StaticSparse64Extreme', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparse64Extreme_50',
        'description': '64 base, extreme compression (~1.24M)',
    }),
    ('StaticSparse64ExtremeSlim', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparse64ExtremeSlim_50',
        'description': '64 base, slimmer decoder',
    }),
    ('StaticSparse64DecoderHeavy', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparse64DecoderHeavy_50',
        'description': '64 base, slim encoder, fuller decoder (RigL-inspired)',
    }),
    ('StaticSparseAsymmetric', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseAsymmetric_50',
        'description': 'RigL-learned asymmetric pattern',
    }),
    ('StaticSparseAsymmetricSlim', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseAsymmetricSlim_50',
        'description': 'Ultra-slim asymmetric (~400K)',
    }),
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RigL + StaticSparse (Double Sparse: Architecture + Weight Sparsity)
    # ═══════════════════════════════════════════════════════════════════════════════
    ('RigLStaticSparse48Wide30', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigLStaticSparse48Wide30_50',
        'description': 'StaticSparse48Wide + 30% RigL sparsity',
    }),
    ('RigLStaticSparse48Wide40', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigLStaticSparse48Wide40_50',
        'description': 'StaticSparse48Wide + 40% RigL sparsity',
    }),
    ('RigLStaticSparse48Wide50', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigLStaticSparse48Wide50_50',
        'description': 'StaticSparse48Wide + 50% RigL sparsity',
    }),
    ('RigLStaticSparse48Wide60', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigLStaticSparse48Wide60_50',
        'description': 'StaticSparse48Wide + 60% RigL sparsity',
    }),
    ('RigLStaticSparse64Extreme40', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigLStaticSparse64Extreme40_50',
        'description': 'StaticSparse64Extreme + 40% RigL sparsity',
    }),
    ('RigLStaticSparse64Extreme50', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigLStaticSparse64Extreme50_50',
        'description': 'StaticSparse64Extreme + 50% RigL sparsity',
    }),
    ('RigLStaticSparse64Extreme60', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigLStaticSparse64Extreme60_50',
        'description': 'StaticSparse64Extreme + 60% RigL sparsity',
    }),
    ('RigLStaticSparse64Extreme70', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigLStaticSparse64Extreme70_50',
        'description': 'StaticSparse64Extreme + 70% RigL sparsity',
    }),
])


def print_results_table(results, title="TEST SET RESULTS SUMMARY"):
    """Print results in a nicely formatted table."""
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)
    
    # Header
    print(f"{'Model':<28} | {'PSNR (dB)':<18} | {'SSIM':<18} | {'Non-zero Params':<16} | {'Sparsity':<10} | {'Time (s)':<8}")
    print("-" * 130)
    
    # Sort by PSNR
    sorted_results = sorted(results.items(), key=lambda x: x[1]['psnr_mean'], reverse=True)
    
    for model_name, metrics in sorted_results:
        psnr_str = f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}"
        ssim_str = f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"
        time_str = f"{metrics['eval_time']:.1f}"
        
        # Use actual measured parameters from mask_info
        nonzero = metrics.get('nonzero_params', 0)
        total = metrics.get('total_params', 0)
        sparsity = metrics.get('sparsity', 0)
        
        if nonzero > 0:
            params_str = f"{nonzero/1e6:.2f}M/{total/1e6:.2f}M"
            sparsity_str = f"{sparsity:.1f}%"
        else:
            params_str = "N/A"
            sparsity_str = "N/A"
        
        print(f"{model_name:<28} | {psnr_str:<18} | {ssim_str:<18} | {params_str:<16} | {sparsity_str:<10} | {time_str:<8}")
    
    print("=" * 130)
    
    # Print detailed comparison
    print("\n" + "=" * 130)
    print("DETAILED MODEL COMPARISON (sorted by PSNR)")
    print("=" * 130)
    
    best_psnr = sorted_results[0][1]['psnr_mean']
    
    for model_name, metrics in sorted_results:
        model_info = TRAINED_MODELS.get(model_name, {})
        psnr_diff = metrics['psnr_mean'] - best_psnr
        
        nonzero = metrics.get('nonzero_params', 0)
        total = metrics.get('total_params', 0)
        sparsity = metrics.get('sparsity', 0)
        has_rigl = metrics.get('has_rigl', False)
        
        print(f"\n{model_name}:")
        print(f"  Description: {model_info.get('description', 'N/A')}")
        print(f"  PSNR: {metrics['psnr_mean']:.2f} dB ({psnr_diff:+.2f} dB vs best)")
        print(f"  SSIM: {metrics['ssim_mean']:.4f}")
        if nonzero > 0:
            print(f"  Parameters: {nonzero:,} / {total:,} non-zero ({sparsity:.1f}% sparse)")
        print(f"  RigL model: {has_rigl}")
        print(f"  Samples evaluated: {metrics['num_samples']}")
    
    print("\n" + "=" * 130)


def print_domain_shift_comparison(fastmri_results, m4raw_results):
    """Print comparison between FastMRI and M4Raw results."""
    print("\n" + "=" * 120)
    print("DOMAIN SHIFT ANALYSIS: FastMRI vs M4Raw")
    print("=" * 120)
    
    # Header
    print(f"{'Model':<15} | {'FastMRI PSNR':<15} | {'M4Raw PSNR':<15} | {'Δ PSNR':<10} | {'FastMRI SSIM':<15} | {'M4Raw SSIM':<15} | {'Δ SSIM':<10}")
    print("-" * 120)
    
    for model_name in fastmri_results.keys():
        if model_name not in m4raw_results:
            continue
        
        fm = fastmri_results[model_name]
        m4 = m4raw_results[model_name]
        
        psnr_diff = m4['psnr_mean'] - fm['psnr_mean']
        ssim_diff = m4['ssim_mean'] - fm['ssim_mean']
        
        print(f"{model_name:<15} | {fm['psnr_mean']:.2f} ± {fm['psnr_std']:.2f}    | "
              f"{m4['psnr_mean']:.2f} ± {m4['psnr_std']:.2f}    | {psnr_diff:+.2f}      | "
              f"{fm['ssim_mean']:.4f} ± {fm['ssim_std']:.4f} | "
              f"{m4['ssim_mean']:.4f} ± {m4['ssim_std']:.4f} | {ssim_diff:+.4f}")
    
    print("=" * 120)
    
    # Summary
    print("\n📊 Domain Shift Summary:")
    avg_psnr_drop = np.mean([m4raw_results[m]['psnr_mean'] - fastmri_results[m]['psnr_mean'] 
                            for m in fastmri_results if m in m4raw_results])
    avg_ssim_drop = np.mean([m4raw_results[m]['ssim_mean'] - fastmri_results[m]['ssim_mean'] 
                            for m in fastmri_results if m in m4raw_results])
    
    print(f"  Average PSNR change: {avg_psnr_drop:+.2f} dB")
    print(f"  Average SSIM change: {avg_ssim_drop:+.4f}")
    
    # Find most robust model (smallest drop)
    psnr_drops = {m: m4raw_results[m]['psnr_mean'] - fastmri_results[m]['psnr_mean'] 
                  for m in fastmri_results if m in m4raw_results}
    most_robust = max(psnr_drops, key=psnr_drops.get)
    print(f"  Most robust model: {most_robust} (PSNR drop: {psnr_drops[most_robust]:+.2f} dB)")
    
    print("=" * 120)


def evaluate_on_dataset(models_to_test, data_loader, device, args, dataset_name="Test"):
    """Evaluate all models on a given dataset."""
    results = {}
    loaded_models = {}  # Cache loaded models for reuse
    
    for model_name in models_to_test:
        if model_name not in TRAINED_MODELS:
            print(f"\nWarning: Model '{model_name}' not found in trained models list. Skipping.")
            continue
        
        model_info = TRAINED_MODELS[model_name]
        checkpoint_dir = model_info['checkpoint']
        
        # Choose best_model.pt or model.pt
        model_file = 'best_model.pt' if args.use_best else 'model.pt'
        checkpoint_path = pathlib.Path(checkpoint_dir) / model_file
        
        if not checkpoint_path.exists():
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}. Skipping {model_name}.")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Testing: {model_name} on {dataset_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'=' * 80}")
        
        try:
            # Load model (don't use cache for single-shot as we modify the model)
            use_single_shot = getattr(args, 'single_shot', False) and model_name.startswith('RigL')
            
            if model_name in loaded_models and not use_single_shot:
                model, mask_info, rigl_masks = loaded_models[model_name]
            else:
                print("Loading model...")
                model, saved_args, mask_info = load_model(checkpoint_path, device, apply_rigl_masks=True)
                
                # Load RigL masks for single-shot adaptation
                rigl_masks = {}
                if mask_info['has_rigl']:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    raw_masks = checkpoint.get('rigl_scheduler', {}).get('masks', {})
                    
                    # Determine if model is wrapped in DataParallel
                    is_dataparallel = any(k.startswith('module.') for k in model.state_dict().keys())
                    
                    for k, v in raw_masks.items():
                        # Match mask keys to model parameter naming
                        if is_dataparallel and not k.startswith('module.'):
                            new_key = 'module.' + k
                        elif not is_dataparallel and k.startswith('module.'):
                            new_key = k[7:]
                        else:
                            new_key = k
                        rigl_masks[new_key] = v
                
                if not use_single_shot:
                    loaded_models[model_name] = (model, mask_info, rigl_masks)
                
                # Print mask/sparsity info
                if mask_info['has_rigl']:
                    print(f"  RigL model detected - masks applied: {mask_info['masks_applied']}")
                print(f"  Parameters: {mask_info['nonzero_params']:,} / {mask_info['total_params']:,} non-zero")
                print(f"  Sparsity: {mask_info['sparsity']:.1f}%")
            
            # Evaluate (single-shot or standard)
            if use_single_shot and rigl_masks:
                print(f"Single-shot adaptation on {dataset_name} set...")
                print(f"  Steps: {args.single_shot_steps}, LR: {args.single_shot_lr}")
                metrics = single_shot_evaluate(
                    model, data_loader, device, rigl_masks,
                    criterion=torch.nn.L1Loss(),
                    model_name=model_name,
                    num_steps=args.single_shot_steps,
                    lr=args.single_shot_lr,
                    save_dir=getattr(args, 'single_shot_save_dir', None)
                )
                if 'loss_before' in metrics:
                    print(f"  Loss: {metrics['loss_before']:.6f} → {metrics['loss_after']:.6f} (best: {metrics['best_loss']:.6f} @ step {metrics['best_step']})")
            else:
                print(f"Evaluating on {dataset_name} set...")
                metrics = evaluate_model(model, data_loader, device, model_name)
            
            # Add mask info to metrics
            metrics['total_params'] = mask_info['total_params']
            metrics['nonzero_params'] = mask_info['nonzero_params']
            metrics['sparsity'] = mask_info['sparsity']
            metrics['has_rigl'] = mask_info['has_rigl']
            
            results[model_name] = metrics
            
            print(f"\nResults for {model_name} on {dataset_name}:")
            print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
            print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
            print(f"  Evaluation time: {metrics['eval_time']:.1f} seconds")
            if metrics.get('single_shot'):
                print(f"  Single-shot: YES ({metrics['adaptation_steps']} steps)")
            
        except Exception as e:
            print(f"\nError testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Clean up all loaded models
    for model, _, _ in loaded_models.values():
        del model
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test trained MRI reconstruction models')
    parser.add_argument('--data-path', type=pathlib.Path, 
                        default=pathlib.Path('../data'),
                        help='Path to the dataset directory')
    parser.add_argument('--use-best', action='store_true', default=True,
                        help='Use best_model.pt instead of model.pt')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loader workers')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to test (default: all)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Save results to file')
    # M4Raw domain shift options
    parser.add_argument('--m4raw', action='store_true', default=False,
                        help='Include M4Raw domain shift evaluation')
    parser.add_argument('--m4raw-only', action='store_true', default=False,
                        help='Only evaluate on M4Raw (skip FastMRI test)')
    parser.add_argument('--m4raw-acquisition', type=str, default=None,
                        choices=['T1', 'T2', 'FLAIR', None],
                        help='Filter M4Raw by acquisition type')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples for M4Raw evaluation (default: all)')
    # Single-shot adaptation options
    parser.add_argument('--single-shot', action='store_true', default=False,
                        help='Enable single-shot adaptation for RigL models')
    parser.add_argument('--single-shot-steps', type=int, default=20,
                        help='Number of adaptation steps for single-shot (default: 20)')
    parser.add_argument('--single-shot-lr', type=float, default=1e-3,
                        help='Learning rate for single-shot adaptation (default: 1e-3)')
    parser.add_argument('--single-shot-save-dir', type=str, default=None,
                        help='Directory to save best adapted models and masked params (default: None)')
    # Model filtering options
    parser.add_argument('--rigl-only', action='store_true', default=False,
                        help='Only evaluate RigL models (models starting with "RigL")')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Determine which models to test
    if args.models:
        models_to_test = args.models
    else:
        models_to_test = list(TRAINED_MODELS.keys())
    
    # Filter to RigL models only if requested
    if args.rigl_only:
        models_to_test = [m for m in models_to_test if m.startswith('RigL')]
        print(f"\n🔍 RigL-only mode: filtering to {len(models_to_test)} RigL models")
    
    print(f"\nModels to evaluate ({len(models_to_test)}): {', '.join(models_to_test)}")
    
    if args.single_shot:
        print(f"📊 Single-shot adaptation ENABLED (steps={args.single_shot_steps}, lr={args.single_shot_lr})")
        print(f"   Will adapt pruned weights on first sample, evaluate on rest")
        if args.single_shot_save_dir:
            print(f"   💾 Saving best adapted models to: {args.single_shot_save_dir}")
    
    fastmri_results = {}
    m4raw_results = {}
    
    # ==================== FastMRI Test Set Evaluation ====================
    if not args.m4raw_only:
        print("\n" + "=" * 80)
        print("FASTMRI TEST SET EVALUATION")
        print("=" * 80)
        
        print(f"\nLoading test data from: {args.data_path / 'multicoil_test'}")
        test_loader, num_samples = create_test_loader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print(f"Total test samples: {num_samples}")
        
        fastmri_results = evaluate_on_dataset(
            models_to_test, test_loader, device, args, "FastMRI Test"
        )
        
        if fastmri_results:
            print_results_table(fastmri_results, "FASTMRI TEST SET RESULTS")
    
    # ==================== M4Raw Domain Shift Evaluation ====================
    if args.m4raw or args.m4raw_only:
        print("\n" + "=" * 80)
        print("M4RAW DOMAIN SHIFT EVALUATION")
        if args.m4raw_acquisition:
            print(f"Acquisition filter: {args.m4raw_acquisition}")
        print("=" * 80)
        
        m4raw_loader, m4raw_samples = create_m4raw_loader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            acquisition_filter=args.m4raw_acquisition,
            max_samples=args.max_samples
        )
        
        if m4raw_loader is not None and m4raw_samples > 0:
            print(f"Total M4Raw samples: {m4raw_samples}")
            
            m4raw_results = evaluate_on_dataset(
                models_to_test, m4raw_loader, device, args, "M4Raw"
            )
            
            if m4raw_results:
                title = "M4RAW DOMAIN SHIFT RESULTS"
                if args.m4raw_acquisition:
                    title += f" ({args.m4raw_acquisition})"
                print_results_table(m4raw_results, title)
        else:
            print("Warning: M4Raw dataset not available or empty.")
    
    # ==================== Domain Shift Comparison ====================
    if fastmri_results and m4raw_results:
        print_domain_shift_comparison(fastmri_results, m4raw_results)
    
    # ==================== Save Results ====================
    all_results = {}
    if fastmri_results:
        all_results['fastmri'] = fastmri_results
    if m4raw_results:
        all_results['m4raw'] = m4raw_results
    
    if args.output_file and all_results:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")
    
    if not fastmri_results and not m4raw_results:
        print("\nNo models were successfully evaluated.")


if __name__ == '__main__':
    main()

