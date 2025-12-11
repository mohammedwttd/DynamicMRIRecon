#!/usr/bin/env python3
"""
Test script to evaluate RigL models (10-80), LightUNet, Unet, and StaticSparse models on FastMRI test set.
All models are 50-epoch models using best_model.pt checkpoint.

Usage:
    python tests/test_rigl_lightunet_unet_30epochs.py --data-path ../data --device cuda
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


def create_test_loader(data_path, resolution=320, batch_size=1, num_workers=8, max_samples=None):
    """Create test data loader."""
    # Load full dataset first
    test_data = SliceData(
        root=data_path / 'multicoil_test',
        transform=DataTransform(resolution),
        sample_rate=1
    )
    
    # If max_samples specified and we have more, limit it using Subset
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
    M4Raw dataset for domain shift evaluation.
    
    M4Raw has different data format:
    - Resolution: 256x256 (will be padded to 320x320)
    - kspace shape: (slices, coils, H, W)
    - Different acquisition types: T1, T2, FLAIR
    """
    def __init__(self, root, transform, sample_rate=1, acquisition_filter=None, max_samples=None):
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
        reached_limit = False
        for fname in sorted(files):
            if reached_limit:
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
                            _ = data['kspace'][slice_idx]
                            _ = data['reconstruction_rss'][slice_idx]
                            self.examples.append((fname, slice_idx))
                            if max_samples is not None and len(self.examples) >= max_samples:
                                reached_limit = True
                                break
                        except OSError:
                            skipped += 1
                            continue
            except Exception as e:
                print(f"Warning: Could not read {fname}: {e}")
                continue
        
        if skipped > 0:
            print(f"Skipped {skipped} corrupted slices")
        if max_samples is not None and len(self.examples) >= max_samples:
            print(f"Reached max_samples limit: {max_samples}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        fname, slice_idx = self.examples[i]
        
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice_idx]  # Shape: (coils, H, W)
            target = data['reconstruction_rss'][slice_idx]  # Shape: (H, W)
            
            attrs = {
                'norm': data.attrs.get('max', 1.0),
                'acquisition': data.attrs.get('acquisition', 'unknown')
            }
            
            return self.transform(kspace, target, attrs, fname.name, slice_idx)


class M4RawTransform:
    """
    Transform for M4Raw data.
    M4Raw has 256x256 resolution, we use k-space zero-padding to upsample to 320x320.
    Zero-padding in k-space = sinc interpolation in image domain (no blurring).
    Returns complex format (H, W, 2) like DataTransform for model compatibility.
    """
    def __init__(self, target_resolution=320):
        self.target_resolution = target_resolution
    
    def __call__(self, kspace, target, attrs, fname, slice_idx):
        # kspace shape: (coils, H, W) - M4Raw format (complex)
        kspace = transforms.to_tensor(kspace)  # (coils, H, W, 2)
        
        orig_h, orig_w = kspace.shape[1], kspace.shape[2]
        
        # Zero-pad k-space to target resolution for upsampling
        # Zero-padding in frequency domain = sinc interpolation in image domain
        if orig_h < self.target_resolution or orig_w < self.target_resolution:
            pad_h = (self.target_resolution - orig_h) // 2
            pad_w = (self.target_resolution - orig_w) // 2
            
            # Pad k-space symmetrically
            kspace_padded = torch.zeros(
                kspace.shape[0], self.target_resolution, self.target_resolution, 2,
                dtype=kspace.dtype
            )
            kspace_padded[:, pad_h:pad_h+orig_h, pad_w:pad_w+orig_w, :] = kspace
            kspace = kspace_padded
        
        # Compute image from (padded) k-space
        image = transforms.ifft2_regular(kspace)  # (coils, H, W, 2)
        
        # Center crop to target resolution (in case of any size mismatch)
        image = transforms.complex_center_crop(image, (self.target_resolution, self.target_resolution))
        
        # RSS combination (keeps complex dim)
        rss_image = fastmri.rss(image)  # (H, W, 2) complex
        
        # Process target - use bilinear for target since it's already in image domain
        target = torch.tensor(target).float()
        
        if target.shape[0] != self.target_resolution or target.shape[1] != self.target_resolution:
            target = target.unsqueeze(0).unsqueeze(0)
            target = F.interpolate(target, size=(self.target_resolution, self.target_resolution), 
                                   mode='bilinear', align_corners=False)
            target = target.squeeze(0)
        else:
            target = target.unsqueeze(0)
        
        target = normalize(target)
        
        # Return complex image (H, W, 2) format for model compatibility
        # The model's subsampling layer will convert to magnitude internally
        mean = std = 0
        norm = np.float32(attrs.get('norm', 1.0)) if isinstance(attrs.get('norm', 1.0), (int, float)) else np.float32(1.0)
        
        return rss_image, target, mean, std, norm  # (H, W, 2) complex


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


# ==================== FastMRI Knee Dataset Support ====================

class KneeSliceData(Dataset):
    """
    FastMRI Knee dataset for evaluation.
    
    Knee data has different resolution (typically 320x320 or 640x368).
    """
    def __init__(self, root, transform, sample_rate=1, max_samples=None):
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
        
        reached_limit = False
        for fname in sorted(files):
            if reached_limit:
                break
            try:
                with h5py.File(fname, 'r') as data:
                    kspace = data['kspace']
                    num_slices = kspace.shape[0]
                    
                    for slice_idx in range(num_slices):
                        self.examples.append((fname, slice_idx))
                        if max_samples is not None and len(self.examples) >= max_samples:
                            reached_limit = True
                            break
            except Exception as e:
                print(f"Warning: Could not read {fname}: {e}")
                continue
        
        if max_samples is not None and len(self.examples) >= max_samples:
            print(f"Reached max_samples limit: {max_samples}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        fname, slice_idx = self.examples[i]
        
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice_idx]
            
            # Try different target field names (varies by dataset)
            target = None
            for target_key in ['reconstruction_rss', 'reconstruction_esc', 'target', 'image']:
                if target_key in data:
                    target = data[target_key][slice_idx]
                    break
            
            # If no target found, compute RSS from kspace
            if target is None:
                kspace_tensor = transforms.to_tensor(kspace)
                image = transforms.ifft2_regular(kspace_tensor)
                image_mag = torch.sqrt(image[..., 0]**2 + image[..., 1]**2)
                target = torch.sqrt(torch.sum(image_mag**2, dim=0)).numpy()
            
            attrs = {
                'norm': data.attrs.get('max', 1.0),
            }
            
            return self.transform(kspace, target, attrs, fname.name, slice_idx)


class KneeTransform:
    """
    Transform for FastMRI Knee data.
    Uses center crop in complex domain (like train.py) to avoid blurring.
    """
    def __init__(self, target_resolution=320):
        self.target_resolution = target_resolution
    
    def __call__(self, kspace, target, attrs, fname, slice_idx):
        kspace = transforms.to_tensor(kspace)  # (coils, H, W, 2)
        
        # Compute image from k-space
        image = transforms.ifft2_regular(kspace)  # (coils, H, W, 2)
        
        # Center crop in complex domain (preserves high-frequency details, no blurring)
        image = transforms.complex_center_crop(image, (self.target_resolution, self.target_resolution))
        
        # RSS combination (same as fastmri.rss)
        rss_image = fastmri.rss(image)  # (H, W)
        
        # Process target with center crop (not bilinear interpolation!)
        target = transforms.to_tensor(target)
        if target.shape[-2] != self.target_resolution or target.shape[-1] != self.target_resolution:
            target = transforms.center_crop(target, (self.target_resolution, self.target_resolution))
        
        target = normalize(target)
        rss_image = normalize(rss_image.unsqueeze(0))  # (1, H, W)
        
        # Return magnitude image directly (H, W) format for model compatibility
        # The evaluate function will add batch and channel dims
        mean = std = 0
        norm = np.float32(attrs.get('norm', 1.0)) if isinstance(attrs.get('norm', 1.0), (int, float)) else np.float32(1.0)
        
        return rss_image.squeeze(0), target, mean, std, norm  # (H, W)


def create_knee_loader(data_path, resolution=320, batch_size=1, num_workers=8, max_samples=None):
    """Create FastMRI Knee data loader for domain shift evaluation."""
    # Try different possible paths for knee data
    possible_paths = [
        data_path / 'knee' / 'multicoil_test',
        data_path / 'knee' / 'multicoil_val',
        data_path / 'knee_multicoil_val',
        data_path / 'knee_multicoil_test',
        data_path / 'singlecoil_knee_val',
        data_path / 'knee' / 'singlecoil_val',
    ]
    
    knee_path = None
    for path in possible_paths:
        if path.exists():
            knee_path = path
            break
    
    if knee_path is None:
        print(f"Warning: Knee dataset not found. Tried: {[str(p) for p in possible_paths]}")
        return None, 0
    
    print(f"Found knee dataset at: {knee_path}")
    
    knee_data = KneeSliceData(
        root=knee_path,
        transform=KneeTransform(target_resolution=resolution),
        sample_rate=1,
        max_samples=max_samples
    )
    
    knee_loader = DataLoader(
        dataset=knee_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    return knee_loader, len(knee_data)


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
            # Mask keys don't have '.weight' suffix, so we need to check both
            # e.g., mask key: "module.reconstruction_model.down_sample_layers.0.layers.0"
            #       param name: "module.reconstruction_model.down_sample_layers.0.layers.0.weight"
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
    
    # Handle DataParallel state dict
    state_dict = checkpoint['model']
    
    # Filter out thop profiling keys
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if 'total_ops' not in k and 'total_params' not in k
    }
    
    # Check if model was saved with DataParallel
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


def save_recon_image(recons, target, save_path, batch_idx, model_name, psnr_val, ssim_val):
    """Save reconstruction comparison image."""
    import matplotlib.pyplot as plt
    
    # Get first image from batch
    recon_img = recons[0].numpy() if len(recons.shape) > 2 else recons.numpy()
    target_img = target[0].numpy() if len(target.shape) > 2 else target.numpy()
    
    # Normalize for display
    recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
    target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
    diff_img = np.abs(recon_img - target_img)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(target_img, cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(recon_img, cmap='gray')
    axes[1].set_title(f'Reconstruction\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}')
    axes[1].axis('off')
    
    axes[2].imshow(diff_img, cmap='hot')
    axes[2].set_title('Error (|Recon - GT|)')
    axes[2].axis('off')
    
    plt.suptitle(f'{model_name} - Sample {batch_idx}')
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{model_name}_sample_{batch_idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model, data_loader, device, model_name="Model", save_images=False, save_path=None, dataset_name=""):
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
            
            # Save image every 100 samples
            if save_images and save_path and (batch_idx % 100 == 0):
                img_save_path = os.path.join(save_path, dataset_name) if dataset_name else save_path
                save_recon_image(recons, target_cpu, img_save_path, batch_idx, model_name, psnr_val, ssim_val)
            
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


def single_shot_evaluate(model, data_loader, device, masks, criterion, model_name="Model",
                         num_steps=20, lr=1e-3, save_images=False, save_path=None, dataset_name=""):
    """
    Single-shot learning evaluation: Adapt model on first example, then evaluate.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device
        masks: RigL masks dictionary
        criterion: Loss function
        model_name: Name for logging
        num_steps: Number of gradient steps on single example
        lr: Learning rate for adaptation
        save_images: Whether to save reconstruction images
        save_path: Directory to save images
        dataset_name: Name of dataset for organizing saved images
    
    Returns:
        dict with metrics before and after adaptation
    """
    from train import FrozenMaskTrainer
    
    # Create trainer with Xavier reinit
    trainer = FrozenMaskTrainer(model, masks, device=device, reinit='xavier')
    
    # Get first example for adaptation
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    
    input_data, target, mean, std, norm = first_batch
    input_data = input_data.to(device)
    target = target.to(device)
    
    # Add channel dim if needed
    if len(input_data.shape) == 4:  # (B, H, W, 2) complex
        input_data = input_data.unsqueeze(1)  # (B, 1, H, W, 2)
    
    print(f"\n  [{model_name}] Single-shot adaptation on 1 example with {num_steps} steps...")
    
    # Evaluate BEFORE adaptation
    model.eval()
    with torch.no_grad():
        output_before = model(input_data)
        loss_before = criterion(output_before, target).item()
    
    # Adapt using single-shot learning
    losses, _ = trainer.single_shot_adapt(input_data, target, criterion, num_steps=num_steps, lr=lr)
    
    # Evaluate AFTER adaptation on the same example
    model.eval()
    with torch.no_grad():
        output_after = model(input_data)
        loss_after = criterion(output_after, target).item()
    
    # Now evaluate on rest of the dataset
    print(f"  [{model_name}] Evaluating on full dataset after adaptation...")
    
    psnr_list = []
    ssim_list = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            inp, tgt, _, _, _ = data
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            if len(inp.shape) == 4:
                inp = inp.unsqueeze(1)
            
            output = model(inp)
            
            resolution = tgt.shape[-1]
            recons = output.cpu().squeeze(1)
            tgt_cpu = tgt.cpu()
            
            # Handle shape
            if len(recons.shape) > 2:
                recons = recons.view(-1, resolution, resolution)
            if len(tgt_cpu.shape) > 2:
                tgt_cpu = tgt_cpu.view(-1, resolution, resolution)
            
            psnr_val = psnr(tgt_cpu.numpy(), recons.numpy())
            ssim_val = ssim(tgt_cpu.numpy(), recons.numpy())
            
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            
            # Save image every 100 samples
            if save_images and save_path and (batch_idx % 100 == 0):
                img_save_path = os.path.join(save_path, f"{dataset_name}_singleshot") if dataset_name else save_path
                save_recon_image(recons, tgt_cpu, img_save_path, batch_idx, model_name, psnr_val, ssim_val)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"    Processed {batch_idx + 1}/{len(data_loader)}...")
    
    return {
        'loss_before': loss_before,
        'loss_after': loss_after,
        'adaptation_losses': losses,
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'num_samples': len(psnr_list),
        'num_steps': num_steps,
    }


# Define models to test - all 50-epoch models with best_model.pt
MODELS_TO_TEST = OrderedDict([
    ('RigL10', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL10_50',
        'description': 'UNet with RigL at 10% sparsity',
    }),
    ('RigL20', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL20_50',
        'description': 'UNet with RigL at 20% sparsity',
    }),
    ('RigL30', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL30_50',
        'description': 'UNet with RigL at 30% sparsity',
    }),
    ('RigL40', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL40_50',
        'description': 'UNet with RigL at 40% sparsity',
    }),
    ('RigL50', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL50_50',
        'description': 'UNet with RigL at 50% sparsity',
    }),
    ('RigL60', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL60_50',
        'description': 'UNet with RigL at 60% sparsity',
    }),
    ('RigL70', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL70_50',
        'description': 'UNet with RigL at 70% sparsity',
    }),
    ('RigL80', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL80_50',
        'description': 'UNet with RigL at 80% sparsity',
    }),
    ('RigL90', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL90_50',
        'description': 'UNet with RigL at 90% sparsity',
    }),
    ('RigL95', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL95_50',
        'description': 'UNet with RigL at 95% sparsity',
    }),
    ('RigL99', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL99_50',
        'description': 'UNet with RigL at 99% sparsity',
    }),
    ('LightUNet', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_LightUNet_50',
        'description': 'Light UNet with 20 base channels',
    }),
    ('Unet', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_Unet_50',
        'description': 'Standard UNet with 32 base channels',
    }),
    ('StaticSparseLight', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseLight_50',
        'description': 'UNet with static sparse masks at 50% sparsity (light)',
    }),
    ('StaticSparseMedium', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseMedium_50',
        'description': 'UNet with static sparse masks at 70% sparsity (medium)',
    }),
    ('StaticSparseHeavy', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseHeavy_50',
        'description': 'UNet with static sparse masks at 90% sparsity (heavy)',
    }),
    ('StaticSparse48Wide', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparse48Wide_50',
        'description': 'UNet with 48 base channels, 70% sparsity',
    }),
    ('StaticSparse64Extreme', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparse64Extreme_50',
        'description': 'UNet with 64 base channels, 85% sparsity',
    }),
    ('StaticSparseFlat32', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseFlat32_50',
        'description': 'UNet with 32 channels, 3 pools, 60% sparsity',
    }),
    ('StaticSparseUltraLight', {
        'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_StaticSparseUltraLight_50',
        'description': 'UNet with 16 channels, 3 pools, 40% sparsity',
    }),
])


def print_results_table(results, title="TEST SET RESULTS"):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'Model':<18} | {'PSNR (dB)':<20} | {'SSIM':<20} | {'Non-zero Params':<18} | {'Sparsity':<10}")
    print("-" * 100)
    
    for model_name in MODELS_TO_TEST.keys():
        if model_name not in results:
            continue
        metrics = results[model_name]
        psnr_str = f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}"
        ssim_str = f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"
        params_str = f"{metrics['nonzero_params']/1e6:.2f}M"
        sparsity_str = f"{metrics['sparsity']:.1f}%"
        print(f"{model_name:<18} | {psnr_str:<20} | {ssim_str:<20} | {params_str:<18} | {sparsity_str:<10}")
    
    print("=" * 100)


def print_domain_shift_comparison(fastmri_results, m4raw_results):
    """Print comparison between FastMRI and M4Raw results."""
    print("\n" + "=" * 130)
    print("DOMAIN SHIFT ANALYSIS: FastMRI Brain vs M4Raw")
    print("=" * 130)
    print(f"{'Model':<12} | {'FastMRI PSNR':<18} | {'M4Raw PSNR':<18} | {'Δ PSNR':<10} | {'FastMRI SSIM':<18} | {'M4Raw SSIM':<18} | {'Δ SSIM':<10}")
    print("-" * 130)
    
    for model_name in MODELS_TO_TEST.keys():
        if model_name not in fastmri_results or model_name not in m4raw_results:
            continue
        
        fm = fastmri_results[model_name]
        m4 = m4raw_results[model_name]
        
        psnr_diff = m4['psnr_mean'] - fm['psnr_mean']
        ssim_diff = m4['ssim_mean'] - fm['ssim_mean']
        
        fm_psnr = f"{fm['psnr_mean']:.2f} ± {fm['psnr_std']:.2f}"
        m4_psnr = f"{m4['psnr_mean']:.2f} ± {m4['psnr_std']:.2f}"
        fm_ssim = f"{fm['ssim_mean']:.4f} ± {fm['ssim_std']:.4f}"
        m4_ssim = f"{m4['ssim_mean']:.4f} ± {m4['ssim_std']:.4f}"
        
        print(f"{model_name:<12} | {fm_psnr:<18} | {m4_psnr:<18} | {psnr_diff:+.2f}      | {fm_ssim:<18} | {m4_ssim:<18} | {ssim_diff:+.4f}")
    
    print("=" * 130)


def print_domain_shift_comparison_knee(fastmri_results, knee_results):
    """Print comparison between FastMRI Brain and Knee results."""
    print("\n" + "=" * 130)
    print("DOMAIN SHIFT ANALYSIS: FastMRI Brain vs FastMRI Knee")
    print("=" * 130)
    print(f"{'Model':<12} | {'Brain PSNR':<18} | {'Knee PSNR':<18} | {'Δ PSNR':<10} | {'Brain SSIM':<18} | {'Knee SSIM':<18} | {'Δ SSIM':<10}")
    print("-" * 130)
    
    for model_name in MODELS_TO_TEST.keys():
        if model_name not in fastmri_results or model_name not in knee_results:
            continue
        
        fm = fastmri_results[model_name]
        kn = knee_results[model_name]
        
        psnr_diff = kn['psnr_mean'] - fm['psnr_mean']
        ssim_diff = kn['ssim_mean'] - fm['ssim_mean']
        
        fm_psnr = f"{fm['psnr_mean']:.2f} ± {fm['psnr_std']:.2f}"
        kn_psnr = f"{kn['psnr_mean']:.2f} ± {kn['psnr_std']:.2f}"
        fm_ssim = f"{fm['ssim_mean']:.4f} ± {fm['ssim_std']:.4f}"
        kn_ssim = f"{kn['ssim_mean']:.4f} ± {kn['ssim_std']:.4f}"
        
        print(f"{model_name:<12} | {fm_psnr:<18} | {kn_psnr:<18} | {psnr_diff:+.2f}      | {fm_ssim:<18} | {kn_ssim:<18} | {ssim_diff:+.4f}")
    
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(description='Test RigL (10-80), LightUNet, and Unet models')
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
    # Dataset selection options
    parser.add_argument('--skip-brain', action='store_true', default=False,
                        help='Skip FastMRI brain test evaluation')
    parser.add_argument('--brain-max-samples', type=int, default=-1,
                        help='Maximum number of FastMRI brain samples to evaluate (default: -1 for all)')
    # M4Raw options
    parser.add_argument('--m4raw', action='store_true', default=True,
                        help='Include M4Raw domain shift evaluation')
    parser.add_argument('--m4raw-only', action='store_true', default=False,
                        help='Only evaluate on M4Raw (skip FastMRI and Knee)')
    parser.add_argument('--m4raw-max-samples', type=int, default=-1,
                        help='Maximum number of M4Raw samples to evaluate (default: -1 for all)')
    # Knee options
    parser.add_argument('--knee', action='store_true', default=True,
                        help='Include FastMRI Knee domain shift evaluation')
    parser.add_argument('--knee-only', action='store_true', default=False,
                        help='Only evaluate on Knee (skip FastMRI brain and M4Raw)')
    parser.add_argument('--knee-max-samples', type=int, default=-1,
                        help='Maximum number of Knee samples to evaluate (default: -1 for all)')
    # Single-shot learning options
    parser.add_argument('--single-shot', action='store_true', default=False,
                        help='Enable single-shot adaptation (adapt on 1 example, evaluate on rest)')
    parser.add_argument('--single-shot-steps', type=int, default=20,
                        help='Number of gradient steps for single-shot adaptation (default: 20)')
    parser.add_argument('--single-shot-lr', type=float, default=1e-4,
                        help='Learning rate for single-shot adaptation (default: 1e-4)')
    # Image saving options
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='Save reconstruction images (1 per 100 samples)')
    parser.add_argument('--save-path', type=str, default='recon_images',
                        help='Directory to save reconstruction images (default: recon_images)')
    # Model selection
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='List of specific models to test (default: all models). '
                             'Example: --models StaticSparse48Wide StaticSparse64Extreme')
    
    args = parser.parse_args()
    
    # Filter MODELS_TO_TEST if specific models are requested
    global MODELS_TO_TEST
    if args.models:
        filtered_models = OrderedDict()
        for model_name in args.models:
            if model_name in MODELS_TO_TEST:
                filtered_models[model_name] = MODELS_TO_TEST[model_name]
            else:
                print(f"Warning: Model '{model_name}' not found in MODELS_TO_TEST. Skipping.")
        if filtered_models:
            MODELS_TO_TEST = filtered_models
            print(f"Testing only specified models: {list(MODELS_TO_TEST.keys())}")
        else:
            print("Error: No valid models specified. Using all models.")
    
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 80}")
    print("RIGL (10-80), LIGHTUNET, UNET, AND STATICSPARSE EVALUATION (50 EPOCHS)")
    print(f"{'=' * 80}")
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    fastmri_results = {}
    m4raw_results = {}
    knee_results = {}
    
    # ==================== FastMRI Test Set Evaluation ====================
    if not args.m4raw_only and not args.knee_only and not args.skip_brain:
        print(f"\n{'=' * 80}")
        print("FASTMRI BRAIN TEST SET EVALUATION")
        print(f"{'=' * 80}")
        
        max_samples = args.brain_max_samples if args.brain_max_samples > 0 else None
        print(f"\nLoading FastMRI test data from: {args.data_path / 'multicoil_test'}")
        test_loader, num_samples = create_test_loader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=max_samples
        )
        print(f"Total test samples: {num_samples}")
        
        for model_name, model_info in MODELS_TO_TEST.items():
            checkpoint_dir = model_info['checkpoint']
            checkpoint_path = pathlib.Path(checkpoint_dir) / 'best_model.pt'
            
            if not checkpoint_path.exists():
                print(f"\nWarning: Checkpoint not found: {checkpoint_path}. Skipping {model_name}.")
                continue
            
            print(f"\n{'=' * 80}")
            print(f"Testing: {model_name} on FastMRI Brain")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"{'=' * 80}")
            
            try:
                print("Loading model...")
                model, saved_args, masks_applied = load_model(checkpoint_path, device, apply_masks=True)
                
                total_params, nonzero_params = count_nonzero_params(model)
                sparsity = 100 * (1 - nonzero_params / total_params) if total_params > 0 else 0
                print(f"  Parameters after masking: {nonzero_params:,}/{total_params:,} non-zero ({sparsity:.2f}% sparse)")
                
                # Single-shot learning: adapt on first example, evaluate on rest
                if args.single_shot and masks_applied > 0:
                    print(f"Single-shot adaptation on FastMRI Brain...")
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    masks = checkpoint.get('rigl_scheduler', {}).get('masks', {})
                    
                    if masks:
                        metrics = single_shot_evaluate(
                            model, test_loader, device, masks,
                            criterion=torch.nn.L1Loss(),
                            model_name=model_name,
                            num_steps=args.single_shot_steps,
                            lr=args.single_shot_lr,
                            save_images=args.save_images, save_path=args.save_path, dataset_name="brain"
                        )
                        print(f"\n  Loss before adapt: {metrics['loss_before']:.6f}")
                        print(f"  Loss after adapt:  {metrics['loss_after']:.6f}")
                    else:
                        print("  No masks found, falling back to standard eval")
                        metrics = evaluate_model(model, test_loader, device, model_name,
                                                 save_images=args.save_images, save_path=args.save_path, dataset_name="brain")
                else:
                    print(f"Evaluating on FastMRI test set...")
                    metrics = evaluate_model(model, test_loader, device, model_name,
                                             save_images=args.save_images, save_path=args.save_path, dataset_name="brain")
                
                metrics['total_params'] = total_params
                metrics['nonzero_params'] = nonzero_params
                metrics['sparsity'] = sparsity
                metrics['masks_applied'] = masks_applied
                fastmri_results[model_name] = metrics
                
                print(f"\nResults for {model_name} on FastMRI Brain:")
                print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
                print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
                
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if fastmri_results:
            print_results_table(fastmri_results, "FASTMRI BRAIN RESULTS (50 EPOCHS, BEST MODEL)")
    
    # ==================== M4Raw Evaluation ====================
    if (args.m4raw or args.m4raw_only) and not args.knee_only:
        print(f"\n{'=' * 80}")
        print("M4RAW DOMAIN SHIFT EVALUATION")
        print(f"{'=' * 80}")
        
        # -1 means no limit
        max_samples = args.m4raw_max_samples if args.m4raw_max_samples > 0 else None
        m4raw_loader, m4raw_samples = create_m4raw_loader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=max_samples
        )
        
        if m4raw_loader is not None and m4raw_samples > 0:
            print(f"Total M4Raw samples: {m4raw_samples}")
            
            for model_name, model_info in MODELS_TO_TEST.items():
                checkpoint_dir = model_info['checkpoint']
                checkpoint_path = pathlib.Path(checkpoint_dir) / 'best_model.pt'
                
                if not checkpoint_path.exists():
                    continue
                
                print(f"\n{'=' * 80}")
                print(f"Testing: {model_name} on M4Raw")
                print(f"{'=' * 80}")
                
                try:
                    print("Loading model...")
                    model, saved_args, masks_applied = load_model(checkpoint_path, device, apply_masks=True)
                    
                    total_params, nonzero_params = count_nonzero_params(model)
                    sparsity = 100 * (1 - nonzero_params / total_params) if total_params > 0 else 0
                    
                    # Single-shot learning: adapt on first example, evaluate on rest
                    if args.single_shot and masks_applied > 0:
                        print(f"Single-shot adaptation on M4Raw...")
                        # Get masks from checkpoint
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        masks = checkpoint.get('rigl_scheduler', {}).get('masks', {})
                        
                        if masks:
                            metrics = single_shot_evaluate(
                                model, m4raw_loader, device, masks,
                                criterion=torch.nn.L1Loss(),
                                model_name=model_name,
                                num_steps=args.single_shot_steps,
                                lr=args.single_shot_lr,
                                save_images=args.save_images, save_path=args.save_path, dataset_name="m4raw"
                            )
                            print(f"\n  Loss before adapt: {metrics['loss_before']:.6f}")
                            print(f"  Loss after adapt:  {metrics['loss_after']:.6f}")
                        else:
                            print("  No masks found, falling back to standard eval")
                            metrics = evaluate_model(model, m4raw_loader, device, model_name,
                                                     save_images=args.save_images, save_path=args.save_path, dataset_name="m4raw")
                    else:
                        print(f"Evaluating on M4Raw...")
                        metrics = evaluate_model(model, m4raw_loader, device, model_name,
                                                 save_images=args.save_images, save_path=args.save_path, dataset_name="m4raw")
                    
                    metrics['total_params'] = total_params
                    metrics['nonzero_params'] = nonzero_params
                    metrics['sparsity'] = sparsity
                    metrics['masks_applied'] = masks_applied
                    m4raw_results[model_name] = metrics
                    
                    print(f"\nResults for {model_name} on M4Raw:")
                    print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
                    print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
                    
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\nError testing {model_name} on M4Raw: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if m4raw_results:
                print_results_table(m4raw_results, "M4RAW RESULTS (50 EPOCHS, BEST MODEL)")
        else:
            print("Warning: M4Raw dataset not available or empty.")
    
    # ==================== FastMRI Knee Evaluation ====================
    if (args.knee or args.knee_only) and not args.m4raw_only:
        print(f"\n{'=' * 80}")
        print("FASTMRI KNEE DOMAIN SHIFT EVALUATION")
        print(f"{'=' * 80}")
        
        max_samples = args.knee_max_samples if args.knee_max_samples > 0 else None
        knee_loader, knee_samples = create_knee_loader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=max_samples
        )
        
        if knee_loader is not None and knee_samples > 0:
            print(f"Total Knee samples: {knee_samples}")
            
            for model_name, model_info in MODELS_TO_TEST.items():
                checkpoint_dir = model_info['checkpoint']
                checkpoint_path = pathlib.Path(checkpoint_dir) / 'best_model.pt'
                
                if not checkpoint_path.exists():
                    continue
                
                print(f"\n{'=' * 80}")
                print(f"Testing: {model_name} on FastMRI Knee")
                print(f"{'=' * 80}")
                
                try:
                    print("Loading model...")
                    model, saved_args, masks_applied = load_model(checkpoint_path, device, apply_masks=True)
                    
                    total_params, nonzero_params = count_nonzero_params(model)
                    sparsity = 100 * (1 - nonzero_params / total_params) if total_params > 0 else 0
                    
                    # Single-shot learning: adapt on first example, evaluate on rest
                    if args.single_shot and masks_applied > 0:
                        print(f"Single-shot adaptation on Knee...")
                        # Get masks from checkpoint
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        masks = checkpoint.get('rigl_scheduler', {}).get('masks', {})
                        
                        if masks:
                            metrics = single_shot_evaluate(
                                model, knee_loader, device, masks,
                                criterion=torch.nn.L1Loss(),
                                model_name=model_name,
                                num_steps=args.single_shot_steps,
                                lr=args.single_shot_lr,
                                save_images=args.save_images, save_path=args.save_path, dataset_name="knee"
                            )
                            print(f"\n  Loss before adapt: {metrics['loss_before']:.6f}")
                            print(f"  Loss after adapt:  {metrics['loss_after']:.6f}")
                        else:
                            print("  No masks found, falling back to standard eval")
                            metrics = evaluate_model(model, knee_loader, device, model_name,
                                                     save_images=args.save_images, save_path=args.save_path, dataset_name="knee")
                    else:
                        print(f"Evaluating on Knee...")
                        metrics = evaluate_model(model, knee_loader, device, model_name,
                                                 save_images=args.save_images, save_path=args.save_path, dataset_name="knee")
                    
                    metrics['total_params'] = total_params
                    metrics['nonzero_params'] = nonzero_params
                    metrics['sparsity'] = sparsity
                    metrics['masks_applied'] = masks_applied
                    knee_results[model_name] = metrics
                    
                    print(f"\nResults for {model_name} on Knee:")
                    print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
                    print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
                    
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\nError testing {model_name} on Knee: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if knee_results:
                print_results_table(knee_results, "FASTMRI KNEE RESULTS (50 EPOCHS, BEST MODEL)")
        else:
            print("Warning: Knee dataset not available or empty.")
    
    # ==================== Domain Shift Comparison ====================
    if fastmri_results and m4raw_results:
        print_domain_shift_comparison(fastmri_results, m4raw_results)
    
    if fastmri_results and knee_results:
        print_domain_shift_comparison_knee(fastmri_results, knee_results)
    
    # ==================== Print sorted by PSNR ====================
    # Show sorted results for the primary dataset
    primary_results = fastmri_results if fastmri_results else (m4raw_results if m4raw_results else knee_results)
    primary_name = "FastMRI Brain" if fastmri_results else ("M4Raw" if m4raw_results else "FastMRI Knee")
    
    if primary_results:
        print("\n" + "=" * 120)
        print(f"RESULTS SORTED BY PSNR - {primary_name} (BEST TO WORST)")
        print("=" * 120)
        print(f"{'Rank':<6} | {'Model':<12} | {'PSNR (dB)':<20} | {'SSIM':<20} | {'Sparsity':<10}")
        print("-" * 120)
        
        sorted_results = sorted(primary_results.items(), key=lambda x: x[1]['psnr_mean'], reverse=True)
        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            psnr_str = f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}"
            ssim_str = f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"
            sparsity_str = f"{metrics['sparsity']:.1f}%"
            print(f"{rank:<6} | {model_name:<12} | {psnr_str:<20} | {ssim_str:<20} | {sparsity_str:<10}")
        
        print("=" * 120)
    
    # ==================== Save Results ====================
    if args.output_file:
        import json
        output = {}
        if fastmri_results:
            output['fastmri_brain'] = fastmri_results
        if m4raw_results:
            output['m4raw'] = m4raw_results
        if knee_results:
            output['fastmri_knee'] = knee_results
        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    main()

