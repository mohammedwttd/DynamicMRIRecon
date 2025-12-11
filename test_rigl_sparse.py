#!/usr/bin/env python3
"""
Test script to evaluate RigL (with sparsified weights), LightUNet, and Unet on FastMRI test set.
"""

import argparse
import pathlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def load_model(checkpoint_path, device, apply_sparsity=False, sparsity_threshold=1e-8):
    """Load model from checkpoint."""
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
    
    # Apply sparsity - zero out small weights
    if apply_sparsity:
        zero_count = 0
        total_count = 0
        for key in filtered_state_dict:
            if 'reconstruction_model' in key and 'weight' in key:
                param = filtered_state_dict[key]
                if param.dim() >= 2:
                    mask = param.abs() < sparsity_threshold
                    zero_count += mask.sum().item()
                    total_count += param.numel()
                    filtered_state_dict[key] = param * (~mask).float()
        
        print(f"  Applied sparsity: zeroed {zero_count:,}/{total_count:,} weights ({100*zero_count/total_count:.2f}%)")
    
    # Check if model was saved with DataParallel
    if list(filtered_state_dict.keys())[0].startswith('module.'):
        model = torch.nn.DataParallel(model)
    
    model.load_state_dict(filtered_state_dict)
    model.eval()
    
    return model, args


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


def count_parameters(model):
    """Count total and non-zero parameters."""
    total = 0
    nonzero = 0
    for name, param in model.named_parameters():
        if 'reconstruction_model' in name:
            total += param.numel()
            nonzero += (param != 0).sum().item()
    return total, nonzero


def main():
    parser = argparse.ArgumentParser(description='Test RigL sparse, LightUNet, and Unet')
    parser.add_argument('--data-path', type=pathlib.Path, 
                        default=pathlib.Path('../data'),
                        help='Path to the dataset directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create test loader
    print(f"\nLoading test data from: {args.data_path / 'multicoil_test'}")
    test_loader, num_samples = create_test_loader(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Total test samples: {num_samples}")
    
    # Models to test
    models_config = [
        {
            'name': 'RigL (sparse)',
            'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL_50/model.pt',
            'apply_sparsity': True,
            'sparsity_threshold': 1e-8,
        },
        {
            'name': 'RigL (dense)',
            'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL_50/model.pt',
            'apply_sparsity': False,
        },
        {
            'name': 'LightUNet',
            'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_LightUNet_30/best_model.pt',
            'apply_sparsity': False,
        },
        {
            'name': 'Unet',
            'checkpoint': 'summary/16/cartesian_1_4_0.08_0.0005_fixed_Unet_30/best_model.pt',
            'apply_sparsity': False,
        },
    ]
    
    results = {}
    
    for config in models_config:
        name = config['name']
        checkpoint_path = pathlib.Path(config['checkpoint'])
        
        if not checkpoint_path.exists():
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}. Skipping {name}.")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Testing: {name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'=' * 80}")
        
        try:
            # Load model
            print("Loading model...")
            model, saved_args = load_model(
                checkpoint_path, 
                device,
                apply_sparsity=config.get('apply_sparsity', False),
                sparsity_threshold=config.get('sparsity_threshold', 1e-8)
            )
            
            # Count parameters
            total_params, nonzero_params = count_parameters(model)
            sparsity = 100 * (1 - nonzero_params / total_params)
            print(f"  Parameters: {nonzero_params:,}/{total_params:,} non-zero ({sparsity:.2f}% sparse)")
            
            # Evaluate
            print(f"Evaluating on test set...")
            metrics = evaluate_model(model, test_loader, device, name)
            metrics['total_params'] = total_params
            metrics['nonzero_params'] = nonzero_params
            metrics['sparsity'] = sparsity
            results[name] = metrics
            
            print(f"\nResults for {name}:")
            print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
            print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
            print(f"  Evaluation time: {metrics['eval_time']:.1f} seconds")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError testing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: FASTMRI TEST SET RESULTS")
    print("=" * 100)
    print(f"{'Model':<20} | {'PSNR (dB)':<20} | {'SSIM':<20} | {'Params':<15} | {'Sparsity':<10}")
    print("-" * 100)
    
    for name, metrics in results.items():
        psnr_str = f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}"
        ssim_str = f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"
        params_str = f"{metrics['nonzero_params']/1e6:.2f}M"
        sparsity_str = f"{metrics['sparsity']:.1f}%"
        print(f"{name:<20} | {psnr_str:<20} | {ssim_str:<20} | {params_str:<15} | {sparsity_str:<10}")
    
    print("=" * 100)


if __name__ == '__main__':
    main()

