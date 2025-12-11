#!/usr/bin/env python3
"""
5-Model Ablation Study: Comparing DCN and FDConv contributions

Models:
1. Regular U-Net      (32 ch) - Baseline with full capacity
2. Light U-Net        (20 ch) - Reduced baseline
3. Light + DCN        (20 ch) - DCNv2 skip aligners only
4. Light + FD         (20 ch) - FDConv bottleneck only  
5. Light + Both       (20 ch) - DCNv2 + FDConv (full DCNFDUnet)

Run: python run_5_model_comparison.py
"""

import os
import subprocess
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Common settings
DATA_PATH = '../data'
INIT = 'cartesian'
BATCH_SIZE = 1
EPOCHS = 30
NUM_POOL_LAYERS = 4
DROP_PROB = 0.1
ACCELERATION = 4
CENTER_FRACTION = 0.08

# Model-specific configurations
MODELS = {
    # Model 1: Regular U-Net (32 channels) - BASELINE
    'Unet_32ch': {
        'model': 'Unet',
        'num_chans': 32,
        'rec_lr': 5e-4,
        'description': 'Regular U-Net (32 ch) - Full capacity baseline'
    },
    
    # Model 2: Light U-Net (20 channels) - Reduced baseline  
    'Unet_20ch': {
        'model': 'Unet',
        'num_chans': 20,
        'rec_lr': 5e-4,
        'description': 'Light U-Net (20 ch) - Reduced baseline'
    },
    
    # Model 3: Light + DCN (20 channels + DCNv2 skip aligners)
    'LightDCN': {
        'model': 'LightDCNUnet',
        'num_chans': 20,
        'rec_lr': 5e-4,
        'description': 'Light + DCN (20 ch) - DCNv2 skip alignment'
    },
    
    # Model 4: Light + FD (20 channels + FDConv bottleneck)
    'LightFD': {
        'model': 'LightFDUnet',
        'num_chans': 20,
        'rec_lr': 3e-4,  # Lower LR for FDConv stability
        'fd_kernel_num': 4,
        'fd_use_simple': True,  # Use simple (stable) version
        'description': 'Light + FD (20 ch) - FDConv bottleneck'
    },
    
    # Model 5: Light + Both (20 channels + DCNv2 + FDConv)
    'LightBoth': {
        'model': 'DCNFDUnet',
        'num_chans': 20,
        'rec_lr': 3e-4,  # Lower LR for FDConv stability
        'fd_kernel_num': 4,
        'fd_use_simple': True,  # Use simple (stable) version
        'description': 'Light + Both (20 ch) - DCNv2 + FDConv'
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def build_command(name, config):
    """Build the training command for a model."""
    cmd = [
        'python', 'train.py',
        '--data-path', DATA_PATH,
        '--init', INIT,
        '--type', config['model'],
        '--batch-size', str(BATCH_SIZE),
        '--num-epochs', str(EPOCHS),
        '--num-pool-layers', str(NUM_POOL_LAYERS),
        '--drop-prob', str(DROP_PROB),
        '--acceleration', str(ACCELERATION),
        '--center-fraction', str(CENTER_FRACTION),
        '--num-chans', str(config['num_chans']),
        '--rec-lr', str(config['rec_lr']),
        '--sub-lr', '0.025',  # Standard for cartesian
        '--exp-name', f'ablation_{name}',
    ]
    
    # Add FDConv parameters if applicable
    if 'fd_kernel_num' in config:
        cmd.extend(['--fd-kernel-num', str(config['fd_kernel_num'])])
    if 'fd_use_simple' in config:
        if config['fd_use_simple']:
            cmd.append('--fd-use-simple')
    
    return cmd


def print_banner(text, char='═'):
    """Print a banner."""
    width = 80
    print(char * width)
    print(f" {text}")
    print(char * width)


def run_single_model(name, config, dry_run=False):
    """Run training for a single model."""
    print_banner(f"MODEL: {name}", '─')
    print(f"Description: {config['description']}")
    print(f"Architecture: {config['model']}")
    print(f"Channels: {config['num_chans']}")
    print(f"Learning Rate: {config['rec_lr']}")
    if 'fd_kernel_num' in config:
        print(f"FD Kernel Num: {config['fd_kernel_num']}")
        print(f"FD Use Simple: {config.get('fd_use_simple', False)}")
    print()
    
    cmd = build_command(name, config)
    cmd_str = ' '.join(cmd)
    print(f"Command: {cmd_str}")
    print()
    
    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {name} interrupted by user")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run 5-model ablation study')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()) + ['all'], default=['all'],
                       help='Which models to run')
    args = parser.parse_args()
    
    # Select models to run
    if 'all' in args.models:
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = args.models
    
    print_banner("5-MODEL ABLATION STUDY")
    print(f"Models to run: {models_to_run}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Print summary table
    print_banner("MODEL SUMMARY", '─')
    print(f"{'Name':<15} {'Model':<15} {'Channels':<10} {'Description'}")
    print("─" * 80)
    for name in models_to_run:
        config = MODELS[name]
        print(f"{name:<15} {config['model']:<15} {config['num_chans']:<10} {config['description']}")
    print()
    
    # Run each model
    results = {}
    for i, name in enumerate(models_to_run, 1):
        print_banner(f"RUNNING MODEL {i}/{len(models_to_run)}: {name}")
        success = run_single_model(name, MODELS[name], dry_run=args.dry_run)
        results[name] = success
        print()
    
    # Print results summary
    print_banner("RESULTS SUMMARY")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    
    total_success = sum(results.values())
    print(f"\nCompleted: {total_success}/{len(results)} models")
    
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())

