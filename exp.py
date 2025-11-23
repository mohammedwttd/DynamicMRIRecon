#!/home/mohammed-wa/miniconda3/envs/mpilot/bin/python
import os
import json
import shlex
import sys

data_path = '../data'

lr = {
    'Unet': {
        'rec_lr': 5e-4,
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'DynamicUnet': {
        'rec_lr': 5e-4,
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'CondUnet': {
        'rec_lr': 5e-4,
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'HybridCondUnet': {
        'rec_lr': 5e-4,
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'SmallCondUnet': {
        'rec_lr': 5e-4,
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'FDUnet': {
        'rec_lr': 2e-4,  # Lower LR for full FDConv (proper init, but freq-domain is sensitive)
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'HybridSnakeFDUnet': {
        'rec_lr': 3e-4,  # Balanced LR for Snake (geometry) + FDConv (frequency)
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'SmallHybridSnakeFDUnet': {
        'rec_lr': 4e-4,  # Slightly higher LR for smaller model
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'DCNFDUnet': {
        'rec_lr': 3e-4,  # Lower LR for FDConv stability (frequency-domain is sensitive)
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'SmallDCNFDUnet': {
        'rec_lr': 4e-4,  # Slightly higher LR for smaller model
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'vit-l-pretrained-cartesian-decoder': {
        'rec_lr': 1e-4,
        'sub_lr': {
            'cartesian': 0.05,
            'radial': 0.0025
        }
    },
    'vit-l-pretrained-cartesian': {
        'rec_lr': 1e-4,
        'sub_lr': {
            'cartesian': 0.5,
            'radial': 0.01 * 3
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'vit-l-pretrained-radial': {
        'rec_lr': 1e-4,
        'sub_lr': {
            'cartesian': 0.5,
            'radial': 0.01
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'vit-l': {
        'rec_lr': 5e-4,
        'sub_lr': {
            'cartesian': 0.1,
            'radial': 0.01
        },
        'noise': {
            'cartesian': 20,
            'radial': 80,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
}



acc_weight = 0.005
vel_weight = 0.001
batch_size = 1
n_shots = 16

# ═══════════════════════════════════════════════════════════════════════════════
# Model Selection & Training Parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Available models:
#   'Unet'                - Regular U-Net (baseline, ~2.5M params)
#   'DynamicUnet'         - Channel swapping U-Net (constant params)
#   'CondUnet'            - Full CondConv U-Net (~18M params, input-adaptive)
#   'HybridCondUnet'      - CondConv at bottleneck only (~1.8M params, 70-80% of baseline)
#   'SmallCondUnet'       - Full CondConv with fewer channels (~1.7M params, 60-70% of baseline)
#   'FDUnet'              - Frequency Dynamic Conv U-Net (~3M params, CVPR 2025, perfect for MRI!)
#   'HybridSnakeFDUnet'   - Dual-Domain U-Net: Snake (encoder) + FDConv (bottleneck+decoder)
#                           (~17M params, full capacity version)
#   'SmallHybridSnakeFDUnet' - Parameter-matched Dual-Domain U-Net (~2.5M params, FAIR comparison)
#                           Same architecture, fewer channels for fair comparison with baseline
#   'DCNFDUnet'           - DCNv2 Skip Aligner + FDConv Bottleneck (~2.5M params, baseline-level)
#                           Standard conv + DCNv2 adaptive alignment + FDConv frequency filtering
#   'SmallDCNFDUnet'      - Compact DCN-FD U-Net (~2.0M params, 80% of baseline)
#                           Fewer channels, minimal FDConv kernels for efficiency

model = 'SmallDCNFDUnet' 
init = 'cartesian'
noise = ''
noise_behaviour = ''

num_epochs = 30
trajectory_learning = 0
sample_rate = 1  # Start with reasonable sample rate
inter_gap_mode = "changing_downwards_15"

# ═══════════════════════════════════════════════════════════════════════════════
# Model-Specific Parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize all model-specific parameters with defaults
# (Will be overridden by model-specific blocks below)
swap_frequency = 10  # Dynamic U-Net
num_experts = 8  # CondUnet variants
fd_kernel_num = 4  # FDConv-based models
fd_use_simple = False  # FDConv-based models
snake_layers = 2  # Snake-based models
snake_kernel_size = 9  # Snake-based models
num_chans = 32  # Default base channels

# Dynamic U-Net parameters (only used if model='DynamicUnet')
# (Already set above)

# CondUnet parameters (used for CondUnet, HybridCondUnet, SmallCondUnet)
if model in ['CondUnet', 'HybridCondUnet', 'SmallCondUnet']:
    num_experts = 2  # Number of expert kernels per CondConv layer
    # Recommended values:
    #   4  - Lightweight, good for limited compute/memory
    #   8  - Balanced, recommended default (good performance/efficiency)
    #   16 - Maximum capacity, may overfit on small datasets

# FDUnet parameters (used for FDUnet)
if model == 'FDUnet':
    fd_kernel_num = 4  # Number of frequency-diverse kernels (only used if use_simple=False)
    # Recommended values:
    #   4  - Ultra lightweight, ~8x params vs standard Conv2d
    #   8  - Lightweight, ~16x params vs standard Conv2d
    #   16 - Moderate, ~32x params vs standard Conv2d
    #   32 - Heavy, ~64x params vs standard Conv2d
    #   64 - Very heavy, ~128x params vs standard Conv2d (original paper)
    # NOTE: kernel_num multiplies params by (kernel_num * 2), so keep it small!
    
    fd_use_simple = False  # Use simplified FDConv for FAIR COMPARISON with baseline U-Net
    # True  = Similar params to baseline U-Net (~2.5M), frequency-domain learning
    # False = Much higher params due to multiple frequency kernels

# HybridSnakeFDUnet parameters (Dual-Domain: Snake + FDConv)
if model == 'HybridSnakeFDUnet':
    snake_layers = 2  # Number of encoder layers to use Snake Conv (1-2 recommended)
    # 1 = Only first encoder layer (full resolution geometry)
    # 2 = First two encoder layers (balance geometry/computation)
    # 3+ = Heavier, more geometric detail but higher cost
    
    snake_kernel_size = 9  # Kernel size for Snake Conv (must be odd)
    # 7  = Lighter, shorter tubular structures
    # 9  = Balanced (recommended)
    # 11 = Heavier, longer tubular structures
    
    # FDConv settings (for bottleneck + decoder)
    fd_kernel_num = 4  # Keep small for parameter efficiency
    fd_use_simple = False  # Use full FDConv for frequency-domain learning
    
    num_chans = 32  # Full capacity (default)

elif model == 'SmallHybridSnakeFDUnet':
    # Parameter-matched version: ~2.5M params (same as baseline U-Net)
    snake_layers = 1  # Use only 1 Snake layer to save params
    snake_kernel_size = 9  # Keep same kernel size for effectiveness
    
    # FDConv settings (for bottleneck + decoder)
    fd_kernel_num = 4  # Keep small for efficiency
    fd_use_simple = False  # Use full FDConv
    
    num_chans = 12  # Reduced channels to match baseline param count
    # With chans=12, snake_layers=1, fd_kernel_num=4:
    #   Estimated: ~2.3-2.7M parameters (close to baseline 2.5M)
    # Comparison:
    #   chans=16 → ~4-5M params (too heavy)
    #   chans=12 → ~2.5M params (FAIR comparison)
    #   chans=10 → ~1.7M params (too light)

elif model == 'DCNFDUnet':
    fd_kernel_num = 2  # Reduce to 2 (FDConv multiplies params heavily!)
    fd_use_simple = False  # Use full FDConv at bottleneck
    
    num_chans = 20  # Reduced channels for ~2.5M params
    # FDConv bottleneck is VERY expensive with kernel_num=4
    # Actual measurements:
    #   chans=32, kernel_num=4 → 14M params (way too high!)
    #   chans=20, kernel_num=2 → ~2-3M params (target!)
    # Note: FDConv multiplies bottleneck params by (kernel_num * complexity_factor)

elif model == 'SmallDCNFDUnet':
    # Small DCN-FD U-Net: Minimal FDConv overhead for ~2M params
    fd_kernel_num = 2  # Keep at 2 for frequency diversity
    fd_use_simple = False  # Use full FDConv
    
    num_chans = 16  # Further reduced for ~2.0M params (80% of baseline)
    # Target: ~2.0M parameters
    # Configuration:
    #   chans=20, kernel_num=2 → ~2.5M params
    #   chans=18, kernel_num=2 → ~2.0M params (target!)
    #   chans=16, kernel_num=2 → ~1.6M params
    # This gives us a compact model while retaining DCNv2 alignment + FDConv filtering

# All model-specific parameters are now set (either by conditionals or defaults at top)

clr = lr[model]
sub_lr = clr['sub_lr'][init]
rec_lr = clr['rec_lr']
noise_std = clr['noise'][noise] if noise != '' else 0

TSP = ''
SNR = ''
weight_decay = 0
interp_gap = 10
acceleration = 4
center_fraction = 0.08

#model settings
img_size = [320, 320]
in_chans = 1
out_chans = 1
num_blocks = 1
sample_per_shot = 1600
drop_prob = 0.1

#relevant only for humus
window_size = 10
embed_dim = 66


#noise
noise_mode = None
epsilon = 0
noise_p = 0

if 'pgd' in noise:
    noise_behaviour += "_" + noise
    epsilon = noise_std
    noise_p = 0.5

if init == 'radial' and noise == 'radial':
    epsilon = noise_std
    noise_behaviour += "_noise"
    noise_p = 0.5

if init == 'cartesian' and noise == 'cartesian':
    epsilon = noise_std
    noise_behaviour += "_noise"
    noise_p = 0.5

if noise == 'image':
    epsilon = noise_std
    noise_behaviour += "_image"
    noise_p = 0.5


noise_type = "linf"
test_name = f'{n_shots}/{init}_{sample_rate}_'

if init == "cartesian":
    test_name += f'{acceleration}_{center_fraction}_'
else:
    test_name += f'{n_shots}_{sample_per_shot}_'

if trajectory_learning == 1:
    test_name += f'{rec_lr}_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}_{interp_gap}_{model}_{num_epochs}'
else:
    test_name += f'{rec_lr}_fixed_{model}_{num_epochs}'

if TSP == '--TSP':
    test_name += f'{rec_lr}_TSP_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}'

if SNR == '--SNR':
    test_name += '_SNR_flat_0.01'

if epsilon != 0:
    test_name += f"_{noise_behaviour}"
    test_name += f"_intensity_{epsilon}"
    test_name += f"_noise_p_{noise_p}"

command = f'python3 train.py --test-name={test_name} ' \
          f'--n-shots={n_shots} ' \
          f'--trajectory-learning={trajectory_learning} ' \
          f'--sub-lr={sub_lr} ' \
          f'--initialization={init} ' \
          f'--batch-size={batch_size} ' \
          f'--lr={rec_lr} ' \
          f'--num-epochs={num_epochs} ' \
          f'--acc-weight={acc_weight} ' \
          f'--vel-weight={vel_weight} ' \
          f'--data-path={data_path} ' \
          f'--sample-rate={sample_rate} ' \
          f'--data-parallel {TSP} ' \
          f'--weight-decay={weight_decay} ' \
          f'--inter-gap-mode={inter_gap_mode} ' \
          f'--model={model} ' \
          f'--in-chans={in_chans} ' \
          f'--out-chans={out_chans} ' \
          f'--num-blocks={num_blocks} ' \
          f'--window-size={window_size} ' \
          f'--embed-dim={embed_dim} ' \
          f'--interp_gap={interp_gap} ' \
          f'--drop-prob={drop_prob} '\
          f'--num-chans={num_chans} ' \
          f'--sample-per-shot={sample_per_shot} ' \
          f'--noise-mode={noise_mode} ' \
          f'--noise-behaviour={noise_behaviour} ' \
          f'--epsilon={epsilon} ' \
          f'--noise-type={noise_type} '  \
          f'--noise-p={noise_p} '  \
          f'--acceleration={acceleration} '  \
          f'--center-fraction={center_fraction} ' \
          f'--swap-frequency={swap_frequency} ' \
          f'--num-experts={num_experts} ' \
          f'--fd-kernel-num={fd_kernel_num} ' \
          f'{"--fd-use-simple " if fd_use_simple else ""}' \
          f'--snake-layers={snake_layers} ' \
          f'--snake-kernel-size={snake_kernel_size} '

# Print configuration summary
print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Model:              {model}")
if model == 'DynamicUnet':
    print(f"  Swap Frequency:   {swap_frequency} batches")
elif model in ['CondUnet', 'HybridCondUnet', 'SmallCondUnet']:
    print(f"  Num Experts:      {num_experts}")
elif model == 'FDUnet':
    print(f"  FD Kernel Num:    {fd_kernel_num}")
    print(f"  Use Simple FDConv: {fd_use_simple}")
elif model == 'HybridSnakeFDUnet':
    print(f"  Snake Layers:     {snake_layers} (encoder)")
    print(f"  Snake Kernel Size: {snake_kernel_size}")
    print(f"  FD Kernel Num:    {fd_kernel_num} (bottleneck+decoder)")
    print(f"  Use Simple FDConv: {fd_use_simple}")
    print(f"  Base Channels:    {num_chans}")
    print(f"  Architecture:     Anato-Spectral (Snake+FDConv)")
elif model == 'SmallHybridSnakeFDUnet':
    print(f"  Snake Layers:     {snake_layers} (encoder)")
    print(f"  Snake Kernel Size: {snake_kernel_size}")
    print(f"  FD Kernel Num:    {fd_kernel_num} (bottleneck+decoder)")
    print(f"  Use Simple FDConv: {fd_use_simple}")
    print(f"  Base Channels:    {num_chans} (reduced for param matching)")
    print(f"  Architecture:     Anato-Spectral (Snake+FDConv)")
    print(f"  Target:           ~2.5M params (FAIR comparison with baseline)")
elif model == 'DCNFDUnet':
    print(f"  FD Kernel Num:    {fd_kernel_num} (bottleneck)")
    print(f"  Use Simple FDConv: {fd_use_simple}")
    print(f"  Base Channels:    {num_chans}")
    print(f"  Encoder/Decoder:  Standard Conv")
    print(f"  Skip Aligners:    DCNv2 (adaptive alignment)")
    print(f"  Bottleneck:       FDConv (global frequency filtering)")
    print(f"  Target:           ~2.5M params (baseline-level)")
elif model == 'SmallDCNFDUnet':
    print(f"  FD Kernel Num:    {fd_kernel_num} (bottleneck)")
    print(f"  Use Simple FDConv: {fd_use_simple}")
    print(f"  Base Channels:    {num_chans} (reduced)")
    print(f"  Encoder/Decoder:  Standard Conv")
    print(f"  Skip Aligners:    DCNv2 (adaptive alignment)")
    print(f"  Bottleneck:       FDConv (global frequency filtering)")
    print(f"  Target:           ~2.0M params (80% of baseline)")
print(f"Initialization:     {init}")
print(f"Learning Rate:      {rec_lr} (reconstruction), {sub_lr} (subsampling)")
print(f"Epochs:             {num_epochs}")
print(f"Batch Size:         {batch_size}")
print(f"Shots:              {n_shots}")
print(f"Test Name:          {test_name}")
print("=" * 80)
print("Note: Parameters and FLOPs will be calculated at training start")
print("      (Install 'thop' for FLOP calculation: pip install thop)")
print("=" * 80 + "\n")

os.system(command)
