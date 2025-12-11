#!/home/mohammed-wa/miniconda3/envs/mpilot/bin/python
import os
import json
import shlex
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MRI reconstruction training')
parser.add_argument('--model', type=str, default=None, help='Model name to train')
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
parser.add_argument('--dataset', type=str, default='mri', choices=['mri', 'imagenet'],
                    help='Dataset to use: mri (FastMRI) or imagenet')
parser.add_argument('--imagenet-path', type=str, default='../ImageNet',
                    help='Path to ImageNet dataset (only used if --dataset=imagenet)')
args, _ = parser.parse_known_args()

data_path = '../data'
dataset = args.dataset
imagenet_path = args.imagenet_path

# ImageNet-specific settings
# ImageNet has ~1.28M training images
# For 10 RigL updates per epoch with batch_size=32: update_freq = 1.28M / 32 / 10 = 4000
IMAGENET_BATCH_SIZE = 32
IMAGENET_RIGL_UPDATES_PER_EPOCH = 10
IMAGENET_TRAIN_SIZE = 1281167  # Approximate ImageNet train size

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
        'rec_lr': 1e-4,  # Slightly higher LR for smaller model
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
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURABLE U-NET VARIANTS (5-model ablation study)
    # All use ConfigurableUNet with different flags:
    #   - LightUNet:    use_dcn=False, use_fdconv=False (baseline)
    #   - LightDCN:     use_dcn=True,  use_fdconv=False
    #   - LightFD:      use_dcn=False, use_fdconv=True
    #   - LightDCNFD:   use_dcn=True,  use_fdconv=True (ours)
    # ═══════════════════════════════════════════════════════════════════════════
    'LightUNet': {
        'rec_lr': 5e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'LightDCN': {
        'rec_lr': 5e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'LightFD': {
        'rec_lr': 3e-4,  # Lower LR for FDConv
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'LightDCNFD': {
        'rec_lr': 3e-4,  # Lower LR for FDConv
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    # ═══════════════════════════════════════════════════════════════════════════
    # FULL FD U-NET: ALL conv blocks use FDConv
    # ═══════════════════════════════════════════════════════════════════════════
    'FullFDUnet': {
        'rec_lr': 2e-4,  # Lower LR for full FDConv (frequency-domain needs care)
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'FullFD': {
        'rec_lr': 2e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'LightFullFD': {
        'rec_lr': 2e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'FullFDDCN': {
        'rec_lr': 2e-4,  # Lower LR for FDConv + DCN
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    # ═══════════════════════════════════════════════════════════════════════════
    # HYBRID FD U-NET: FDConv only at deeper levels (L2+)
    # ═══════════════════════════════════════════════════════════════════════════
    'HybridFDUnet': {
        'rec_lr': 3e-4,  # Slightly higher than FullFD since early layers are standard
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'HybridFD': {
        'rec_lr': 3e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'HybridFDDCN': {
        'rec_lr': 2e-4,  # Lower LR when using DCN
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    # ═══════════════════════════════════════════════════════════════════════════
    # DEEP FD U-NET: FDConv only at deepest level (L3 + bottleneck)
    # ═══════════════════════════════════════════════════════════════════════════
    'DeepFDUnet': {
        'rec_lr': 3e-4,  # Higher LR since less FDConv layers
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'DeepFD': {
        'rec_lr': 3e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'DeepFDDCN': {
        'rec_lr': 2e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    # ═══════════════════════════════════════════════════════════════════════════
    # LIGHT OD U-NET: ODConv (Omni-Dimensional Dynamic Conv) at bottleneck
    # ═══════════════════════════════════════════════════════════════════════════
    'LightOD': {
        'rec_lr': 5e-4,  # Standard LR, ODConv is lightweight
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'LightODUnet': {
        'rec_lr': 5e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'LightODDCN': {
        'rec_lr': 3e-4,  # Slightly lower for DCN
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    # ═══════════════════════════════════════════════════════════════════════════
    # RigL U-NET: Dynamic Sparse Training
    # Automatically moves capacity from encoder to decoder during training
    # Use RigL10, RigL20, ..., RigL80 to set sparsity percentage
    # ═══════════════════════════════════════════════════════════════════════════
    'RigLUnet': {
        'rec_lr': 5e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'RigL': {
        'rec_lr': 5e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'RigLLight': {
        'rec_lr': 5e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    # RigL with specific sparsity levels (10% to 80%)
    'RigL10': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL20': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL30': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL40': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL50': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL60': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL70': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL80': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL90': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL95': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigL99': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # ═══════════════════════════════════════════════════════════════════════════
    # UNET WITHOUT SKIP CONNECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    'UnetSkipLess': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'SkipLess': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'NoSkip': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # RigL on UNet without skip connections
    'RigLSkipLess': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLNoSkip': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLSkipLess10': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLSkipLess20': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLSkipLess30': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLSkipLess40': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLSkipLess50': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLSkipLess60': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLSkipLess70': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLSkipLess80': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # ═══════════════════════════════════════════════════════════════════════════
    # STAMP UNET: Simultaneous Training and Model Pruning (channel-wise/structured)
    # Based on: https://github.com/nkdinsdale/STAMP.git
    # Paper: "STAMP: Simultaneous Training and Model Pruning for Low Data Regimes 
    #         in Medical Image Segmentation" (Medical Image Analysis, 2022)
    # ═══════════════════════════════════════════════════════════════════════════
    'STAMP': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'STAMPUnet': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'STAMP10': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'STAMP20': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'STAMP30': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'STAMP40': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'STAMP50': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # STAMP with different importance scoring modes (paper ablations)
    'STAMP_L1': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'STAMP_L2': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'STAMP_Random': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # ═══════════════════════════════════════════════════════════════════════════
    # STATIC SPARSE UNET: RigL-inspired architecture with static channel reduction
    # Bottleneck aggressively narrowed, early/late layers preserved (like RigL learns)
    # ═══════════════════════════════════════════════════════════════════════════
    'StaticSparseLight': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'StaticSparseMedium': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'StaticSparseHeavy': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'StaticSparse': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # StaticSparse48Wide: 48 base channels, slim bottleneck (~3.35M params like regular UNet)
    'StaticSparse48Wide': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # StaticSparseUltraLight: 32 base, extreme bottleneck compression (~0.5M params)
    'StaticSparseUltraLight': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # StaticSparseFlat32: Uniform 32 channels throughout (~300K params, very compact)
    'StaticSparseFlat32': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # StaticSparse64Extreme: 64 base, extreme compression (~1.5M params, heavier than heavy)
    'StaticSparse64Extreme': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # StaticSparse64ExtremeSlim: 64 base with slimmer decoder
    'StaticSparse64ExtremeSlim': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # StaticSparse64DecoderHeavy: 64 base, slim encoder, fuller decoder (RigL-inspired)
    'StaticSparse64DecoderHeavy': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # ═══════════════════════════════════════════════════════════════════════════
    # ASYMMETRIC StaticSparse: RigL-learned pattern - slim encoder, fuller decoder
    # Based on RigL60 learning: Encoder 36% dense, Decoder 50% dense
    # ═══════════════════════════════════════════════════════════════════════════
    'StaticSparseAsymmetric': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'StaticSparseAsymmetricSlim': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # ═══════════════════════════════════════════════════════════════════════════
    # RigL + StaticSparse: Double sparse - architectural + dynamic weight sparsity
    # These combine StaticSparse architecture (channel reduction) with RigL training (weight sparsity)
    # ═══════════════════════════════════════════════════════════════════════════
    # RigLStaticSparse48Wide: 2.04M params architecture + 50% weight sparsity → ~1.02M effective
    'RigLStaticSparse48Wide': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse48Wide10': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse48Wide20': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse48Wide30': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse48Wide40': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse48Wide50': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse48Wide60': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse48Wide70': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse48Wide80': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # RigLStaticSparse64Extreme: 1.24M params architecture + 50% weight sparsity → ~0.62M effective
    'RigLStaticSparse64Extreme': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse64Extreme10': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse64Extreme20': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse64Extreme30': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse64Extreme40': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse64Extreme50': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse64Extreme60': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse64Extreme70': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'RigLStaticSparse64Extreme80': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # UnetMasked{X} - Train PRUNED weights (frozen active weights from RigL{X})
    'UnetMasked10': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMasked20': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMasked30': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMasked40': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMasked50': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMasked60': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMasked70': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMasked80': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # UnetMaskedDecoder{X} - Train only DECODER's pruned weights (encoder/bottleneck pruned stay at 0)
    'UnetMaskedDecoder10': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMaskedDecoder20': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMaskedDecoder30': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMaskedDecoder40': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMaskedDecoder50': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMaskedDecoder60': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMaskedDecoder70': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    'UnetMaskedDecoder80': {'rec_lr': 5e-4, 'sub_lr': {'cartesian': 0.025, 'radial': 0.005}, 'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}},
    # Legacy names (backward compatibility)
    'LightDCNUnet': {
        'rec_lr': 5e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
    },
    'LightFDUnet': {
        'rec_lr': 3e-4,
        'sub_lr': {'cartesian': 0.025, 'radial': 0.005},
        'noise': {'cartesian': 10, 'radial': 30, 'image': 6e-5, 'radial_pgd': 1, 'cartesian_pgd': 4, 'none': 0}
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
# Batch size (higher for ImageNet)
if dataset == 'imagenet':
    batch_size = IMAGENET_BATCH_SIZE
else:
    batch_size = 1  # MRI default
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

# Model can be set via command line: python exp.py --model LightDCNFD
model = args.model if args.model else 'UnetMasked30'
init = 'cartesian'
noise = ''
noise_behaviour = ''

num_epochs = args.epochs if args.epochs else 50
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
od_kernel_num = 1  # ODConv-based models (single kernel)
snake_layers = 2  # Snake-based models
snake_kernel_size = 9  # Snake-based models
num_chans = 32  # Default base channels
use_dcn = False  # ConfigurableUNet: DCNv2 skip aligners
use_fdconv = False  # ConfigurableUNet: FDConv bottleneck
use_rigl = False  # RigL dynamic sparse training
rigl_sparsity = 0.5  # Target sparsity (0.5 = 50% zeros)
# RigL update frequency: for ImageNet, calculate to get ~10 updates per epoch
if dataset == 'imagenet':
    iters_per_epoch = IMAGENET_TRAIN_SIZE // batch_size
    rigl_update_freq = max(1, iters_per_epoch // IMAGENET_RIGL_UPDATES_PER_EPOCH)
    print(f"[ImageNet RigL] iters/epoch={iters_per_epoch}, update_freq={rigl_update_freq} (~{IMAGENET_RIGL_UPDATES_PER_EPOCH} updates/epoch)")
else:
    rigl_update_freq = 100  # MRI default: update every 100 iterations
rigl_delta = 0.31 # Fraction of weights to reallocate
freeze_masked = False  # Train only non-masked weights (frozen mask training)
freeze_masked_decoder_only = False  # Train only DECODER's pruned weights
rigl_checkpoint = None  # Checkpoint to load RigL masks from (for freeze_masked)
use_stamp = False  # STAMP channel-wise pruning (https://github.com/nkdinsdale/STAMP.git)
stamp_channel_drop_rate = 0.1  # Paper default: b_drop = 0.1 (10% dropout)
stamp_prune_epochs = []  # Epochs at which to prune (auto-generated if empty)
stamp_prune_ratio = 0.5  # Keep ratio at each prune (50% channels kept)
stamp_recovery_epochs = 5  # Paper default: Recovery epochs = 5
stamp_mode = 'Taylor'  # Importance scoring: 'Taylor' (default), 'L1', 'L2', 'Random'

# UnetMaskedDecoder{X} - Only train decoder's pruned weights, encoder/bottleneck pruned stay at 0
# e.g., UnetMaskedDecoder50 loads masks from RigL50_50 checkpoint
if model.startswith('UnetMaskedDecoder') and model[17:].isdigit():
    sparsity_pct = int(model[17:])
    freeze_masked_decoder_only = True
    rigl_checkpoint = f'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL{sparsity_pct}_50/best_model.pt'
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    print(f"UnetMaskedDecoder{sparsity_pct}: Loading masks from {rigl_checkpoint}")
    print(f"  -> Only decoder's pruned weights will be trained")
    print(f"  -> Encoder/bottleneck pruned weights stay frozen at 0")

# UnetMasked{X} - Automatically enable frozen mask training with RigL{X} masks
# e.g., UnetMasked50 loads masks from RigL50_50 checkpoint
elif model.startswith('UnetMasked') and model[10:].isdigit():
    sparsity_pct = int(model[10:])
    freeze_masked = True
    rigl_checkpoint = f'summary/16/cartesian_1_4_0.08_0.0005_fixed_RigL{sparsity_pct}_50/best_model.pt'
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    print(f"UnetMasked{sparsity_pct}: Loading masks from {rigl_checkpoint}")

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
    fd_kernel_num = 4  # Keep at 2 for frequency diversity
    fd_use_simple = False  # Use full FDConv
    
    num_chans = 20  # Further reduced for ~2.0M params (80% of baseline)
    # Target: ~2.0M parameters
    # Configuration:
    #   chans=20, kernel_num=2 → ~2.5M params
    #   chans=18, kernel_num=2 → ~2.0M params (target!)
    #   chans=16, kernel_num=2 → ~1.6M params
    # This gives us a compact model while retaining DCNv2 alignment + FDConv filtering

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURABLE U-NET VARIANTS (Ablation Study)
# All use ConfigurableUNet with different use_dcn/use_fdconv combinations
# ═══════════════════════════════════════════════════════════════════════════════

elif model == 'LightUNet':
    # Baseline: No DCN, No FDConv
    num_chans = 20
    use_dcn = False
    use_fdconv = False

elif model == 'LightDCN':
    # DCN only: DCNv2 skip aligners, standard bottleneck
    num_chans = 20
    use_dcn = True
    use_fdconv = False

elif model == 'LightFD':
    # FDConv only: Standard skips, FDConv bottleneck
    num_chans = 20
    use_dcn = False
    use_fdconv = True
    fd_kernel_num = 4

elif model == 'LightDCNFD':
    # Both: DCNv2 skip aligners + FDConv bottleneck (Ours)
    num_chans = 20
    use_dcn = True
    use_fdconv = True
    fd_kernel_num = 4

# ═══════════════════════════════════════════════════════════════════════════
# FULL FD U-NET: ALL conv blocks use FDConv (encoder, bottleneck, decoder)
# ═══════════════════════════════════════════════════════════════════════════
elif model in ['FullFDUnet', 'FullFD', 'LightFullFD']:
    # Full FDConv everywhere: No DCN by default
    num_chans = 20
    use_dcn = False
    use_fdconv = True  # Not used directly, but for consistency
    fd_kernel_num = 4

elif model == 'FullFDDCN':
    # Full FDConv everywhere + DCN skip refiners
    num_chans = 20
    use_dcn = True
    use_fdconv = True
    fd_kernel_num = 4

# ═══════════════════════════════════════════════════════════════════════════
# HYBRID FD U-NET: FDConv only at deeper levels (L2+)
# ═══════════════════════════════════════════════════════════════════════════
elif model in ['HybridFDUnet', 'HybridFD']:
    # Level 0-1: StandardConv (preserve edges/details)
    # Level 2-3 + Bottleneck: FDConv (global frequency context)
    num_chans = 20
    use_dcn = False
    use_fdconv = True
    fd_kernel_num = 4

elif model == 'HybridFDDCN':
    # Same as HybridFDUnet but with DCN skip refiners
    num_chans = 20
    use_dcn = True
    use_fdconv = True
    fd_kernel_num = 4

# ═══════════════════════════════════════════════════════════════════════════
# DEEP FD U-NET: FDConv only at deepest level (L3 + bottleneck)
# ═══════════════════════════════════════════════════════════════════════════
elif model in ['DeepFDUnet', 'DeepFD']:
    # More conservative than HybridFD - FDConv only at L3 + bottleneck
    num_chans = 20
    use_dcn = False
    use_fdconv = True
    fd_kernel_num = 4

elif model == 'DeepFDDCN':
    # DeepFD + DCN skip refiners
    num_chans = 20
    use_dcn = True
    use_fdconv = True
    fd_kernel_num = 4

# ═══════════════════════════════════════════════════════════════════════════
# LIGHT OD U-NET: ODConv at bottleneck (omni-dimensional dynamic convolution)
# ═══════════════════════════════════════════════════════════════════════════
elif model in ['LightOD', 'LightODUnet']:
    # ODConv at bottleneck with single kernel
    # Provides channel, filter, spatial attention dynamically
    num_chans = 20
    use_dcn = False
    use_fdconv = False
    od_kernel_num = 1

elif model == 'LightODDCN':
    # ODConv at bottleneck + DCN skip refiners
    num_chans = 20
    use_dcn = True
    use_fdconv = False
    od_kernel_num = 1

# ═══════════════════════════════════════════════════════════════════════════
# RigL U-NET: Dynamic Sparse Training
# ═══════════════════════════════════════════════════════════════════════════
elif model in ['RigLUnet', 'RigL']:
    # RigL with 50% sparsity, automatically reallocates weights
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    use_rigl = True
    rigl_sparsity = 0.5
    rigl_update_freq = 2000
    rigl_delta = 0.1

elif model == 'RigLLight':
    # Lighter RigL for faster experimentation
    num_chans = 20
    use_dcn = False
    use_fdconv = False
    use_rigl = True
    rigl_sparsity = 0.5
    rigl_update_freq = 100
    rigl_delta = 0.3

# RigL with specific sparsity: RigL10, RigL20, ..., RigL80
elif model.startswith('RigL') and model[4:].isdigit():
    # Extract sparsity from model name (e.g., RigL50 -> 0.5)
    sparsity_pct = int(model[4:])
    rigl_sparsity = sparsity_pct / 100.0
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    use_rigl = True
    rigl_delta = 0.1
    print(f"RigL with {sparsity_pct}% sparsity selected")

# ═══════════════════════════════════════════════════════════════════════════════
# UNet without Skip Connections
# ═══════════════════════════════════════════════════════════════════════════════
elif model in ['UnetSkipLess', 'SkipLess', 'NoSkip']:
    # UNet without skip connections - pure encoder-decoder
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    use_rigl = False
    print(f"UnetSkipLess selected: UNet without skip connections")

# RigLSkipLess: RigL with UNet without skip connections
elif model in ['RigLSkipLess', 'RigLNoSkip']:
    # RigL with 50% sparsity on skip-less UNet
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    use_rigl = True
    rigl_sparsity = 0.5
    rigl_delta = 0.1
    print(f"RigLSkipLess selected: RigL on UNet without skip connections")

# RigLSkipLess with specific sparsity: RigLSkipLess10, RigLSkipLess50, etc.
elif model.startswith('RigLSkipLess') and model[11:].isdigit():
    sparsity_pct = int(model[11:])
    rigl_sparsity = sparsity_pct / 100.0
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    use_rigl = True
    rigl_delta = 0.1
    print(f"RigLSkipLess with {sparsity_pct}% sparsity selected")

# ═══════════════════════════════════════════════════════════════════════════════
# STAMP UNet: Simultaneous Training and Model Pruning
# Based on: https://github.com/nkdinsdale/STAMP.git
# Paper: "STAMP: Simultaneous Training and Model Pruning for Low Data Regimes 
#         in Medical Image Segmentation" (Medical Image Analysis, 2022)
# ═══════════════════════════════════════════════════════════════════════════════
# Channel-wise (structured) pruning - more hardware efficient than RigL
elif model in ['STAMP', 'STAMPUnet']:
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    use_rigl = False
    use_stamp = True
    stamp_channel_drop_rate = 0.1  # Paper default: b_drop = 0.1
    stamp_prune_epochs = []  # Auto-generate: every recovery_epochs until 80% of training
    stamp_prune_ratio = 0.99  # Keep 75% of channels at each prune
    stamp_recovery_epochs = 1  # Paper default: 5 recovery epochs between prunings
    stamp_mode = 'Taylor'  # Paper default: Taylor expansion importance scoring
    print(f"STAMP UNet selected: Channel-wise structured pruning (paper defaults)")
    print(f"  Mode: Taylor expansion (gradient * activation)")

# STAMP with different importance scoring modes: STAMP_L1, STAMP_L2, STAMP_Random
elif model.startswith('STAMP_'):
    mode_suffix = model.split('_')[1]
    stamp_mode = mode_suffix  # 'L1', 'L2', or 'Random'
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    use_rigl = False
    use_stamp = True
    stamp_channel_drop_rate = 0.1  # Paper default: b_drop = 0.1
    stamp_prune_epochs = []  # Auto-generate based on recovery_epochs
    stamp_prune_ratio = 0.75  # Keep 75% of channels at each prune
    stamp_recovery_epochs = 5  # Paper default
    print(f"STAMP with {stamp_mode} importance scoring selected")

# STAMP with specific channel drop rate: STAMP10, STAMP20, ..., STAMP50
elif model.startswith('STAMP') and model[5:].isdigit():
    drop_pct = int(model[5:])
    stamp_channel_drop_rate = drop_pct / 100.0
    num_chans = 32
    use_dcn = False
    use_fdconv = False
    use_rigl = False
    use_stamp = True
    stamp_prune_epochs = []  # Auto-generate based on recovery_epochs
    stamp_prune_ratio = 0.5
    stamp_recovery_epochs = 5  # Paper default
    stamp_mode = 'Taylor'  # Paper default
    print(f"STAMP with {drop_pct}% channel dropout selected (b_drop={stamp_channel_drop_rate})")

# ═══════════════════════════════════════════════════════════════════════════════
# Static Sparse UNet - Inspired by RigL learned sparsity patterns
# ═══════════════════════════════════════════════════════════════════════════════
# Key insight: RigL learns non-uniform sparsity - bottleneck ~70%, early/late ~0%
# These models statically implement this pattern through channel reduction
elif model == 'StaticSparseLight':
    # ~1.1M params (67% reduction from standard UNet)
    num_chans = 32  # Base channels (will be reduced at bottleneck)
    use_dcn = False
    use_fdconv = False

elif model == 'StaticSparseMedium' or model == 'StaticSparse':
    # ~580K params (83% reduction) - matches RigL50 effective capacity
    num_chans = 32
    use_dcn = False
    use_fdconv = False

elif model == 'StaticSparseHeavy':
    # ~319K params (90% reduction) - very compact
    num_chans = 32
    use_dcn = False
    use_fdconv = False

elif model == 'StaticSparse48Wide':
    # 48 base channels with slim bottleneck (~3.35M params like regular UNet)
    # Wide early layers (48, 96) for better feature extraction
    # Heavily compressed bottleneck (96 instead of 384)
    num_chans = 48
    use_dcn = False
    use_fdconv = False

elif model == 'StaticSparseUltraLight':
    # 32 base channels with EXTREME bottleneck reduction (~0.5M params)
    # Full capacity at edges, severely compressed center (like RigL99 learns)
    # Bottleneck only 32 channels instead of 256
    num_chans = 32
    use_dcn = False
    use_fdconv = False

elif model == 'StaticSparseFlat32':
    # UNIFORM 32 channels throughout - flat architecture (~300K params)
    # Every layer has exactly 32 channels (no pyramid structure)
    num_chans = 32
    use_dcn = False
    use_fdconv = False

elif model == 'StaticSparse64Extreme':
    # 64 base channels with EXTREME compression (~1.5M params)
    # Wide early layers (64, 128) but extremely compressed deep layers
    # Heavier compression than StaticSparseHeavy
    num_chans = 64
    use_dcn = False
    use_fdconv = False

elif model == 'StaticSparse64ExtremeSlim':
    # 64 base channels with slimmer decoder
    # Wide encoder (64, 128, 128, 96) but slim decoder (64, 64, 64, 64)
    num_chans = 64
    use_dcn = False
    use_fdconv = False

elif model == 'StaticSparse64DecoderHeavy':
    # RigL-inspired: slim encoder, fuller decoder
    # Encoder: 64 -> 64 -> 64 -> 48 (compressed)
    # Decoder: 96 -> 128 -> 128 -> 64 (heavy capacity)
    num_chans = 64
    use_dcn = False
    use_fdconv = False

# ═══════════════════════════════════════════════════════════════════════════════
# ASYMMETRIC StaticSparse: RigL-learned pattern (slim encoder, fuller decoder)
# ═══════════════════════════════════════════════════════════════════════════════
elif model == 'StaticSparseAsymmetric':
    # Asymmetric architecture learned from RigL60 on StaticSparse64Extreme
    # Key insight: Encoder can be 36% dense, Decoder needs 50% dense
    # Full capacity at edges (down0, up3), very slim in middle
    num_chans = 48  # Base channels
    use_dcn = False
    use_fdconv = False
    print(f"StaticSparseAsymmetric: RigL-learned asymmetric pattern")
    print(f"  Slim encoder (~36% density), Fuller decoder (~50% density)")

elif model == 'StaticSparseAsymmetricSlim':
    # Ultra-slim asymmetric - even more aggressive compression
    # Target: ~400K params for ultra-efficient deployment
    num_chans = 32  # Smaller base
    use_dcn = False
    use_fdconv = False
    print(f"StaticSparseAsymmetricSlim: Ultra-slim RigL-learned pattern")
    print(f"  ~400K params, very aggressive encoder compression")

# ═══════════════════════════════════════════════════════════════════════════════
# RigL + StaticSparse: DOUBLE SPARSE (Architecture + Dynamic Weight Sparsity)
# Combines channel reduction (StaticSparse) with weight pruning (RigL)
# ═══════════════════════════════════════════════════════════════════════════════
elif model == 'RigLStaticSparse48Wide' or model.startswith('RigLStaticSparse48Wide'):
    # StaticSparse48Wide architecture (2.04M) + RigL weight sparsity
    num_chans = 48  # Base channels for StaticSparse48Wide
    use_dcn = False
    use_fdconv = False
    use_rigl = True
    rigl_delta = 0.1
    # Extract sparsity from name if specified (e.g., RigLStaticSparse48Wide50 -> 50%)
    if model == 'RigLStaticSparse48Wide':
        rigl_sparsity = 0.5  # Default 50%
    else:
        sparsity_suffix = model.replace('RigLStaticSparse48Wide', '')
        if sparsity_suffix.isdigit():
            rigl_sparsity = int(sparsity_suffix) / 100.0
        else:
            rigl_sparsity = 0.5
    print(f"RigLStaticSparse48Wide with {rigl_sparsity*100:.0f}% weight sparsity")
    print(f"  Architecture: 2.04M params → {2.04*(1-rigl_sparsity):.2f}M effective with RigL")

elif model == 'RigLStaticSparse64Extreme' or model.startswith('RigLStaticSparse64Extreme'):
    # StaticSparse64Extreme architecture (1.24M) + RigL weight sparsity
    num_chans = 64  # Base channels for StaticSparse64Extreme
    use_dcn = False
    use_fdconv = False
    use_rigl = True
    rigl_delta = 0.1
    # Extract sparsity from name if specified (e.g., RigLStaticSparse64Extreme50 -> 50%)
    if model == 'RigLStaticSparse64Extreme':
        rigl_sparsity = 0.5  # Default 50%
    else:
        sparsity_suffix = model.replace('RigLStaticSparse64Extreme', '')
        if sparsity_suffix.isdigit():
            rigl_sparsity = int(sparsity_suffix) / 100.0
        else:
            rigl_sparsity = 0.5
    print(f"RigLStaticSparse64Extreme with {rigl_sparsity*100:.0f}% weight sparsity")
    print(f"  Architecture: 1.24M params → {1.24*(1-rigl_sparsity):.2f}M effective with RigL")

# Legacy names (backward compatibility)
elif model == 'LightDCNUnet':
    num_chans = 32
    use_dcn = True
    use_fdconv = False

elif model == 'LightFDUnet':
    num_chans = 32
    use_dcn = False
    use_fdconv = True
    fd_kernel_num = 4

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
          f'--dataset={dataset} ' \
          f'--imagenet-path={imagenet_path} ' \
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
          f'--snake-kernel-size={snake_kernel_size} ' \
          f'{"--use-dcn " if use_dcn else ""}' \
          f'{"--use-fdconv " if use_fdconv else ""}' \
          f'{"--use-rigl " if use_rigl else ""}' \
          f'--rigl-sparsity={rigl_sparsity} ' \
          f'--rigl-update-freq={rigl_update_freq} ' \
          f'--rigl-delta={rigl_delta} ' \
          f'{"--freeze-masked " if freeze_masked else ""}' \
          f'{"--freeze-masked-decoder-only " if freeze_masked_decoder_only else ""}' \
          f'{f"--rigl-checkpoint={rigl_checkpoint} " if rigl_checkpoint else ""}' \
          f'{"--use-stamp " if use_stamp else ""}' \
          f'--stamp-channel-drop-rate={stamp_channel_drop_rate} ' \
          f'--stamp-prune-ratio={stamp_prune_ratio} ' \
          f'--stamp-recovery-epochs={stamp_recovery_epochs} ' \
          f'--stamp-mode={stamp_mode} ' \
          f'{"--stamp-prune-epochs " + " ".join(map(str, stamp_prune_epochs)) + " " if stamp_prune_epochs else ""}'

# Print configuration summary
print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Dataset:            {dataset}")
if dataset == 'imagenet':
    print(f"  ImageNet Path:    {imagenet_path}")
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
elif model in ['FullFDUnet', 'FullFD', 'LightFullFD', 'FullFDDCN']:
    print(f"  Base Channels:    {num_chans}")
    print(f"  FD Kernel Num:    {fd_kernel_num}")
    print(f"  Use DCN:          {use_dcn}")
    print(f"  Architecture:     ALL conv blocks use FDConv")
    print(f"  Encoder:          FDConv blocks")
    print(f"  Bottleneck:       FDConv block")
    print(f"  Decoder:          FDConv blocks")
    if use_dcn:
        print(f"  Skip Refiners:    DCNv2 (adaptive alignment)")
elif model in ['HybridFDUnet', 'HybridFD', 'HybridFDDCN']:
    print(f"  Base Channels:    {num_chans}")
    print(f"  FD Kernel Num:    {fd_kernel_num}")
    print(f"  Use DCN:          {use_dcn}")
    print(f"  Architecture:     Hybrid (StandardConv early, FDConv deep)")
    print(f"  Level 0 (320):    StandardConv - Preserve edges")
    print(f"  Level 1 (160):    StandardConv - Local features")
    print(f"  Level 2 (80):     FDConv - Global context")
    print(f"  Level 3 (40):     FDConv - Frequency patterns")
elif model in ['DeepFDUnet', 'DeepFD', 'DeepFDDCN']:
    print(f"  Base Channels:    {num_chans}")
    print(f"  FD Kernel Num:    {fd_kernel_num}")
    print(f"  Use DCN:          {use_dcn}")
    print(f"  Architecture:     Deep (FDConv only at deepest)")
    print(f"  Level 0 (320):    StandardConv - Preserve edges")
    print(f"  Level 1 (160):    StandardConv - Local features")
    print(f"  Level 2 (80):     StandardConv - Mid-level features")
    print(f"  Level 3 (40):     FDConv - Global frequency patterns")
    print(f"  Bottleneck (20):  FDConv - Maximum global context")
    print(f"  Bottleneck (20):  FDConv - Maximum global context")
    if use_dcn:
        print(f"  Skip Refiners:    DCNv2 (adaptive alignment)")
elif model in ['LightOD', 'LightODUnet', 'LightODDCN']:
    print(f"  Base Channels:    {num_chans}")
    print(f"  OD Kernel Num:    {od_kernel_num}")
    print(f"  Use DCN:          {use_dcn}")
    print(f"  Architecture:     ODConv at Bottleneck")
    print(f"  Encoder:          StandardConv blocks")
    print(f"  Bottleneck:       ODConv (channel + filter + spatial attention)")
    print(f"  Decoder:          StandardConv blocks")
    if use_dcn:
        print(f"  Skip Refiners:    DCNv2 (adaptive alignment)")
elif model.startswith('RigLStaticSparse'):
    # Double-sparse models: StaticSparse architecture + RigL weight sparsity
    if 'StaticSparse48Wide' in model:
        arch_name = 'StaticSparse48Wide'
        arch_params = '2.04M'
    else:  # StaticSparse64Extreme
        arch_name = 'StaticSparse64Extreme'
        arch_params = '1.24M'
    print(f"  Base Channels:    {num_chans}")
    print(f"  Architecture:     {arch_name} ({arch_params} params)")
    print(f"  RigL Sparsity:    {rigl_sparsity*100:.0f}% (weight-level)")
    print(f"  Update Freq:      2x per epoch (dynamic)")
    print(f"  Delta:            {rigl_delta*100:.0f}% weights reallocated per update")
    print(f"  Sparsity Type:    DOUBLE SPARSE (architecture + weights)")
    print(f"  Mechanism:        Channel reduction (architecture) + weight pruning (RigL)")
    print(f"  Expected:         Ultra-compact model with learned sparsity pattern")
elif model in ['RigLUnet', 'RigL', 'RigLLight'] or (model.startswith('RigL') and model[4:].isdigit() and 'SkipLess' not in model and 'StaticSparse' not in model):
    print(f"  Base Channels:    {num_chans}")
    print(f"  RigL Sparsity:    {rigl_sparsity*100:.0f}%")
    print(f"  Update Freq:      2x per epoch (dynamic)")
    print(f"  Delta:            {rigl_delta*100:.0f}% weights reallocated per update")
    print(f"  Architecture:     Dynamic Sparse U-Net")
    print(f"  Mechanism:        Drops low-magnitude weights, grows high-gradient positions")
    print(f"  Expected:         Encoder becomes sparser, Decoder becomes denser")
elif model in ['UnetSkipLess', 'SkipLess', 'NoSkip']:
    print(f"  Base Channels:    {num_chans}")
    print(f"  Architecture:     UNet WITHOUT Skip Connections")
    print(f"  Encoder:          Standard ConvBlocks with pooling")
    print(f"  Decoder:          Standard ConvBlocks with upsampling")
    print(f"  Skip Connections: DISABLED (pure encoder-decoder)")
elif model in ['RigLSkipLess', 'RigLNoSkip'] or model.startswith('RigLSkipLess'):
    print(f"  Base Channels:    {num_chans}")
    print(f"  RigL Sparsity:    {rigl_sparsity*100:.0f}%")
    print(f"  Update Freq:      2x per epoch (dynamic)")
    print(f"  Delta:            {rigl_delta*100:.0f}% weights reallocated per update")
    print(f"  Architecture:     Dynamic Sparse UNet WITHOUT Skip Connections")
    print(f"  Skip Connections: DISABLED")
    print(f"  Mechanism:        Drops low-magnitude weights, grows high-gradient positions")
elif model in ['STAMP', 'STAMPUnet'] or model.startswith('STAMP'):
    print(f"  Base Channels:    {num_chans}")
    print(f"  b_drop (dropout): {stamp_channel_drop_rate*100:.0f}% (paper default: 10%)")
    print(f"  Keep Ratio:       {stamp_prune_ratio*100:.0f}% at each prune")
    print(f"  Recovery Epochs:  {stamp_recovery_epochs} (paper default: 5)")
    print(f"  Prune Epochs:     {stamp_prune_epochs if stamp_prune_epochs else 'auto (every recovery_epochs)'}")
    print(f"  Architecture:     STAMP UNet (Simultaneous Training and Model Pruning)")
    print(f"  Pruning Type:     Channel-wise (structured) - actual network resize!")
    print(f"  Mechanism:        Targeted dropout → importance → ACTUAL channel deletion")
elif model in ['LightUNet', 'LightDCN', 'LightFD', 'LightDCNFD', 'LightDCNUnet', 'LightFDUnet']:
    print(f"  Base Channels:    {num_chans}")
    print(f"  Use DCN:          {use_dcn}")
    print(f"  Use FDConv:       {use_fdconv}")
    if use_fdconv:
        print(f"  FD Kernel Num:    {fd_kernel_num}")
    variant = ""
    if use_dcn and use_fdconv:
        variant = "DCN + FDConv (Ours)"
    elif use_dcn:
        variant = "DCN only"
    elif use_fdconv:
        variant = "FDConv only"
    else:
        variant = "Baseline (no DCN, no FDConv)"
    print(f"  Variant:          {variant}")
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
