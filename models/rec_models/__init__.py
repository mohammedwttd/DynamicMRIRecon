"""
MRI Reconstruction Models and Layers.

This module provides:
- layers/: Reusable layer components (FDConv, ODConv, SnakeConv)
- models/: Complete reconstruction architectures (UNet variants, ViT, HUMUS)

Note: For DCNv2/DeformConv2d, use torchvision.ops.DeformConv2d directly.
"""

# Re-export layers
from .layers import (
    FDConv,
    DSConv,
    SnakeConvBlock,
    ODConv2d,
)

# Re-export models
from .models import (
    # Base
    UnetModel,
    DynamicUnetModel,
    # Conditional
    CondUnetModel,
    HybridCondUnetModel,
    SmallCondUnetModel,
    # FD
    FDUnetModel,
    LightFDUnet,
    # DCN
    LightDCNUnet,
    # DCN + FD
    ConfigurableUNet,
    FullFDUnet,
    HybridFDUnet,
    DCNFDUnet,
    SmallDCNFDUnet,
    # Hybrid
    HybridSnakeFDUnet,
    # Complex
    ComplexUnetModel,
    # HUMUS
    HUMUSBlock,
    HUMUSNet,
    # ViT
    ReconNet,
)

__all__ = [
    # Layers
    'FDConv',
    'DSConv',
    'SnakeConvBlock',
    'ODConv2d',
    # Models
    'UnetModel',
    'DynamicUnetModel',
    'CondUnetModel',
    'HybridCondUnetModel',
    'SmallCondUnetModel',
    'FDUnetModel',
    'LightFDUnet',
    'LightDCNUnet',
    'ConfigurableUNet',
    'FullFDUnet',
    'HybridFDUnet',
    'DCNFDUnet',
    'SmallDCNFDUnet',
    'HybridSnakeFDUnet',
    'ComplexUnetModel',
    'HUMUSBlock',
    'HUMUSNet',
    'ReconNet',
]
