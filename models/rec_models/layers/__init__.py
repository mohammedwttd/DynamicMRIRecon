"""
Reusable layer components for MRI reconstruction models.

Note: Some layers require lazy imports due to optional dependencies.
"""

from .fdconv_layer import FDConv
from .snake_conv_layer import DSConv, SnakeConvBlock
from .ODConv_layer import ODConv2d, Attention

# DCNv2 is available via torchvision.ops.DeformConv2d
# The custom dcnv2_module.py requires compilation and is not imported by default
# Use: from torchvision.ops import DeformConv2d

__all__ = [
    'FDConv',
    'DSConv',
    'SnakeConvBlock',
    'ODConv2d',
    'Attention',
]

