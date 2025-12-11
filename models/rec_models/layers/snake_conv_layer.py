"""
Dynamic Snake Convolution for Tubular Structure Segmentation.

Based on: "Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation"
ICCV 2023

Snake convolution adaptively adjusts the receptive field to capture tubular structures
(vessels, nerves, boundaries) by learning offsets along two orthogonal directions (X and Y).
This is crucial for MRI reconstruction where preserving anatomical topology is essential.

Key innovations:
1. Iterative offset learning along X and Y axes (mimics snake movement)
2. Morphology-aware kernel adaptation for tubular structures
3. Lightweight compared to full deformable convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DSConv(nn.Module):
    """
    Dynamic Snake Convolution.
    
    Learns deformable offsets specifically designed for tubular structures by
    iteratively refining the kernel position along X and Y axes.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, padding=1, bias=True):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the snake kernel (should be odd, typically 9)
            stride (int): Stride of convolution
            padding (int): Padding added to input
            bias (bool): If True, adds learnable bias
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Standard convolution for initial features
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                             padding=1, bias=bias)
        
        # Offset learning for X-axis snake movement
        self.offset_x = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, kernel_size, kernel_size=1, bias=True)
        )
        
        # Offset learning for Y-axis snake movement
        self.offset_y = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, kernel_size, kernel_size=1, bias=True)
        )
        
        # Deformable convolution along X-axis
        self.dsc_x = DSC_Axis(in_channels, out_channels, kernel_size, axis='x')
        
        # Deformable convolution along Y-axis  
        self.dsc_y = DSC_Axis(in_channels, out_channels, kernel_size, axis='y')
        
        # Fusion layer to combine standard, X-snake, and Y-snake features
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self._initialize_offsets()
    
    def _initialize_offsets(self):
        """Initialize offset layers to output zero initially (identity)."""
        for m in [self.offset_x, self.offset_y]:
            # Zero init the final conv so offsets start at 0
            nn.init.constant_(m[-1].weight, 0)
            nn.init.constant_(m[-1].bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor [B, out_channels, H', W']
        """
        # Standard convolution branch
        out_standard = self.conv(x)
        
        # Learn offsets for snake movement
        offset_x = self.offset_x(x)  # [B, kernel_size, H, W]
        offset_y = self.offset_y(x)  # [B, kernel_size, H, W]
        
        # Apply snake convolution along X and Y axes
        out_x = self.dsc_x(x, offset_x)
        out_y = self.dsc_y(x, offset_y)
        
        # Fuse all three branches
        out = torch.cat([out_standard, out_x, out_y], dim=1)
        out = self.fusion(out)
        
        return out


class DSC_Axis(nn.Module):
    """
    Dynamic Snake Convolution along a single axis (X or Y).
    
    Applies 1D deformable convolution along the specified axis to capture
    tubular structures aligned with that direction.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, axis='x'):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the 1D kernel
            axis (str): Either 'x' or 'y', specifying the axis of snake movement
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.axis = axis
        
        # 1D convolution kernel weights along the axis
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x, offset):
        """
        Args:
            x: Input tensor [B, C, H, W]
            offset: Learned offsets [B, kernel_size, H, W]
        
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        K = self.kernel_size
        
        # Create base grid
        if self.axis == 'x':
            # For X-axis: move horizontally, keep Y fixed
            grid = self._get_x_grid(B, H, W, K, offset, x.device)
        else:
            # For Y-axis: move vertically, keep X fixed
            grid = self._get_y_grid(B, H, W, K, offset, x.device)
        
        # Sample features at offset positions using bilinear interpolation
        # grid: [B, H, W, K, 2] -> reshape to [B, H*W*K, 2]
        grid = grid.view(B, H * W * K, 2)
        
        # x: [B, C, H, W] -> expand for sampling
        # We need to sample K points for each spatial location
        sampled = []
        for k in range(K):
            grid_k = grid[:, k::K, :]  # [B, H*W, 2]
            grid_k = grid_k.view(B, H, W, 2)
            # F.grid_sample expects grid in [-1, 1] range
            sampled_k = F.grid_sample(x, grid_k, mode='bilinear', 
                                     padding_mode='zeros', align_corners=True)
            sampled.append(sampled_k)
        
        sampled = torch.stack(sampled, dim=-1)  # [B, C, H, W, K]
        
        # Apply 1D convolution along the kernel dimension
        # sampled: [B, C, H, W, K], weight: [out_channels, in_channels, K]
        sampled = sampled.permute(0, 2, 3, 1, 4)  # [B, H, W, C, K]
        sampled = sampled.reshape(B * H * W, C, K)
        
        # weight: [out_channels, in_channels, K]
        out = F.conv1d(sampled, self.weight)  # [B*H*W, out_channels, 1]
        out = out.view(B, H, W, self.out_channels).permute(0, 3, 1, 2)  # [B, out_channels, H, W]
        
        return out
    
    def _get_x_grid(self, B, H, W, K, offset, device):
        """
        Generate sampling grid for X-axis snake convolution.
        
        For each pixel (h, w), samples K points horizontally:
        [(w + offset[0], h), (w + offset[1], h), ..., (w + offset[K-1], h)]
        """
        # Base positions
        y_base = torch.arange(0, H, device=device).float().view(1, H, 1, 1).expand(B, H, W, K)
        x_base = torch.arange(0, W, device=device).float().view(1, 1, W, 1).expand(B, H, W, K)
        
        # Add learned offsets along X direction
        # offset: [B, K, H, W] -> [B, H, W, K]
        offset = offset.permute(0, 2, 3, 1)
        
        # Create grid in range [-1, 1] for grid_sample
        x_grid = (x_base + offset) / (W - 1) * 2 - 1  # Normalize to [-1, 1]
        y_grid = y_base / (H - 1) * 2 - 1
        
        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, K, 2]
        return grid
    
    def _get_y_grid(self, B, H, W, K, offset, device):
        """
        Generate sampling grid for Y-axis snake convolution.
        
        For each pixel (h, w), samples K points vertically:
        [(w, h + offset[0]), (w, h + offset[1]), ..., (w, h + offset[K-1])]
        """
        # Base positions
        y_base = torch.arange(0, H, device=device).float().view(1, H, 1, 1).expand(B, H, W, K)
        x_base = torch.arange(0, W, device=device).float().view(1, 1, W, 1).expand(B, H, W, K)
        
        # Add learned offsets along Y direction
        # offset: [B, K, H, W] -> [B, H, W, K]
        offset = offset.permute(0, 2, 3, 1)
        
        # Create grid in range [-1, 1] for grid_sample
        x_grid = x_base / (W - 1) * 2 - 1
        y_grid = (y_base + offset) / (H - 1) * 2 - 1  # Normalize to [-1, 1]
        
        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, K, 2]
        return grid


class SnakeConvBlock(nn.Module):
    """
    Convolutional block using Dynamic Snake Convolution.
    
    Uses Snake convolution for the first conv (topology capture) and
    standard conv for the second (feature refinement).
    """
    
    def __init__(self, in_chans, out_chans, drop_prob, snake_kernel_size=9):
        """
        Args:
            in_chans (int): Number of input channels
            out_chans (int): Number of output channels
            drop_prob (float): Dropout probability
            snake_kernel_size (int): Kernel size for snake convolution
        """
        super().__init__()
        
        # First conv: Snake convolution for topology
        self.conv1 = DSConv(in_chans, out_chans, kernel_size=snake_kernel_size)
        self.norm1 = nn.InstanceNorm2d(out_chans)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(drop_prob)
        
        # Second conv: Standard convolution for refinement
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_chans)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d(drop_prob)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, in_chans, H, W]
        
        Returns:
            Output tensor [B, out_chans, H, W]
        """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        
        return out

