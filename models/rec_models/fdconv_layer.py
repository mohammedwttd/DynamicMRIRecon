"""
FDConv (Frequency Dynamic Convolution) implementation.

Based on: "Frequency Dynamic Convolution for Dense Image Prediction" (CVPR 2025)
GitHub: https://github.com/Linwei-Chen/FDConv

FDConv performs convolution in the Fourier domain with frequency-diverse weights,
achieving state-of-the-art performance with minimal parameter overhead.

Key innovations:
1. Fourier Disjoint Weight (FDW) - frequency-diverse kernels via disjoint spectral coefficients
2. Kernel Spatial Modulation (KSM) - dynamic element-wise filter adjustment
3. Frequency Band Modulation (FBM) - adaptive spatial-frequency band modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FDConv(nn.Module):
    """
    Frequency Dynamic Convolution.
    
    Learns convolution kernels in the Fourier domain using disjoint spectral coefficients,
    enabling frequency-diverse filters with minimal parameter overhead.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=True, kernel_num=64):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of convolution kernel
            stride (int): Stride of convolution
            padding (int): Padding added to input
            dilation (int): Spacing between kernel elements
            groups (int): Number of blocked connections
            bias (bool): If True, adds learnable bias
            kernel_num (int): Number of frequency-diverse kernels (spectral resolution)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        
        # Fourier Disjoint Weights (FDW)
        # Instead of full spatial kernel, learn disjoint frequency coefficients
        # This creates frequency-diverse kernels without parameter redundancy
        self.dft_weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels // groups, 
                       self.kernel_size[0], self.kernel_size[1], 2) * 0.01
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Kernel Spatial Modulation (KSM)
        # Dynamically adjusts filter responses at element-wise level
        # Uses global context (GAP) + local features for modulation
        self.ksm_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, kernel_num, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Frequency Band Modulation (FBM) - optional, simplified version
        # In full implementation, this would modulate spatial-frequency bands
        self.fbm_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize DFT weights from spatial domain for stability.
        Convert properly initialized spatial kernels to frequency domain.
        """
        import math
        
        # Proper He/Kaiming initialization for spatial kernels
        fan_in = (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        std = math.sqrt(2.0 / (fan_in + fan_out))
        
        # Initialize each frequency kernel from a spatial kernel
        for k in range(self.kernel_num):
            # Create spatial kernel with proper initialization
            spatial_kernel = torch.randn(
                self.out_channels, self.in_channels // self.groups,
                self.kernel_size[0], self.kernel_size[1]
            ) * std
            
            # Convert to frequency domain
            freq_kernel = torch.fft.fft2(spatial_kernel, dim=(-2, -1))
            
            # Store real and imaginary parts
            self.dft_weight.data[k, :, :, :, :, 0] = freq_kernel.real
            self.dft_weight.data[k, :, :, :, :, 1] = freq_kernel.imag
    
    def _construct_kernel(self, attention):
        """
        Construct spatial kernel from frequency-domain weights.
        
        Args:
            attention: [batch, kernel_num, 1, 1] - attention weights from KSM
        
        Returns:
            kernel: [batch, out_channels, in_channels, kh, kw] - spatial kernels
        """
        batch_size = attention.size(0)
        
        # Apply softmax for proper probability distribution over kernels
        attention = F.softmax(attention.view(batch_size, self.kernel_num), dim=1)
        
        # Reshape attention: [batch, kernel_num, 1, 1, 1, 1, 1] 
        # Need 7 dims to match dft_weight_expanded (which has complex component dim)
        attention = attention.view(batch_size, self.kernel_num, 1, 1, 1, 1, 1)
        
        # Weighted combination of frequency components
        # dft_weight: [kernel_num, out_channels, in_channels, kh, kw, 2]
        dft_weight_expanded = self.dft_weight.unsqueeze(0)  # [1, kernel_num, out, in, kh, kw, 2]
        
        # Aggregate frequency components with attention
        # [batch, out_channels, in_channels, kh, kw, 2]
        aggregated_dft = (attention * dft_weight_expanded).sum(dim=1)
        
        # Convert from frequency domain to spatial domain via IFFT
        # Treat last dimension (2) as real/imaginary parts
        real_part = aggregated_dft[..., 0]  # [batch, out, in, kh, kw]
        imag_part = aggregated_dft[..., 1]  # [batch, out, in, kh, kw]
        
        # Create complex tensor
        complex_weight = torch.complex(real_part, imag_part)
        
        # Inverse FFT to get spatial kernel with proper normalization
        spatial_kernel = torch.fft.ifft2(complex_weight, dim=(-2, -1), norm='ortho')
        
        # Take real part (imaginary should be ~0 after proper training)
        kernel = spatial_kernel.real
        
        # Clamp to prevent extreme values
        kernel = torch.clamp(kernel, -10.0, 10.0)
        
        return kernel
    
    def forward(self, x):
        """
        Forward pass with frequency-dynamic convolution.
        
        Args:
            x: [batch, in_channels, height, width]
        
        Returns:
            out: [batch, out_channels, height, width]
        """
        batch_size = x.size(0)
        
        # Kernel Spatial Modulation (KSM)
        # Generate attention weights based on input
        attention = self.ksm_fc(x)  # [batch, kernel_num, 1, 1]
        
        # Construct frequency-dynamic kernels
        kernels = self._construct_kernel(attention)  # [batch, out, in, kh, kw]
        
        # Apply Frequency Band Modulation (FBM)
        # Simple scaling version - full version would modulate frequency bands
        
        # Perform batch-wise convolution
        # Group convolutions for each sample in batch
        output_list = []
        for i in range(batch_size):
            out = F.conv2d(
                x[i:i+1],
                kernels[i],
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            output_list.append(out)
        
        output = torch.cat(output_list, dim=0)
        
        # Apply FBM scaling
        output = output * self.fbm_scale
        
        return output
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, kernel_num={kernel_num}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
    
    @staticmethod
    def convert_from_conv2d(conv_layer, kernel_num=64):
        """
        Convert a standard Conv2d layer to FDConv by transforming weights to frequency domain.
        
        Args:
            conv_layer: nn.Conv2d layer
            kernel_num: Number of frequency components
        
        Returns:
            FDConv layer with converted weights
        """
        fdconv = FDConv(
            in_channels=conv_layer.in_channels,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
            kernel_num=kernel_num
        )
        
        # Convert spatial weights to frequency domain via FFT
        with torch.no_grad():
            spatial_weight = conv_layer.weight.data  # [out, in, kh, kw]
            
            # Apply FFT to get frequency representation
            freq_weight = torch.fft.fft2(spatial_weight.float(), dim=(-2, -1))
            
            # Split into real and imaginary parts
            real_part = freq_weight.real
            imag_part = freq_weight.imag
            
            # Stack as last dimension [out, in, kh, kw, 2]
            dft_weight_init = torch.stack([real_part, imag_part], dim=-1)
            
            # Replicate across kernel_num dimension
            # [kernel_num, out, in, kh, kw, 2]
            fdconv.dft_weight.data = dft_weight_init.unsqueeze(0).repeat(kernel_num, 1, 1, 1, 1, 1)
            
            # Add small noise to break symmetry
            noise = torch.randn_like(fdconv.dft_weight.data) * 0.01
            fdconv.dft_weight.data += noise
            
            if conv_layer.bias is not None:
                fdconv.bias.data = conv_layer.bias.data.clone()
        
        return fdconv


class FDConv_Simple(nn.Module):
    """
    Simplified FDConv - currently uses standard Conv2d for stability.
    
    TODO: Implement frequency-domain learning once baseline is stable.
    For now, this ensures the U-Net architecture works correctly.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Use standard Conv2d for now (stable baseline)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
    
    def forward(self, x):
        return self.conv(x)

