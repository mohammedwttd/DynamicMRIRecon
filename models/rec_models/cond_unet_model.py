"""
Conditional U-Net using CondConv (Conditionally Parameterized Convolutions).

Based on:
- CondConv: Conditionally Parameterized Convolutions for Efficient Inference (NeurIPS 2019)
- U-Net: Convolutional networks for biomedical image segmentation (MICCAI 2015)

CondConv allows the network to have multiple expert kernels and dynamically combine them
based on the input, providing more model capacity without linear increase in computation.
"""

import torch
from torch import nn
from torch.nn import functional as F
import math


class CondConv2d(nn.Module):
    """
    Conditionally Parameterized 2D Convolution.
    
    Instead of a single set of weights, maintains multiple expert kernels and
    dynamically combines them based on input-dependent routing weights.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, num_experts=8):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of convolving kernel
            stride (int or tuple): Stride of convolution
            padding (int or tuple): Zero-padding added to both sides of input
            dilation (int or tuple): Spacing between kernel elements
            groups (int): Number of blocked connections from input to output
            bias (bool): If True, adds learnable bias to output
            num_experts (int): Number of expert kernels to maintain
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts
        
        # Expert kernels: [num_experts, out_channels, in_channels, kernel_h, kernel_w]
        self.weight = nn.Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, *self.kernel_size)
        )
        
        if bias:
            # Expert biases: [num_experts, out_channels]
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Routing function: global average pooling + FC layer
        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Sigmoid()  # Normalize routing weights to [0, 1]
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize expert weights using Kaiming initialization."""
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        
        if self.bias is not None:
            for i in range(self.num_experts):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)
    
    def forward(self, x):
        """
        Forward pass with dynamic kernel mixing.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_channels, height, width]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch, out_channels, height, width]
        """
        batch_size = x.size(0)
        
        # Compute routing weights: [batch, num_experts]
        routing_weights = self.routing(x)  # [batch, num_experts]
        
        # Store original routing weights for bias computation
        routing_weights_flat = routing_weights  # [batch, num_experts]
        
        # Aggregate expert kernels based on routing weights
        # weight: [num_experts, out_channels, in_channels, kh, kw]
        # routing_weights: [batch, num_experts]
        
        # For each sample in batch, compute weighted combination of expert kernels
        # This is done by reshaping and using batch matrix multiplication
        
        # Reshape routing weights: [batch, num_experts, 1, 1, 1, 1]
        routing_weights = routing_weights.view(batch_size, self.num_experts, 1, 1, 1, 1)
        
        # Expand weight for batch: [1, num_experts, out_channels, in_channels, kh, kw]
        weight_expanded = self.weight.unsqueeze(0)
        
        # Compute weighted combination: [batch, out_channels, in_channels, kh, kw]
        aggregated_weight = (routing_weights * weight_expanded).sum(dim=1)
        
        if self.bias is not None:
            # Aggregate biases: [batch, out_channels]
            # routing_weights_flat: [batch, num_experts]
            # self.bias: [num_experts, out_channels]
            aggregated_bias = torch.matmul(routing_weights_flat, self.bias)  # [batch, out_channels]
        else:
            aggregated_bias = None
        
        # Apply convolution for each sample in batch
        # This is done using grouped convolution trick
        output_list = []
        for i in range(batch_size):
            out = F.conv2d(
                x[i:i+1], 
                aggregated_weight[i],
                bias=aggregated_bias[i] if aggregated_bias is not None else None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            output_list.append(out)
        
        output = torch.cat(output_list, dim=0)
        return output
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, num_experts={num_experts}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class CondConvBlock(nn.Module):
    """
    A Convolutional Block using CondConv2d.
    
    Consists of two CondConv layers each followed by
    instance normalization, relu activation and dropout.
    """
    
    def __init__(self, in_chans, out_chans, drop_prob, num_experts=8):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
            num_experts (int): Number of expert kernels per CondConv layer.
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.num_experts = num_experts
        
        # First CondConv layer
        self.conv1 = CondConv2d(in_chans, out_chans, kernel_size=3, padding=1, num_experts=num_experts)
        self.norm1 = nn.InstanceNorm2d(out_chans)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout2d(drop_prob)
        
        # Second CondConv layer
        self.conv2 = CondConv2d(out_chans, out_chans, kernel_size=3, padding=1, num_experts=num_experts)
        self.norm2 = nn.InstanceNorm2d(out_chans)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout2d(drop_prob)
    
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output = self.conv1(input)
        output = self.norm1(output)
        output = self.relu1(output)
        output = self.drop1(output)
        
        output = self.conv2(output)
        output = self.norm2(output)
        output = self.relu2(output)
        output = self.drop2(output)
        
        return output
    
    def __repr__(self):
        return f'CondConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob}, num_experts={self.num_experts})'


class CondUnetModel(nn.Module):
    """
    U-Net with Conditionally Parameterized Convolutions.
    
    Each convolutional layer uses CondConv, which maintains multiple expert kernels
    and dynamically combines them based on the input. This provides more model
    capacity and adaptability without a linear increase in parameters.
    """
    
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, num_experts=8):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
            num_experts (int): Number of expert kernels per CondConv layer (default: 8).
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.num_experts = num_experts
        
        # Encoder (down-sampling path)
        self.down_sample_layers = nn.ModuleList([
            CondConvBlock(in_chans, chans, drop_prob, num_experts)
        ])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [CondConvBlock(ch, ch * 2, drop_prob, num_experts)]
            ch *= 2
        
        # Bottleneck
        self.conv = CondConvBlock(ch, ch, drop_prob, num_experts)
        
        # Decoder (up-sampling path)
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [CondConvBlock(ch * 2, ch // 2, drop_prob, num_experts)]
            ch //= 2
        self.up_sample_layers += [CondConvBlock(ch * 2, ch, drop_prob, num_experts)]
        
        # Final output layers (using regular Conv2d for efficiency)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
    
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        
        # Encoder: Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        
        # Bottleneck
        output = self.conv(output)
        
        # Decoder: Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        
        return self.conv2(output)
    
    def get_routing_weights(self, input):
        """
        Get routing weights for all CondConv layers (for analysis/visualization).
        
        Args:
            input (torch.Tensor): Input tensor
        
        Returns:
            dict: Dictionary mapping layer names to routing weights
        """
        routing_weights = {}
        
        # Helper to extract routing weights from a CondConvBlock
        def get_block_routing(block, x, prefix):
            with torch.no_grad():
                w1 = block.conv1.routing(x)
                routing_weights[f'{prefix}_conv1'] = w1.detach().cpu()
                
                # Pass through first conv to get input for second
                x_mid = block.conv1(x)
                x_mid = block.norm1(x_mid)
                x_mid = block.relu1(x_mid)
                
                w2 = block.conv2.routing(x_mid)
                routing_weights[f'{prefix}_conv2'] = w2.detach().cpu()
            
            return x_mid
        
        with torch.no_grad():
            output = input
            
            # Encoder
            for i, layer in enumerate(self.down_sample_layers):
                get_block_routing(layer, output, f'down_{i}')
                output = layer(output)
                output = F.max_pool2d(output, kernel_size=2)
            
            # Bottleneck
            get_block_routing(self.conv, output, 'bottleneck')
        
        return routing_weights
    
    def count_parameters(self):
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return f'CondUnetModel(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'chans={self.chans}, num_pool_layers={self.num_pool_layers}, ' \
            f'drop_prob={self.drop_prob}, num_experts={self.num_experts})'

