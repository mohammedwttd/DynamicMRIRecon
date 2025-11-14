"""
Dynamic U-Net with GradMax-inspired growth mechanisms.

Supports two growth strategies:
1. Sample-and-Select: Add N candidate filters, pick the one with highest gradient
2. Gradient-Informed Splitting: Duplicate the filter with largest gradient, add noise
"""

import torch
from torch import nn
from torch.nn import functional as F
import copy
from typing import Tuple, Optional, List


class DynamicConvBlock(nn.Module):
    """
    A Convolutional Block that can dynamically grow during training.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        # Store layers separately for easier manipulation during growth
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_chans)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout2d(drop_prob)
        
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
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

    def get_growable_layers(self):
        """Returns list of conv layers that can be grown."""
        return [self.conv1, self.conv2]

    def __repr__(self):
        return f'DynamicConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class DynamicUnetModel(nn.Module):
    """
    Dynamic U-Net that can grow during training using GradMax-inspired methods.
    
    Growth methods:
    - 'sample_select': Sample N candidates, select the one with highest gradient
    - 'split': Duplicate the filter with highest gradient and add noise
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, 
                 growth_method='sample_select', n_candidates=10):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
            growth_method (str): 'sample_select' or 'split'
            n_candidates (int): Number of candidate filters to try (for sample_select)
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.growth_method = growth_method
        self.n_candidates = n_candidates

        self.down_sample_layers = nn.ModuleList([DynamicConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [DynamicConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = DynamicConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [DynamicConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [DynamicConvBlock(ch * 2, ch, drop_prob)]
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
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)

    def grow_layer_sample_select(self, layer_idx: str, conv_idx: int, batch_data: torch.Tensor,
                                   target_data: torch.Tensor, criterion) -> Tuple[torch.Tensor, int]:
        """
        Grow a layer using Sample-and-Select method.
        
        Args:
            layer_idx (str): Layer identifier ('bottleneck', 'down_0', 'up_0', etc.)
            conv_idx (int): Which conv in the block (0 or 1)
            batch_data (torch.Tensor): Input batch for gradient calculation
            target_data (torch.Tensor): Target batch for loss calculation
            criterion: Loss function
            
        Returns:
            winner_weights (torch.Tensor): Weights of the winning filter
            winner_idx (int): Index of the winning candidate
        """
        # Get the target layer
        if layer_idx == 'bottleneck':
            block = self.conv
        elif layer_idx.startswith('down_'):
            idx = int(layer_idx.split('_')[1])
            block = self.down_sample_layers[idx]
        elif layer_idx.startswith('up_'):
            idx = int(layer_idx.split('_')[1])
            block = self.up_sample_layers[idx]
        else:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")

        # Get the specific conv layer
        conv_layer = block.conv1 if conv_idx == 0 else block.conv2
        
        # Store original state
        original_out_channels = conv_layer.out_channels
        original_weight = conv_layer.weight.data.clone()
        original_bias = conv_layer.bias.data.clone() if conv_layer.bias is not None else None
        
        # Create N candidate filters
        candidates = []
        for _ in range(self.n_candidates):
            # Initialize new filter with Kaiming initialization
            new_filter = torch.zeros(1, conv_layer.in_channels, 
                                    conv_layer.kernel_size[0], conv_layer.kernel_size[1],
                                    device=conv_layer.weight.device)
            nn.init.kaiming_normal_(new_filter, mode='fan_out', nonlinearity='relu')
            candidates.append(new_filter)
        
        # Test each candidate
        gradient_norms = []
        
        for candidate in candidates:
            # Temporarily add the candidate
            new_weight = torch.cat([original_weight, candidate], dim=0)
            conv_layer.out_channels = original_out_channels + 1
            conv_layer.weight = nn.Parameter(new_weight)
            
            if original_bias is not None:
                new_bias_val = torch.zeros(1, device=conv_layer.bias.device)
                new_bias = torch.cat([original_bias, new_bias_val], dim=0)
                conv_layer.bias = nn.Parameter(new_bias)
            
            # Temporarily update the consuming layer for silent addition
            consuming_layer_info = self._get_consuming_layer_info(layer_idx, conv_idx)
            original_next_weight = None
            
            if consuming_layer_info is not None:
                consuming_block, consuming_attr = consuming_layer_info
                consuming_conv = getattr(consuming_block, consuming_attr)
                original_next_weight = consuming_conv.weight.data.clone()
                
                # Add one zero-initialized input channel
                new_input_channel = torch.zeros(
                    consuming_conv.out_channels, 1,
                    consuming_conv.kernel_size[0], consuming_conv.kernel_size[1],
                    device=consuming_conv.weight.device
                )
                temp_weight = torch.cat([consuming_conv.weight.data, new_input_channel], dim=1)
                consuming_conv.weight.data = temp_weight
                consuming_conv.in_channels += 1
            
            # Forward and backward pass
            self.zero_grad()
            output = self.forward(batch_data)
            loss = criterion(output, target_data)
            loss.backward()
            
            # Calculate gradient norm for this candidate
            if conv_layer.weight.grad is not None:
                grad_norm = torch.norm(conv_layer.weight.grad[-1]).item()
            else:
                grad_norm = 0
            gradient_norms.append(grad_norm)
            
            # Restore consuming layer
            if consuming_layer_info is not None and original_next_weight is not None:
                consuming_block, consuming_attr = consuming_layer_info
                consuming_conv = getattr(consuming_block, consuming_attr)
                consuming_conv.weight.data = original_next_weight
                consuming_conv.in_channels -= 1
            
            # Restore original state for next candidate
            conv_layer.out_channels = original_out_channels
            conv_layer.weight = nn.Parameter(original_weight.clone())
            if original_bias is not None:
                conv_layer.bias = nn.Parameter(original_bias.clone())
        
        # Select winner
        winner_idx = gradient_norms.index(max(gradient_norms))
        winner_weights = candidates[winner_idx]
        
        return winner_weights, winner_idx

    def grow_layer_split(self, layer_idx: str, conv_idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Grow a layer using Gradient-Informed Splitting.
        
        Args:
            layer_idx (str): Layer identifier
            conv_idx (int): Which conv in the block (0 or 1)
            
        Returns:
            split_filter_idx (int): Index of the filter that was split
            new_filter1 (torch.Tensor): First half of split
            new_filter2 (torch.Tensor): Second half of split (with noise)
        """
        # Get the target layer
        if layer_idx == 'bottleneck':
            block = self.conv
        elif layer_idx.startswith('down_'):
            idx = int(layer_idx.split('_')[1])
            block = self.down_sample_layers[idx]
        elif layer_idx.startswith('up_'):
            idx = int(layer_idx.split('_')[1])
            block = self.up_sample_layers[idx]
        else:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")

        conv_layer = block.conv1 if conv_idx == 0 else block.conv2
        
        # Find filter with largest gradient norm
        if conv_layer.weight.grad is None:
            raise ValueError("No gradients available. Run a forward-backward pass first.")
        
        grad_norms = torch.norm(conv_layer.weight.grad.view(conv_layer.out_channels, -1), dim=1)
        split_filter_idx = torch.argmax(grad_norms).item()
        
        # Get the filter to split
        filter_to_split = conv_layer.weight.data[split_filter_idx:split_filter_idx+1].clone()
        
        # Create two copies
        new_filter1 = filter_to_split.clone()
        new_filter2 = filter_to_split.clone()
        
        # Add small noise to break symmetry
        noise = torch.randn_like(new_filter2) * 0.01
        new_filter2 += noise
        
        return split_filter_idx, new_filter1, new_filter2

    def apply_growth(self, layer_idx: str, conv_idx: int, new_filter: torch.Tensor,
                     split_mode: bool = False):
        """
        Apply the growth by permanently adding the new filter(s).
        
        Args:
            layer_idx (str): Layer identifier
            conv_idx (int): Which conv in the block (0 or 1)
            new_filter (torch.Tensor): New filter(s) to add
            split_mode (bool): If True, halve the next layer weights for the split filters
        """
        # Get the target block
        if layer_idx == 'bottleneck':
            block = self.conv
        elif layer_idx.startswith('down_'):
            idx = int(layer_idx.split('_')[1])
            block = self.down_sample_layers[idx]
        elif layer_idx.startswith('up_'):
            idx = int(layer_idx.split('_')[1])
            block = self.up_sample_layers[idx]
        else:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")

        conv_layer = block.conv1 if conv_idx == 0 else block.conv2
        
        # Add new filter(s)
        old_out_channels = conv_layer.out_channels
        new_out_channels = old_out_channels + new_filter.size(0)
        n_new_filters = new_filter.size(0)
        
        # Concatenate weights
        new_weight = torch.cat([conv_layer.weight.data, new_filter], dim=0)
        
        # Update bias
        if conv_layer.bias is not None:
            new_bias_vals = torch.zeros(n_new_filters, device=conv_layer.bias.device)
            new_bias = torch.cat([conv_layer.bias.data, new_bias_vals], dim=0)
        else:
            new_bias = None
        
        # Create new conv layer with updated size
        new_conv = nn.Conv2d(
            conv_layer.in_channels,
            new_out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=(conv_layer.bias is not None)
        ).to(conv_layer.weight.device)
        
        new_conv.weight = nn.Parameter(new_weight)
        if new_bias is not None:
            new_conv.bias = nn.Parameter(new_bias)
        
        # Replace the layer
        if conv_idx == 0:
            block.conv1 = new_conv
        else:
            block.conv2 = new_conv
        
        # Update normalization layers
        if conv_idx == 0:
            block.norm1 = nn.InstanceNorm2d(new_out_channels).to(conv_layer.weight.device)
        else:
            block.norm2 = nn.InstanceNorm2d(new_out_channels).to(conv_layer.weight.device)
        
        # Update tracked dimensions
        block.out_chans = new_out_channels
        
        # Update the next layer that consumes this output
        self._update_consuming_layer(layer_idx, conv_idx, n_new_filters, split_mode)
        
    def _get_consuming_layer_info(self, layer_idx: str, conv_idx: int):
        """
        Get information about which layer consumes the output of the specified layer.
        
        Returns:
            Tuple of (consuming_block, consuming_conv_attr) or None
        """
        if conv_idx == 0:
            # If we grew conv1, conv2 in the same block consumes it
            if layer_idx == 'bottleneck':
                return (self.conv, 'conv2')
            elif layer_idx.startswith('down_'):
                idx = int(layer_idx.split('_')[1])
                return (self.down_sample_layers[idx], 'conv2')
            elif layer_idx.startswith('up_'):
                idx = int(layer_idx.split('_')[1])
                return (self.up_sample_layers[idx], 'conv2')
        else:
            # If we grew conv2, the next block consumes it
            if layer_idx == 'bottleneck':
                return (self.up_sample_layers[0], 'conv1')
            elif layer_idx.startswith('down_'):
                idx = int(layer_idx.split('_')[1])
                if idx < len(self.down_sample_layers) - 1:
                    return (self.down_sample_layers[idx + 1], 'conv1')
                else:
                    return (self.conv, 'conv1')
            elif layer_idx.startswith('up_'):
                idx = int(layer_idx.split('_')[1])
                if idx < len(self.up_sample_layers) - 1:
                    return (self.up_sample_layers[idx + 1], 'conv1')
                else:
                    # Last up layer feeds into final conv - special case
                    return None
        return None
    
    def _update_consuming_layer(self, layer_idx: str, conv_idx: int, n_new_filters: int, split_mode: bool):
        """Update the layer that consumes the output of the grown layer."""
        
        # Determine which layer consumes this output
        if conv_idx == 0:
            # If we grew conv1, conv2 in the same block consumes it
            if layer_idx == 'bottleneck':
                consuming_block = self.conv
                consuming_conv_attr = 'conv2'
            elif layer_idx.startswith('down_'):
                idx = int(layer_idx.split('_')[1])
                consuming_block = self.down_sample_layers[idx]
                consuming_conv_attr = 'conv2'
            elif layer_idx.startswith('up_'):
                idx = int(layer_idx.split('_')[1])
                consuming_block = self.up_sample_layers[idx]
                consuming_conv_attr = 'conv2'
            else:
                return
        else:
            # If we grew conv2, the next block consumes it
            if layer_idx == 'bottleneck':
                # Bottleneck conv2 feeds into first upsampling layer
                consuming_block = self.up_sample_layers[0]
                consuming_conv_attr = 'conv1'
            elif layer_idx.startswith('down_'):
                idx = int(layer_idx.split('_')[1])
                if idx < len(self.down_sample_layers) - 1:
                    # Feeds into next downsampling layer
                    consuming_block = self.down_sample_layers[idx + 1]
                    consuming_conv_attr = 'conv1'
                else:
                    # Last downsampling layer feeds into bottleneck
                    consuming_block = self.conv
                    consuming_conv_attr = 'conv1'
            elif layer_idx.startswith('up_'):
                idx = int(layer_idx.split('_')[1])
                if idx < len(self.up_sample_layers) - 1:
                    # Feeds into next upsampling layer
                    consuming_block = self.up_sample_layers[idx + 1]
                    consuming_conv_attr = 'conv1'
                else:
                    # Last upsampling layer feeds into final conv
                    # For simplicity, we'll handle this separately
                    # The conv2 sequential layers need updating
                    self._update_final_layers(n_new_filters)
                    return
            else:
                return
        
        # Get the consuming conv layer
        consuming_conv = getattr(consuming_block, consuming_conv_attr)
        old_weight = consuming_conv.weight.data
        
        # Create new input channels (zero-initialized for silent addition)
        new_weight_part = torch.zeros(
            consuming_conv.out_channels, n_new_filters,
            consuming_conv.kernel_size[0], consuming_conv.kernel_size[1],
            device=old_weight.device
        )
        
        # Concatenate along input channel dimension
        new_weight = torch.cat([old_weight, new_weight_part], dim=1)
        
        # Create new conv layer
        new_conv = nn.Conv2d(
            consuming_conv.in_channels + n_new_filters,
            consuming_conv.out_channels,
            kernel_size=consuming_conv.kernel_size,
            stride=consuming_conv.stride,
            padding=consuming_conv.padding,
            bias=(consuming_conv.bias is not None)
        ).to(old_weight.device)
        
        new_conv.weight = nn.Parameter(new_weight)
        if consuming_conv.bias is not None:
            new_conv.bias = nn.Parameter(consuming_conv.bias.data.clone())
        
        # Replace the layer
        setattr(consuming_block, consuming_conv_attr, new_conv)
        
        # Update the norm layer if it's norm1 (since we updated conv1)
        if consuming_conv_attr == 'conv1':
            # We don't need to update norm1 since it takes conv1's output, not input
            pass
    
    def _update_final_layers(self, n_new_filters: int):
        """Update the final conv2 sequential layers when last up layer grows."""
        # The conv2 Sequential consists of three Conv2d layers
        # We only need to update the first one which takes the upsampled features
        first_conv = self.conv2[0]
        old_weight = first_conv.weight.data
        
        # Add new input channels (zero-initialized)
        new_weight_part = torch.zeros(
            first_conv.out_channels, n_new_filters,
            first_conv.kernel_size[0], first_conv.kernel_size[1],
            device=old_weight.device
        )
        
        new_weight = torch.cat([old_weight, new_weight_part], dim=1)
        
        # Create new conv
        new_conv = nn.Conv2d(
            first_conv.in_channels + n_new_filters,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None)
        ).to(old_weight.device)
        
        new_conv.weight = nn.Parameter(new_weight)
        if first_conv.bias is not None:
            new_conv.bias = nn.Parameter(first_conv.bias.data.clone())
        
        # Replace in Sequential
        self.conv2[0] = new_conv

    def grow_network(self, layer_idx: str, conv_idx: int, batch_data: torch.Tensor,
                     target_data: torch.Tensor, criterion):
        """
        Grow the network by adding one filter to the specified layer.
        
        Args:
            layer_idx (str): Layer identifier
            conv_idx (int): Which conv in the block (0 or 1)
            batch_data (torch.Tensor): Input batch for gradient calculation
            target_data (torch.Tensor): Target batch for loss calculation
            criterion: Loss function
        """
        if self.growth_method == 'sample_select':
            winner_weights, winner_idx = self.grow_layer_sample_select(
                layer_idx, conv_idx, batch_data, target_data, criterion
            )
            print(f"Growing {layer_idx} conv{conv_idx}: Selected candidate {winner_idx}")
            self.apply_growth(layer_idx, conv_idx, winner_weights, split_mode=False)
            
        elif self.growth_method == 'split':
            split_idx, filter1, filter2 = self.grow_layer_split(layer_idx, conv_idx)
            print(f"Growing {layer_idx} conv{conv_idx}: Split filter {split_idx}")
            new_filters = torch.cat([filter1, filter2], dim=0)
            self.apply_growth(layer_idx, conv_idx, new_filters, split_mode=True)
        
        else:
            raise ValueError(f"Unknown growth method: {self.growth_method}")

    def get_network_size(self) -> dict:
        """Get current size of the network (number of channels per layer)."""
        sizes = {}
        for i, layer in enumerate(self.down_sample_layers):
            sizes[f'down_{i}'] = {
                'conv1_out': layer.conv1.out_channels,
                'conv2_out': layer.conv2.out_channels
            }
        sizes['bottleneck'] = {
            'conv1_out': self.conv.conv1.out_channels,
            'conv2_out': self.conv.conv2.out_channels
        }
        for i, layer in enumerate(self.up_sample_layers):
            sizes[f'up_{i}'] = {
                'conv1_out': layer.conv1.out_channels,
                'conv2_out': layer.conv2.out_channels
            }
        return sizes

