"""
Simplified Dynamic U-Net that swaps channels between layers.

Every N batches:
- Removes one channel from a layer with low gradient magnitude
- Adds one channel to another layer (gradient-informed initialization)
- Keeps total parameter count constant
"""

import torch
from torch import nn
from torch.nn import functional as F
import random
from typing import Dict, List, Tuple


class DynamicConvBlock(nn.Module):
    """
    A Convolutional Block that can dynamically grow/shrink during training.
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

        # Store layers separately for easier manipulation
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

    def __repr__(self):
        return f'DynamicConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class DynamicUnetModel(nn.Module):
    """
    Simplified Dynamic U-Net that swaps channels between layers every N batches.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, 
                 swap_frequency=10):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
            swap_frequency (int): Perform channel swap every N batches (default: 10)
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.swap_frequency = swap_frequency
        
        # Track batch counter
        self.batch_counter = 0
        
        # Build U-Net architecture
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

    def get_swappable_layers(self) -> List[Tuple[str, nn.Module, str]]:
        """
        Get list of conv layers that can participate in channel swapping.
        
        Skip connections make some layers unsuitable for swapping:
        - Encoder conv2 outputs feed into decoder via skip connections
        - Decoder conv1 inputs receive concatenated features
        
        Safe to swap:
        - Encoder conv1 (only affects conv2 in same block)
        - Bottleneck (both conv1 and conv2)
        - Decoder conv2 (only affects next decoder block)
        
        Returns:
            List of tuples: (layer_id, block, conv_name)
        """
        layers = []
        
        # Encoder layers - ONLY conv1 (conv2 feeds skip connections)
        for i, block in enumerate(self.down_sample_layers):
            layers.append((f'down_{i}', block, 'conv1'))
            # Skip conv2 to avoid skip connection issues
        
        # Bottleneck - safe to swap both
        layers.append(('bottleneck', self.conv, 'conv1'))
        layers.append(('bottleneck', self.conv, 'conv2'))
        
        # Decoder layers - ONLY conv2 (conv1 receives concatenated input)
        for i, block in enumerate(self.up_sample_layers):
            # Skip conv1 to avoid concatenation issues
            layers.append((f'up_{i}', block, 'conv2'))
        
        return layers

    def find_channel_to_drop(self, layer_id: str, block: nn.Module, conv_name: str) -> int:
        """
        Find the channel with lowest gradient magnitude to drop.
        
        Args:
            layer_id (str): Layer identifier
            block (nn.Module): The conv block
            conv_name (str): 'conv1' or 'conv2'
            
        Returns:
            int: Index of channel to drop
        """
        conv_layer = getattr(block, conv_name)
        
        # Check if gradients exist
        if conv_layer.weight.grad is None:
            # Random selection if no gradient
            return random.randint(0, conv_layer.out_channels - 1)
        
        # Calculate gradient magnitude for each output channel (filter)
        grad_norms = torch.norm(conv_layer.weight.grad.view(conv_layer.out_channels, -1), dim=1)
        
        # Return index of channel with minimum gradient
        return torch.argmin(grad_norms).item()

    def find_channel_to_add(self, layer_id: str, block: nn.Module, conv_name: str) -> torch.Tensor:
        """
        Create a new channel using gradient-informed initialization.
        Similar to GradMax: find filter with highest gradient and add noise to it.
        
        Args:
            layer_id (str): Layer identifier
            block (nn.Module): The conv block
            conv_name (str): 'conv1' or 'conv2'
            
        Returns:
            torch.Tensor: New filter weights
        """
        conv_layer = getattr(block, conv_name)
        
        if conv_layer.weight.grad is None:
            # Random initialization if no gradient
            new_filter = torch.zeros(1, conv_layer.in_channels, 
                                    conv_layer.kernel_size[0], conv_layer.kernel_size[1],
                                    device=conv_layer.weight.device)
            nn.init.kaiming_normal_(new_filter, mode='fan_out', nonlinearity='relu')
            return new_filter
        
        # Find filter with largest gradient (most important)
        grad_norms = torch.norm(conv_layer.weight.grad.view(conv_layer.out_channels, -1), dim=1)
        best_idx = torch.argmax(grad_norms).item()
        
        # Copy the best filter and add noise
        new_filter = conv_layer.weight.data[best_idx:best_idx+1].clone()
        noise = torch.randn_like(new_filter) * 0.01
        new_filter = new_filter + noise
        
        return new_filter

    def drop_channel(self, layer_id: str, block: nn.Module, conv_name: str, channel_idx: int):
        """
        Drop a channel from a conv layer.
        
        Args:
            layer_id (str): Layer identifier
            block (nn.Module): The conv block
            conv_name (str): 'conv1' or 'conv2'
            channel_idx (int): Index of channel to drop
        """
        conv_layer = getattr(block, conv_name)
        
        # Create mask to keep all channels except the one to drop
        keep_mask = torch.ones(conv_layer.out_channels, dtype=torch.bool)
        keep_mask[channel_idx] = False
        
        # Create new smaller conv layer
        new_out_channels = conv_layer.out_channels - 1
        new_conv = nn.Conv2d(
            conv_layer.in_channels,
            new_out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=(conv_layer.bias is not None)
        ).to(conv_layer.weight.device)
        
        # Copy weights (excluding dropped channel)
        new_conv.weight.data = conv_layer.weight.data[keep_mask]
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data[keep_mask]
        
        # Replace layer
        setattr(block, conv_name, new_conv)
        
        # Update normalization layer
        norm_name = conv_name.replace('conv', 'norm')
        new_norm = nn.InstanceNorm2d(new_out_channels).to(conv_layer.weight.device)
        setattr(block, norm_name, new_norm)
        
        # Update the layer that consumes this output
        self._update_consuming_layer_drop(layer_id, conv_name, channel_idx)
        
        # Update block's tracked channels
        block.out_chans = new_out_channels

    def add_channel(self, layer_id: str, block: nn.Module, conv_name: str, new_filter: torch.Tensor):
        """
        Add a channel to a conv layer.
        
        Args:
            layer_id (str): Layer identifier
            block (nn.Module): The conv block
            conv_name (str): 'conv1' or 'conv2'
            new_filter (torch.Tensor): New filter weights to add
        """
        conv_layer = getattr(block, conv_name)
        
        # Create new larger conv layer
        new_out_channels = conv_layer.out_channels + 1
        new_conv = nn.Conv2d(
            conv_layer.in_channels,
            new_out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=(conv_layer.bias is not None)
        ).to(conv_layer.weight.device)
        
        # Copy old weights and append new filter
        new_conv.weight.data = torch.cat([conv_layer.weight.data, new_filter], dim=0)
        
        if conv_layer.bias is not None:
            new_bias_val = torch.zeros(1, device=conv_layer.bias.device)
            new_conv.bias.data = torch.cat([conv_layer.bias.data, new_bias_val], dim=0)
        
        # Replace layer
        setattr(block, conv_name, new_conv)
        
        # Update normalization layer
        norm_name = conv_name.replace('conv', 'norm')
        new_norm = nn.InstanceNorm2d(new_out_channels).to(conv_layer.weight.device)
        setattr(block, norm_name, new_norm)
        
        # Update the layer that consumes this output
        self._update_consuming_layer_add(layer_id, conv_name)
        
        # Update block's tracked channels
        block.out_chans = new_out_channels

    def _update_consuming_layer_drop(self, layer_id: str, conv_name: str, dropped_idx: int):
        """Update the consuming layer when a channel is dropped."""
        consuming_info = self._get_consuming_layer(layer_id, conv_name)
        if consuming_info is None:
            # Special case: last up layer feeds into final conv2 Sequential
            if layer_id.startswith('up_') and conv_name == 'conv2':
                idx = int(layer_id.split('_')[1])
                if idx == len(self.up_sample_layers) - 1:
                    # Update the first layer in self.conv2 Sequential
                    self._update_final_layers_drop(dropped_idx)
            return
        
        consuming_block, consuming_conv_name = consuming_info
        consuming_conv = getattr(consuming_block, consuming_conv_name)
        
        # Create mask to keep all input channels except the dropped one
        keep_mask = torch.ones(consuming_conv.in_channels, dtype=torch.bool)
        keep_mask[dropped_idx] = False
        
        # Create new conv with one less input channel
        new_conv = nn.Conv2d(
            consuming_conv.in_channels - 1,
            consuming_conv.out_channels,
            kernel_size=consuming_conv.kernel_size,
            stride=consuming_conv.stride,
            padding=consuming_conv.padding,
            bias=(consuming_conv.bias is not None)
        ).to(consuming_conv.weight.device)
        
        # Copy weights (excluding the input channel that was dropped)
        new_conv.weight.data = consuming_conv.weight.data[:, keep_mask, :, :]
        if consuming_conv.bias is not None:
            new_conv.bias.data = consuming_conv.bias.data.clone()
        
        # Replace layer
        setattr(consuming_block, consuming_conv_name, new_conv)

    def _update_consuming_layer_add(self, layer_id: str, conv_name: str):
        """Update the consuming layer when a channel is added (zero-initialized for silent addition)."""
        consuming_info = self._get_consuming_layer(layer_id, conv_name)
        if consuming_info is None:
            # Special case: last up layer feeds into final conv2 Sequential
            if layer_id.startswith('up_') and conv_name == 'conv2':
                idx = int(layer_id.split('_')[1])
                if idx == len(self.up_sample_layers) - 1:
                    # Update the first layer in self.conv2 Sequential
                    self._update_final_layers_add()
            return
        
        consuming_block, consuming_conv_name = consuming_info
        consuming_conv = getattr(consuming_block, consuming_conv_name)
        
        # Create new conv with one more input channel
        new_conv = nn.Conv2d(
            consuming_conv.in_channels + 1,
            consuming_conv.out_channels,
            kernel_size=consuming_conv.kernel_size,
            stride=consuming_conv.stride,
            padding=consuming_conv.padding,
            bias=(consuming_conv.bias is not None)
        ).to(consuming_conv.weight.device)
        
        # Copy old weights and append zero-initialized weights for new input channel
        zero_channel = torch.zeros(
            consuming_conv.out_channels, 1,
            consuming_conv.kernel_size[0], consuming_conv.kernel_size[1],
            device=consuming_conv.weight.device
        )
        new_conv.weight.data = torch.cat([consuming_conv.weight.data, zero_channel], dim=1)
        
        if consuming_conv.bias is not None:
            new_conv.bias.data = consuming_conv.bias.data.clone()
        
        # Replace layer
        setattr(consuming_block, consuming_conv_name, new_conv)

    def _update_final_layers_add(self):
        """Update the final conv2 Sequential layers when a channel is added to the last up layer."""
        # The conv2 Sequential consists of three Conv2d layers
        # We only need to update the first one which takes the upsampled features
        first_conv = self.conv2[0]
        old_weight = first_conv.weight.data
        
        # Add new input channel (zero-initialized for silent addition)
        new_weight_part = torch.zeros(
            first_conv.out_channels, 1,
            first_conv.kernel_size[0], first_conv.kernel_size[1],
            device=old_weight.device
        )
        
        new_weight = torch.cat([old_weight, new_weight_part], dim=1)
        
        # Create new conv
        new_conv = nn.Conv2d(
            first_conv.in_channels + 1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None)
        ).to(old_weight.device)
        
        new_conv.weight.data = new_weight
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()
        
        # Replace in Sequential
        self.conv2[0] = new_conv
    
    def _update_final_layers_drop(self, dropped_idx: int):
        """Update the final conv2 Sequential layers when a channel is dropped from the last up layer."""
        first_conv = self.conv2[0]
        old_weight = first_conv.weight.data
        
        # Create mask to keep all input channels except the dropped one
        keep_mask = torch.ones(first_conv.in_channels, dtype=torch.bool)
        keep_mask[dropped_idx] = False
        
        # Create new conv with one less input channel
        new_conv = nn.Conv2d(
            first_conv.in_channels - 1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None)
        ).to(old_weight.device)
        
        # Copy weights (excluding dropped input channel)
        new_conv.weight.data = old_weight[:, keep_mask, :, :]
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()
        
        # Replace in Sequential
        self.conv2[0] = new_conv

    def _get_consuming_layer(self, layer_id: str, conv_name: str) -> Tuple[nn.Module, str]:
        """
        Get the layer that consumes the output of the specified layer.
        
        Returns:
            Tuple of (consuming_block, consuming_conv_name) or None
        """
        if conv_name == 'conv1':
            # conv2 in the same block consumes conv1
            if layer_id == 'bottleneck':
                return (self.conv, 'conv2')
            elif layer_id.startswith('down_'):
                idx = int(layer_id.split('_')[1])
                return (self.down_sample_layers[idx], 'conv2')
            elif layer_id.startswith('up_'):
                idx = int(layer_id.split('_')[1])
                return (self.up_sample_layers[idx], 'conv2')
        
        elif conv_name == 'conv2':
            # Next block's conv1 consumes conv2
            if layer_id == 'bottleneck':
                return (self.up_sample_layers[0], 'conv1')
            elif layer_id.startswith('down_'):
                idx = int(layer_id.split('_')[1])
                if idx < len(self.down_sample_layers) - 1:
                    return (self.down_sample_layers[idx + 1], 'conv1')
                else:
                    return (self.conv, 'conv1')
            elif layer_id.startswith('up_'):
                idx = int(layer_id.split('_')[1])
                if idx < len(self.up_sample_layers) - 1:
                    return (self.up_sample_layers[idx + 1], 'conv1')
                else:
                    # Last up layer - special case, would need to update final conv
                    return None
        
        return None

    def perform_channel_swap(self):
        """
        Perform one channel swap: drop from one layer, add to another.
        This keeps the total parameter count approximately constant.
        """
        swappable_layers = self.get_swappable_layers()
        
        if len(swappable_layers) < 2:
            print("Not enough layers to swap")
            return
        
        # Randomly select two different layers
        drop_layer_info = random.choice(swappable_layers)
        add_layer_info = random.choice([l for l in swappable_layers if l != drop_layer_info])
        
        drop_id, drop_block, drop_conv = drop_layer_info
        add_id, add_block, add_conv = add_layer_info
        
        drop_conv_layer = getattr(drop_block, drop_conv)
        add_conv_layer = getattr(add_block, add_conv)
        
        # Make sure we don't drop below minimum channels
        if drop_conv_layer.out_channels <= 2:
            print(f"Skipping swap: {drop_id}.{drop_conv} has too few channels ({drop_conv_layer.out_channels})")
            return
        
        # Find channel to drop (lowest gradient)
        channel_to_drop = self.find_channel_to_drop(drop_id, drop_block, drop_conv)
        
        # Find channel to add (gradient-informed)
        new_filter = self.find_channel_to_add(add_id, add_block, add_conv)
        
        print(f"\n=== Channel Swap ===")
        print(f"Dropping channel {channel_to_drop} from {drop_id}.{drop_conv} "
              f"({drop_conv_layer.out_channels} → {drop_conv_layer.out_channels - 1})")
        print(f"Adding channel to {add_id}.{add_conv} "
              f"({add_conv_layer.out_channels} → {add_conv_layer.out_channels + 1})")
        
        # Perform the swap
        self.drop_channel(drop_id, drop_block, drop_conv, channel_to_drop)
        self.add_channel(add_id, add_block, add_conv, new_filter)
        
        print(f"Swap complete!\n")

    def maybe_swap_channels(self):
        """
        Check if it's time to swap channels (every swap_frequency batches).
        Call this after each training batch.
        """
        self.batch_counter += 1
        
        if self.batch_counter % self.swap_frequency == 0:
            print(f"\n[Batch {self.batch_counter}] Performing channel swap...")
            self.perform_channel_swap()

    def get_network_size(self) -> Dict:
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

    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
