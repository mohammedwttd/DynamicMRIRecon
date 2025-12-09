"""
UNet without Skip Connections (UnetSkipLess)

A UNet variant that removes skip connections between encoder and decoder.
This creates a pure encoder-decoder architecture without the characteristic
U-Net concatenation bridges.

Key differences from standard UNet:
- No skip connections between encoder and decoder
- Decoder receives only upsampled features (not concatenated)
- Decoder channels adjusted accordingly (ch instead of ch*2 input)

This architecture is useful for:
- Studying the importance of skip connections
- Creating a simpler baseline
- RigL experiments to see how sparsity patterns differ without skips
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
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

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class UnetSkipLess(nn.Module):
    """
    PyTorch implementation of a U-Net model WITHOUT skip connections.
    
    This is a pure encoder-decoder architecture where the decoder only receives
    upsampled features from the previous layer, without concatenating with
    encoder features.
    
    Architecture:
        Encoder: Input → Conv → Pool → Conv → Pool → ... → Bottleneck
        Decoder: Bottleneck → Upsample → Conv → Upsample → Conv → ... → Output
        
    No skip connections between encoder and decoder stages.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        # ===== ENCODER =====
        # Same as standard UNet encoder
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        
        # ===== BOTTLENECK =====
        self.conv = ConvBlock(ch, ch, drop_prob)

        # ===== DECODER (WITHOUT SKIP CONNECTIONS) =====
        # Key difference: input channels are just ch (not ch * 2)
        # because we don't concatenate with encoder features
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            # Input: ch channels (upsampled only, no skip concat)
            # Output: ch // 2 channels
            self.up_sample_layers += [ConvBlock(ch, ch // 2, drop_prob)]
            ch //= 2
        # Final decoder layer
        self.up_sample_layers += [ConvBlock(ch, ch, drop_prob)]
        
        # ===== OUTPUT =====
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
        output = input
        
        # ===== ENCODER =====
        # Apply down-sampling layers (NO storing for skip connections)
        for layer in self.down_sample_layers:
            output = layer(output)
            output = F.max_pool2d(output, kernel_size=2)

        # ===== BOTTLENECK =====
        output = self.conv(output)

        # ===== DECODER (NO SKIP CONNECTIONS) =====
        # Apply up-sampling layers without concatenation
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            # NO torch.cat with skip connections - just pass through
            output = layer(output)
            
        return self.conv2(output)
    
    def __repr__(self):
        return (
            f'UnetSkipLess(\n'
            f'  in_chans={self.in_chans},\n'
            f'  out_chans={self.out_chans},\n'
            f'  chans={self.chans},\n'
            f'  num_pool_layers={self.num_pool_layers},\n'
            f'  drop_prob={self.drop_prob}\n'
            f')'
        )


# ===== RigL WRAPPER =====
# RigLSkipLess is just an alias - RigL works with ANY model via the RigLScheduler
# The model itself doesn't need to be modified for RigL
RigLSkipLess = UnetSkipLess
RigLSkipLessUnet = UnetSkipLess


if __name__ == "__main__":
    # Quick test
    model = UnetSkipLess(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"UnetSkipLess total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(1, 1, 320, 320)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Compare with standard UNet param count
    from unet_model import UnetModel
    unet = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0
    )
    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"\nStandard UNet parameters: {unet_params:,}")
    print(f"UnetSkipLess parameters:  {total_params:,}")
    print(f"Difference: {unet_params - total_params:,} fewer params (no skip concat layers)")

