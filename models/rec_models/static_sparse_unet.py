"""
Static Sparse UNet - Inspired by RigL learned sparsity patterns.

Key insight from RigL: Sparsity is concentrated in bottleneck layers,
while early/late layers remain dense. This architecture statically
implements this pattern through channel reduction.

This architecture EXACTLY matches UnetModel from unet_model.py:
- Same ConvBlock structure (Conv -> InstanceNorm -> ReLU -> Dropout)
- Same upsampling (bilinear interpolation, not transposed conv)
- Same output layers (three 1x1 convs)
- Same forward pass structure with ModuleLists

RigL60 Learned Pattern (from StaticSparse64Extreme):
- Encoder density decreases going deeper (100% -> 78% -> 54% -> 31% -> 26%)
- Decoder density increases going up (26% -> 42% -> 65% -> 83% -> 67%)
- Effective params: Encoder 269K, Bottleneck 16K, Decoder 194K
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    
    EXACTLY matches unet_model.py ConvBlock.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
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
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class StaticSparseUNet(nn.Module):
    """
    Static Sparse UNet with channel counts inspired by RigL sparsity patterns.
    
    EXACTLY matches UnetModel architecture from unet_model.py:
    - Same ConvBlock, same forward structure
    - Uses bilinear interpolation for upsampling (not transposed conv)
    - Uses ModuleLists for encoder/decoder layers
    - Same output structure (three 1x1 convs)
    
    The only difference is channel counts at each layer, which are
    determined by the sparsity_level to match RigL-learned patterns.
    
    Args:
        in_chans: Number of input channels
        out_chans: Number of output channels
        chans: Base channel count (default 32 like standard UNet)
        num_pool_layers: Number of pooling layers (default 4)
        sparsity_level: How aggressively to apply sparsity pattern
        drop_prob: Dropout probability
    """
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_pool_layers: int = 4,
        sparsity_level: str = 'medium',
        drop_prob: float = 0.0,
        base_chans: int = None,  # Alias for chans (backwards compatibility)
    ):
        super().__init__()
        
        # Handle backwards compatibility
        if base_chans is not None:
            chans = base_chans
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.sparsity_level = sparsity_level
        
        # Channel configurations - all use explicit channel numbers
        # Following UnetModel pattern: bottleneck = down_3 (ch -> ch)
        
        if sparsity_level == 'light':
            # ~30% total parameter reduction
            # Channels: 32 -> 64 -> 96 -> 128 | bottle=128 | 96 -> 64 -> 32 -> 32
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 32,
                'down_1': 64,
                'down_2': 96,
                'down_3': 128,
                'bottleneck': 128,  # Same as down_3 (UnetModel pattern)
                'up_0': 96,         # Mirror of down_2
                'up_1': 64,         # Mirror of down_1
                'up_2': 32,         # Mirror of down_0
                'up_3': 32,
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
            
        elif sparsity_level == 'medium':
            # ~50% total parameter reduction (matches RigL50)
            # Channels: 32 -> 64 -> 64 -> 96 | bottle=96 | 64 -> 64 -> 32 -> 32
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 32,
                'down_1': 64,
                'down_2': 64,
                'down_3': 96,
                'bottleneck': 96,   # Same as down_3 (UnetModel pattern)
                'up_0': 64,
                'up_1': 64,
                'up_2': 32,
                'up_3': 32,
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
            
        elif sparsity_level == 'heavy':
            # ~70% total parameter reduction (matches RigL70-80)
            # Channels: 32 -> 48 -> 48 -> 64 | bottle=64 | 48 -> 48 -> 32 -> 32
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 32,
                'down_1': 48,
                'down_2': 48,
                'down_3': 64,
                'bottleneck': 64,   # Same as down_3 (UnetModel pattern)
                'up_0': 48,
                'up_1': 48,
                'up_2': 32,
                'up_3': 32,
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
        elif sparsity_level == 'wide_slim':
            # 48 base channels with slim bottleneck
            # Wide early layers for better feature extraction
            # Channels: 48 -> 96 -> 144 -> 192 | bottle=192 | 144 -> 96 -> 48 -> 48
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 48,
                'down_1': 96,
                'down_2': 144,
                'down_3': 192,
                'bottleneck': 192,  # Same as down_3 (UnetModel pattern)
                'up_0': 144,        # Mirror of down_2
                'up_1': 96,         # Mirror of down_1
                'up_2': 48,         # Mirror of down_0
                'up_3': 48,
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
            
        elif sparsity_level == 'ultralight':
            # 32 base channels with EXTREME compression
            # Full capacity at edges, severely compressed center
            # Channels: 32 -> 64 -> 64 -> 64 | bottle=64 | 64 -> 64 -> 32 -> 32
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 32,
                'down_1': 64,
                'down_2': 64,
                'down_3': 64,
                'bottleneck': 64,   # Same as down_3 (UnetModel pattern)
                'up_0': 64,
                'up_1': 64,
                'up_2': 32,
                'up_3': 32,
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
            
        elif sparsity_level == 'flat32':
            # UNIFORM 32 channels throughout - flat architecture
            # Every layer has exactly 32 channels (no pyramid)
            # Channels: 32 -> 32 -> 32 -> 32 | bottle=32 | 32 -> 32 -> 32 -> 32
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 32,
                'down_1': 32,
                'down_2': 32,
                'down_3': 32,
                'bottleneck': 32,   # Same as down_3 (UnetModel pattern)
                'up_0': 32,
                'up_1': 32,
                'up_2': 32,
                'up_3': 32,
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
            
        elif sparsity_level == 'wide64_extreme':
            # 64 base channels with EXTREME compression
            # Wide early layers (64, 128) but compressed deep layers
            # Channels: 64 -> 128 -> 128 -> 96 | bottle=96 | 96 -> 128 -> 64 -> 64
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 64,
                'down_1': 128,
                'down_2': 128,
                'down_3': 96,
                'bottleneck': 96,   # Same as down_3 (UnetModel pattern)
                'up_0': 128,        # Mirror of down_2
                'up_1': 128,        # Mirror of down_1
                'up_2': 64,         # Mirror of down_0
                'up_3': 64,
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
            
        elif sparsity_level == 'wide64_extreme_slim':
            # 64 base channels with slimmer decoder
            # Channels: 64 -> 128 -> 128 -> 96 | bottle=96 | 64 -> 64 -> 64 -> 64
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 64,
                'down_1': 128,
                'down_2': 128,
                'down_3': 96,
                'bottleneck': 96,   # Same as down_3 (UnetModel pattern)
                'up_0': 64,
                'up_1': 64,
                'up_2': 64,
                'up_3': 64,
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
            
        elif sparsity_level == 'wide64_decoder_heavy':
            # Shifted toward decoder (RigL-inspired: slim encoder, fuller decoder)
            # Encoder is compressed, decoder has more capacity
            # Channels: 64 -> 64 -> 64 -> 48 | bottle=48 | 96 -> 128 -> 128 -> 64
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 64,   # Full at input
                'down_1': 64,   # Slim (was 128)
                'down_2': 64,   # Slim (was 128)
                'down_3': 48,   # Very slim (was 96)
                'bottleneck': 48,   # Compressed bottleneck
                'up_0': 96,     # Fuller decoder
                'up_1': 128,    # Peak capacity in decoder
                'up_2': 128,    # High capacity
                'up_3': 64,     # Full at output
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
        elif sparsity_level == 'asymmetric_rigl':
            # EXACT MATCH of RigL60-learned parameter distribution
            # Following UnetModel pattern: bottleneck keeps same channels (ch -> ch)
            # 
            # RigL60 learned asymmetric pattern:
            #   - Encoder channels DECREASE going deeper
            #   - Decoder channels INCREASE going up (mirror of encoder)
            # 
            # Channels: down=56->72->64->48 | bottle=48->48 | up=48->64->72->56
            # This mirrors encoder in reverse for decoder
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 56,   # First encoder
                'down_1': 72,   # Second encoder (peak)
                'down_2': 64,   # Third encoder (decreasing)
                'down_3': 48,   # Fourth encoder (slim)
                'bottleneck': 48,  # Same as down_3 output (like UnetModel!)
                'up_0': 64,     # Mirror of down_2
                'up_1': 72,     # Mirror of down_1
                'up_2': 56,     # Mirror of down_0
                'up_3': 56,     # Final output channels
            }
            # Dummy multipliers (not used when explicit_channels set)
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
            
        elif sparsity_level == 'asymmetric_slim':
            # SLIMMER version - following UnetModel pattern
            # Channels: down=40->56->48->32 | bottle=32->32 | up=48->56->40->40
            self.use_explicit_channels = True
            self.explicit_channels = {
                'down_0': 40,
                'down_1': 56,
                'down_2': 48,
                'down_3': 32,
                'bottleneck': 32,  # Same as down_3 output (like UnetModel!)
                'up_0': 48,     # Mirror of down_2
                'up_1': 56,     # Mirror of down_1
                'up_2': 40,     # Mirror of down_0
                'up_3': 40,     # Final output
            }
            self.channel_mult = {k: 1.0 for k in self.explicit_channels}
        else:
            raise ValueError(f"Unknown sparsity_level: {sparsity_level}")
        
        # All sparsity levels now use explicit channels
        down_channels = [
            self.explicit_channels['down_0'],
            self.explicit_channels['down_1'],
            self.explicit_channels['down_2'],
            self.explicit_channels['down_3'],
        ]
        bottle_ch = self.explicit_channels['bottleneck']
        up_channels = [
            self.explicit_channels['up_0'],
            self.explicit_channels['up_1'],
            self.explicit_channels['up_2'],
            self.explicit_channels['up_3'],
        ]
        
        # Store for debugging/parameter counting
        self.channel_config = {
            'down_0': down_channels[0], 'down_1': down_channels[1], 
            'down_2': down_channels[2], 'down_3': down_channels[3],
            'bottleneck': bottle_ch,
            'up_0': up_channels[0], 'up_1': up_channels[1], 
            'up_2': up_channels[2], 'up_3': up_channels[3]
        }
        
        # ================================================================
        # BUILD LAYERS - EXACTLY LIKE UnetModel
        # ================================================================
        
        # Encoder (down_sample_layers) - EXACTLY like UnetModel lines 85-89
        # First layer: in_chans -> down_channels[0]
        # Then: down_channels[i] -> down_channels[i+1]
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, down_channels[0], drop_prob)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers.append(
                ConvBlock(down_channels[i], down_channels[i + 1], drop_prob)
            )
        
        # Bottleneck (called 'conv' in UnetModel) - line 90
        # IMPORTANT: UnetModel uses ch -> ch (same input/output channels!)
        # So bottle_ch should equal down_channels[-1]
        self.conv = ConvBlock(down_channels[-1], bottle_ch, drop_prob)
        
        # Decoder (up_sample_layers) - EXACTLY like UnetModel lines 92-96
        # UnetModel pattern: ConvBlock(ch * 2, ch // 2) for most layers
        # The "ch * 2" comes from: current_output + skip_connection
        self.up_sample_layers = nn.ModuleList()
        
        # Build decoder layers following UnetModel pattern
        # After bottleneck, ch = bottle_ch (which should = down_channels[-1])
        # First decoder: input = bottle_ch + down_channels[-1], output = up_channels[0]
        ch = bottle_ch
        for i in range(num_pool_layers - 1):
            # Skip connection from corresponding encoder layer
            skip_idx = num_pool_layers - 1 - i  # down_3, down_2, down_1
            skip_ch = down_channels[skip_idx]
            self.up_sample_layers.append(
                ConvBlock(ch + skip_ch, up_channels[i], drop_prob)
            )
            ch = up_channels[i]
        
        # Last decoder layer - skip from down_0
        self.up_sample_layers.append(
            ConvBlock(ch + down_channels[0], up_channels[-1], drop_prob)
        )
        
        # Output layers (conv2) - EXACTLY like UnetModel lines 97-101
        final_ch = up_channels[-1]
        self.conv2 = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, kernel_size=1),
            nn.Conv2d(final_ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
        
    def forward(self, input):
        """
        Forward pass - EXACTLY matches UnetModel.
        Uses bilinear interpolation for upsampling (not transposed conv).
        """
        stack = []
        output = input
        
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        
        # Bottleneck
        output = self.conv(output)
        
        # Apply up-sampling layers (bilinear interpolation + skip connection)
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        
        return self.conv2(output)
    
    def get_param_count(self):
        """Return total parameters and breakdown by section."""
        total = sum(p.numel() for p in self.parameters())
        
        encoder_params = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in self.down_sample_layers
        )
        
        bottleneck_params = sum(p.numel() for p in self.conv.parameters())
        
        decoder_params = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in self.up_sample_layers
        )
        
        output_params = sum(p.numel() for p in self.conv2.parameters())
        
        return {
            'total': total,
            'encoder': encoder_params,
            'bottleneck': bottleneck_params,
            'decoder': decoder_params,
            'output': output_params,
        }


def create_static_sparse_unet(sparsity_level='medium', **kwargs):
    """Factory function to create StaticSparseUNet."""
    return StaticSparseUNet(sparsity_level=sparsity_level, **kwargs)


if __name__ == '__main__':
    # Test and compare parameter counts
    print("=" * 80)
    print("Static Sparse UNet - Parameter Comparison")
    print("=" * 80)
    
    try:
        from unet_model import UnetModel  # Standard UNet for comparison
        standard = UnetModel(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
        standard_params = sum(p.numel() for p in standard.parameters())
        print(f"\nStandard UNet (32 base channels): {standard_params:,} params")
    except ImportError:
        standard_params = 7_759_521  # Approximate
        print(f"\nStandard UNet (reference): ~{standard_params:,} params")
    
    print("-" * 80)
    
    for level in ['light', 'medium', 'heavy', 'wide_slim', 'ultralight', 'flat32', 'wide64_extreme', 'wide64_extreme_slim', 'wide64_decoder_heavy', 'asymmetric_rigl', 'asymmetric_slim']:
        model = StaticSparseUNet(sparsity_level=level)
        counts = model.get_param_count()
        reduction = 100 * (1 - counts['total'] / standard_params)
        
        print(f"\nStaticSparseUNet ({level}):")
        print(f"  Total params: {counts['total']:,} ({reduction:.1f}% reduction)")
        print(f"  Encoder: {counts['encoder']:,}, Bottleneck: {counts['bottleneck']:,}")
        print(f"  Decoder: {counts['decoder']:,}, Output: {counts['output']:,}")
        print(f"  Channel config: {model.channel_config}")
    
    # Test forward pass
    print("\n" + "=" * 80)
    print("Forward pass test:")
    x = torch.randn(1, 1, 320, 320)
    for level in ['light', 'medium', 'heavy', 'wide_slim', 'ultralight', 'flat32', 'wide64_extreme', 'wide64_extreme_slim', 'wide64_decoder_heavy', 'asymmetric_rigl', 'asymmetric_slim']:
        model = StaticSparseUNet(sparsity_level=level)
        y = model(x)
        print(f"  {level}: input {x.shape} -> output {y.shape}")

