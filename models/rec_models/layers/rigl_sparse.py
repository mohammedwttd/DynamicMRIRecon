"""
RigL: Rigging the Lottery - Dynamic Sparse Training for MRI Reconstruction

This module implements the RigL algorithm which:
1. Starts with random sparsity (e.g., 50%)
2. Periodically reallocates weights based on gradient magnitude
3. Drops lowest-magnitude weights, grows highest-gradient positions
4. Maintains constant sparsity throughout training

For MRI reconstruction, this naturally moves capacity from encoder to decoder,
as decoder gradients are typically higher (more important for reconstruction).

Reference: Evci et al., "Rigging the Lottery: Making All Tickets Winners" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class RigLScheduler:
    """
    Manages sparse masks and performs RigL updates across the model.
    
    Usage:
        scheduler = RigLScheduler(model, sparsity=0.5, update_freq=100)
        
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            
            # RigL update (call every iteration, internally tracks frequency)
            scheduler.step()
            
            optimizer.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        update_freq: int = 100,
        T_end: int = None,  # When to stop updating (default: 75% of training)
        delta: float = 0.3,  # Fraction of weights to reallocate each update
        exclude_layers: List[str] = None,  # Layer names to exclude from sparsity
        initial_masks_path: str = None,  # Path to pre-computed masks (e.g., from SNIP)
        static_mask: bool = False,  # If True, never update masks (static sparse training)
    ):
        """
        Args:
            model: The neural network to sparsify
            sparsity: Target sparsity ratio (0.5 = 50% weights are zero)
            update_freq: How often to update masks (in iterations)
            T_end: Iteration to stop mask updates (allows fine-tuning with fixed sparsity)
            delta: Fraction of active weights to drop/regrow each update
            exclude_layers: List of layer name patterns to exclude (e.g., ['first_conv', 'final'])
            initial_masks_path: Path to .pt file with pre-computed masks (from SNIP/PUN-IT)
            static_mask: If True, mask is never updated (pure static sparse training)
        """
        self.model = model
        self.sparsity = sparsity
        self.update_freq = update_freq
        self.T_end = T_end
        self.delta = delta
        self.exclude_layers = exclude_layers or []
        self.initial_masks_path = initial_masks_path
        self.static_mask = static_mask
        
        self.iteration = 0
        self.masks: Dict[str, torch.Tensor] = {}
        self.layer_info: Dict[str, dict] = {}  # Track encoder vs decoder
        
        self._initialize_masks()
        self._apply_masks()
        
        # Statistics tracking
        self.encoder_density_history = []
        self.decoder_density_history = []
        
    def _is_excluded(self, name: str) -> bool:
        """Check if layer should be excluded from sparsity."""
        for pattern in self.exclude_layers:
            if pattern in name:
                return True
        return False
    
    def _classify_layer(self, name: str) -> str:
        """Classify layer as encoder, decoder, or bottleneck."""
        name_lower = name.lower()
        if 'down' in name_lower or 'encoder' in name_lower:
            return 'encoder'
        elif 'up' in name_lower or 'decoder' in name_lower:
            return 'decoder'
        elif 'bottleneck' in name_lower or 'conv_' in name_lower:
            return 'bottleneck'
        else:
            # Heuristic: first half of layers = encoder
            return 'unknown'
    
    def _initialize_masks(self):
        """Initialize sparse masks for all eligible layers.
        
        If initial_masks_path is provided, loads pre-computed masks (e.g., from SNIP/PUN-IT).
        Otherwise, creates random sparse masks.
        """
        # Load pre-computed masks if provided
        loaded_masks = None
        if self.initial_masks_path:
            print(f"\n{'='*80}")
            print(f"Loading initial masks from: {self.initial_masks_path}")
            checkpoint = torch.load(self.initial_masks_path, map_location='cpu')
            if 'masks' in checkpoint:
                loaded_masks = checkpoint['masks']
                loaded_sparsity = checkpoint.get('sparsity', 'unknown')
                loaded_method = checkpoint.get('method', 'unknown')
                print(f"  Method: {loaded_method}, Sparsity: {loaded_sparsity}")
            else:
                raise ValueError(f"No 'masks' key found in {self.initial_masks_path}")
        
        init_type = "Pre-computed" if loaded_masks else "Random"
        mode_type = "STATIC" if self.static_mask else "DYNAMIC (RigL)"
        print(f"\n{'='*80}")
        print(f"Sparse Training: {self.sparsity*100:.0f}% sparsity ({init_type} masks, {mode_type})")
        print(f"{'='*80}")
        print(f"{'Status':<8} {'Type':<10} {'Layer Name':<45} {'Shape':<20} {'Active':>12} {'Total':>12} {'Density':>8}")
        print(f"{'-'*80}")
        
        total_params = 0
        sparse_params = 0
        skipped_params = 0
        total_active = 0
        
        # Track excluded layers for display
        self.excluded_layers = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                numel = weight.numel()
                
                if self._is_excluded(name):
                    # Track excluded layer info for display
                    self.excluded_layers[name] = {
                        'type': self._classify_layer(name),
                        'shape': tuple(weight.shape),
                        'numel': numel,
                        'module': module,
                    }
                    skipped_params += numel
                    layer_type = self._classify_layer(name)
                    shape_str = str(list(weight.shape))
                    print(f"[SKIP]   {layer_type.upper():<10} {name:<45} {shape_str:<20} {numel:>12,} {numel:>12,} {'100.0%':>8}")
                    continue
                
                # Check if we have a pre-computed mask for this layer
                # Try multiple name variations to handle DataParallel/reconstruction_model wrapping
                mask = None
                matched_name = None
                if loaded_masks:
                    # Try exact name first
                    name_variants = [name]
                    # Strip common prefixes
                    stripped = name
                    for prefix in ['module.', 'reconstruction_model.', 'module.reconstruction_model.']:
                        if stripped.startswith(prefix):
                            stripped = stripped[len(prefix):]
                            name_variants.append(stripped)
                    # Also try the base name without any prefix
                    name_variants.append(stripped)
                    
                    for variant in name_variants:
                        if variant in loaded_masks:
                            matched_name = variant
                            mask = loaded_masks[variant].float()
                            if mask.shape != weight.shape:
                                print(f"[WARN] Shape mismatch for {name} (matched {variant}): mask {mask.shape} vs weight {weight.shape}, using random")
                                mask = None
                                matched_name = None
                            else:
                                num_active = int(mask.sum().item())
                                num_zeros = numel - num_active
                                status = "[LOAD]"
                            break
                
                if mask is None:
                    # Create random sparse mask
                    mask = torch.ones_like(weight)
                    num_zeros = int(numel * self.sparsity)
                    num_active = numel - num_zeros
                    
                    # Randomly select positions to zero out (use fixed seed for reproducibility)
                    flat_mask = mask.view(-1)
                    generator = torch.Generator().manual_seed(42)
                    zero_indices = torch.randperm(numel, generator=generator)[:num_zeros]
                    flat_mask[zero_indices] = 0
                    mask = flat_mask.view_as(weight)
                    status = "[RAND]" if loaded_masks else "[RIGL]"
                
                self.masks[name] = mask
                self.layer_info[name] = {
                    'type': self._classify_layer(name),
                    'shape': tuple(weight.shape),
                    'numel': numel,
                    'module': module,
                }
                
                total_params += numel
                sparse_params += num_zeros
                total_active += num_active
                
                layer_type = self.layer_info[name]['type']
                density = num_active / numel * 100
                shape_str = str(list(weight.shape))
                print(f"{status}   {layer_type.upper():<10} {name:<45} {shape_str:<20} {num_active:>12,} {numel:>12,} {density:>7.1f}%")
        
        print(f"{'-'*80}")
        print(f"RigL Layers:    {total_params:>12,} total params, {total_active:>12,} active ({total_active/total_params*100:.1f}% dense)")
        print(f"Skipped Layers: {skipped_params:>12,} params (100% dense, not modified by RigL)")
        if loaded_masks:
            print(f"Initial Masks:  Loaded from {self.initial_masks_path}")
        print(f"{'='*80}\n")
    
    def _apply_masks(self):
        """Apply current masks to model weights (zero out masked positions)."""
        with torch.no_grad():
            for name, mask in self.masks.items():
                module = self.layer_info[name]['module']
                module.weight.data *= mask.to(module.weight.device)
    
    def apply_masks(self):
        """Public method to apply masks - call at start of each forward pass."""
        self._apply_masks()
    
    def _get_gradient_scores(self, name: str) -> torch.Tensor:
        """Get gradient magnitude for growing new connections (uses last gradient only)."""
        module = self.layer_info[name]['module']
        if module.weight.grad is None:
            return torch.zeros_like(module.weight)
        return module.weight.grad.abs()
    
    def _get_weight_scores(self, name: str) -> torch.Tensor:
        """Get weight magnitude (for pruning decisions)."""
        module = self.layer_info[name]['module']
        return module.weight.data.abs()
    
    def _cosine_decay(self) -> float:
        """Cosine decay for the number of weights to reallocate."""
        if self.T_end is None:
            return self.delta
        
        progress = min(self.iteration / self.T_end, 1.0)
        return self.delta * (1 + math.cos(math.pi * progress)) / 2
    
    def step(self):
        """
        Called every iteration. Performs RigL update if at update frequency.
        Should be called AFTER loss.backward() but BEFORE optimizer.step().
        """
        self.iteration += 1
        
        # Static mask mode: never update, just enforce sparsity
        if self.static_mask:
            self._apply_masks()
            return
        
        # Check if we should update masks
        if self.iteration % self.update_freq != 0:
            self._apply_masks()  # Always enforce sparsity
            return
        
        # Check if we've passed T_end
        if self.T_end is not None and self.iteration > self.T_end:
            self._apply_masks()
            return
        
        # Perform RigL update using last gradient
        self._rigl_update()
        self._apply_masks()
        self._log_statistics()
    
    def _rigl_update(self):
        """
        Core RigL algorithm - GLOBAL drop/grow across all layers.
        Uses pure PyTorch operations for speed (no Python loops over indices).
        """
        decay = self._cosine_decay()
        
        print(f"\n{'='*70}")
        print(f"[RigL GLOBAL Update @ iter {self.iteration}] delta={decay:.3f}")
        print(f"{'='*70}")
        
        # Step 1: Build global tensors for all weights and gradients
        layer_names = list(self.masks.keys())
        layer_offsets = {}  # Maps layer name to (start_idx, end_idx) in global tensor
        
        # Calculate sizes and offsets
        total_params = sum(self.layer_info[name]['numel'] for name in layer_names)
        offset = 0
        for name in layer_names:
            numel = self.layer_info[name]['numel']
            layer_offsets[name] = (offset, offset + numel)
            offset += numel
        
        # Get device
        first_module = self.layer_info[layer_names[0]]['module']
        device = first_module.weight.device
        
        # Build global tensors
        global_weights = torch.zeros(total_params, device=device)
        global_grads = torch.zeros(total_params, device=device)
        global_mask = torch.zeros(total_params, device=device)
        
        layer_stats_before = {}
        
        for name in layer_names:
            start, end = layer_offsets[name]
            module = self.layer_info[name]['module']
            mask = self.masks[name].to(device)
            
            global_weights[start:end] = module.weight.data.abs().view(-1)
            global_mask[start:end] = mask.view(-1)
            
            # Use last gradient only
            if module.weight.grad is not None:
                global_grads[start:end] = module.weight.grad.abs().view(-1)
            
            layer_stats_before[name] = {
                'active': mask.sum().item(),
                'total': mask.numel(),
                'density': mask.sum().item() / mask.numel() * 100,
                'type': self.layer_info[name]['type']
            }
        
        # Count active/inactive
        active_mask = global_mask > 0
        total_active = active_mask.sum().item()
        total_inactive = (~active_mask).sum().item()
        
        num_to_change = int(total_active * decay)
        
        print(f"\n  Total active: {total_active}, inactive: {total_inactive}")
        print(f"  Reallocating: {num_to_change} weights")
        
        if num_to_change == 0:
            return
        
        # Step 2: Find lowest-magnitude ACTIVE weights to DROP
        # Set inactive to inf so they won't be selected
        drop_scores = global_weights.clone()
        drop_scores[~active_mask] = float('inf')
        _, drop_indices = torch.topk(drop_scores, num_to_change, largest=False)
        
        # Step 3: Find highest-gradient INACTIVE positions to GROW
        # Set active to -inf so they won't be selected
        grow_scores = global_grads.clone()
        grow_scores[active_mask] = float('-inf')
        _, grow_indices = torch.topk(grow_scores, num_to_change, largest=True)
        
        # Count drops/grows per layer (vectorized)
        drops_per_layer = {}
        grows_per_layer = {}
        
        for name in layer_names:
            start, end = layer_offsets[name]
            drops_in_layer = ((drop_indices >= start) & (drop_indices < end)).sum().item()
            grows_in_layer = ((grow_indices >= start) & (grow_indices < end)).sum().item()
            drops_per_layer[name] = drops_in_layer
            grows_per_layer[name] = grows_in_layer
        
        # Apply changes to global mask
        global_mask[drop_indices] = 0
        global_mask[grow_indices] = 1
        
        # Copy back to per-layer masks and initialize new weights
        for name in layer_names:
            start, end = layer_offsets[name]
            module = self.layer_info[name]['module']
            self.masks[name] = global_mask[start:end].view_as(module.weight)
            
            # Initialize grown weights
            if grows_per_layer[name] > 0:
                layer_grow_mask = (grow_indices >= start) & (grow_indices < end)
                layer_grow_indices = grow_indices[layer_grow_mask] - start
                with torch.no_grad():
                    # Use fixed seed for reproducibility
                    generator = torch.Generator(device=device).manual_seed(42 + self.iteration)
                    module.weight.data.view(-1)[layer_grow_indices] = \
                        torch.randn(layer_grow_indices.numel(), device=device, generator=generator) * 0.01
        
        # Print per-layer status (ALL layers including unchanged and skipped)
        print(f"\n  {'='*100}")
        print(f"  ALL LAYERS STATUS:")
        print(f"  {'='*100}")
        print(f"  {'Status':<8} {'Type':<10} {'Layer Name':<40} {'Active':>10} {'Total':>10} {'Density':>8} {'Drop':>8} {'Grow':>8} {'Net':>8}")
        print(f"  {'-'*100}")
        
        encoder_net = 0
        decoder_net = 0
        bottleneck_net = 0
        
        # First print skipped layers
        if hasattr(self, 'excluded_layers'):
            for name, info in self.excluded_layers.items():
                layer_type = info['type']
                numel = info['numel']
                print(f"  {'[SKIP]':<8} {layer_type.upper():<10} {name:<40} {numel:>10,} {numel:>10,} {'100.0%':>8} {'-':>8} {'-':>8} {'-':>8}")
        
        # Then print all RigL layers (changed or not)
        for name in layer_names:
            dropped = drops_per_layer[name]
            grown = grows_per_layer[name]
            net_change = grown - dropped
            layer_type = layer_stats_before[name]['type']
            
            old_active = int(layer_stats_before[name]['active'])
            new_active = old_active + net_change
            total = layer_stats_before[name]['total']
            new_density = new_active / total * 100
            
            # Show change indicator
            if net_change > 0:
                net_str = f"+{net_change}"
            elif net_change < 0:
                net_str = f"{net_change}"
            else:
                net_str = "0"
            
            print(f"  {'[RIGL]':<8} {layer_type.upper():<10} {name:<40} {new_active:>10,} {total:>10,} {new_density:>7.1f}% {dropped:>8} {grown:>8} {net_str:>8}")
            
            if layer_type == 'encoder':
                encoder_net += net_change
            elif layer_type == 'decoder':
                decoder_net += net_change
            else:
                bottleneck_net += net_change
        
        print(f"  {'-'*100}")
        
        # Summary
        print(f"\n  CAPACITY FLOW SUMMARY:")
        print(f"    Encoder:    {'+' if encoder_net >= 0 else ''}{encoder_net:>8} weights")
        print(f"    Decoder:    {'+' if decoder_net >= 0 else ''}{decoder_net:>8} weights")
        print(f"    Bottleneck: {'+' if bottleneck_net >= 0 else ''}{bottleneck_net:>8} weights")
        
        if decoder_net > encoder_net:
            print(f"\n  ✓ Capacity flowing Encoder → Decoder!")
        print(f"  {'='*100}")
    
    def _log_statistics(self):
        """Log encoder vs decoder density statistics."""
        encoder_active = 0
        encoder_total = 0
        decoder_active = 0
        decoder_total = 0
        bottleneck_active = 0
        bottleneck_total = 0
        
        for name, mask in self.masks.items():
            info = self.layer_info[name]
            active = int(mask.sum().item())
            total = info['numel']
            
            if info['type'] == 'encoder':
                encoder_active += active
                encoder_total += total
            elif info['type'] == 'decoder':
                decoder_active += active
                decoder_total += total
            else:
                bottleneck_active += active
                bottleneck_total += total
        
        print(f"\n  DENSITY BY REGION:")
        if encoder_total > 0:
            encoder_density = encoder_active / encoder_total
            self.encoder_density_history.append(encoder_density)
            print(f"    Encoder:    {encoder_active:>10,} / {encoder_total:>10,} active ({encoder_density*100:.1f}% dense)")
        
        if decoder_total > 0:
            decoder_density = decoder_active / decoder_total
            self.decoder_density_history.append(decoder_density)
            print(f"    Decoder:    {decoder_active:>10,} / {decoder_total:>10,} active ({decoder_density*100:.1f}% dense)")
        
        if bottleneck_total > 0:
            bottleneck_density = bottleneck_active / bottleneck_total
            print(f"    Bottleneck: {bottleneck_active:>10,} / {bottleneck_total:>10,} active ({bottleneck_density*100:.1f}% dense)")
        
        total_active = encoder_active + decoder_active + bottleneck_active
        total_all = encoder_total + decoder_total + bottleneck_total
        print(f"    TOTAL:      {total_active:>10,} / {total_all:>10,} active ({total_active/total_all*100:.1f}% dense)")
        
        if encoder_total > 0 and decoder_total > 0:
            ratio = decoder_density / encoder_density if encoder_density > 0 else float('inf')
            print(f"\n  Decoder/Encoder density ratio: {ratio:.2f}x")
    
    def get_layer_densities(self) -> Dict[str, float]:
        """Get current density for each layer."""
        densities = {}
        for name, mask in self.masks.items():
            densities[name] = mask.sum().item() / mask.numel()
        return densities
    
    def state_dict(self) -> dict:
        """Save scheduler state for checkpointing."""
        return {
            'iteration': self.iteration,
            'masks': {k: v.cpu() for k, v in self.masks.items()},
            'encoder_density_history': self.encoder_density_history,
            'decoder_density_history': self.decoder_density_history,
        }
    
    def load_state_dict(self, state: dict):
        """Load scheduler state from checkpoint."""
        self.iteration = state['iteration']
        self.masks = {k: v for k, v in state['masks'].items()}
        self.encoder_density_history = state.get('encoder_density_history', [])
        self.decoder_density_history = state.get('decoder_density_history', [])
        self._apply_masks()
    
    def reconstruct_from_weights(self, start_iteration: int = 0):
        """
        Reconstruct masks from model weights (for resuming without saved rigl state).
        
        Since masked weights are zeroed during training (weight *= mask), we can
        reconstruct the mask by checking which weights are zero.
        
        Args:
            start_iteration: The iteration to resume from (e.g., epoch * iters_per_epoch)
        """
        print(f"\n{'='*80}")
        print(f"RigL: Reconstructing masks from weights (resuming at iter {start_iteration})")
        print(f"{'='*80}")
        print(f"{'Status':<8} {'Type':<10} {'Layer Name':<40} {'Active':>12} {'Total':>12} {'Density':>8}")
        print(f"{'-'*80}")
        
        self.iteration = start_iteration
        
        encoder_active = 0
        encoder_total = 0
        decoder_active = 0
        decoder_total = 0
        bottleneck_active = 0
        bottleneck_total = 0
        
        # Print skipped layers first
        if hasattr(self, 'excluded_layers'):
            for name, info in self.excluded_layers.items():
                layer_type = info['type']
                numel = info['numel']
                print(f"{'[SKIP]':<8} {layer_type.upper():<10} {name:<40} {numel:>12,} {numel:>12,} {'100.0%':>8}")
        
        for name in list(self.masks.keys()):
            module = self.layer_info[name]['module']
            weight = module.weight.data
            
            # Reconstruct mask: 1 where weight != 0, 0 where weight == 0
            mask = (weight != 0).float()
            self.masks[name] = mask
            
            active = int(mask.sum().item())
            total = mask.numel()
            density = active / total * 100
            
            layer_type = self.layer_info[name]['type']
            print(f"{'[RIGL]':<8} {layer_type.upper():<10} {name:<40} {active:>12,} {total:>12,} {density:>7.1f}%")
            
            if layer_type == 'encoder':
                encoder_active += active
                encoder_total += total
            elif layer_type == 'decoder':
                decoder_active += active
                decoder_total += total
            else:
                bottleneck_active += active
                bottleneck_total += total
        
        print(f"{'-'*80}")
        print(f"\nDENSITY BY REGION:")
        if encoder_total > 0:
            print(f"  Encoder:    {encoder_active:>10,} / {encoder_total:>10,} active ({encoder_active/encoder_total*100:.1f}% dense)")
        if decoder_total > 0:
            print(f"  Decoder:    {decoder_active:>10,} / {decoder_total:>10,} active ({decoder_active/decoder_total*100:.1f}% dense)")
        if bottleneck_total > 0:
            print(f"  Bottleneck: {bottleneck_active:>10,} / {bottleneck_total:>10,} active ({bottleneck_active/bottleneck_total*100:.1f}% dense)")
        
        total_active = encoder_active + decoder_active + bottleneck_active
        total_all = encoder_total + decoder_total + bottleneck_total
        print(f"  TOTAL:      {total_active:>10,} / {total_all:>10,} active ({total_active/total_all*100:.1f}% dense)")
        print(f"{'='*80}\n")


class SparseConv2d(nn.Conv2d):
    """
    A Conv2d layer with built-in sparse mask support.
    Can be used standalone or managed by RigLScheduler.
    """
    
    def __init__(self, *args, sparsity: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity
        self.register_buffer('mask', torch.ones_like(self.weight))
        self._init_sparse_mask()
    
    def _init_sparse_mask(self):
        """Initialize random sparse mask."""
        numel = self.weight.numel()
        num_zeros = int(numel * self.sparsity)
        
        flat_mask = self.mask.view(-1)
        # Use fixed seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        zero_indices = torch.randperm(numel, generator=generator)[:num_zeros]
        flat_mask[zero_indices] = 0
        self.mask = flat_mask.view_as(self.weight)
        
        # Apply initial mask
        with torch.no_grad():
            self.weight.data *= self.mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask during forward pass
        masked_weight = self.weight * self.mask
        return F.conv2d(
            x, masked_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


def apply_rigl_to_model(
    model: nn.Module,
    sparsity: float = 0.5,
    update_freq: int = 100,
    total_iters: int = None,
    exclude_first_last: bool = True,
) -> RigLScheduler:
    """
    Convenience function to apply RigL to any model.
    
    Args:
        model: The model to sparsify
        sparsity: Target sparsity (0.5 = 50% zeros)
        update_freq: How often to update masks
        total_iters: Total training iterations (for T_end calculation)
        exclude_first_last: Whether to exclude first and last conv layers
    
    Returns:
        RigLScheduler instance
    
    Example:
        model = UNet(...)
        rigl = apply_rigl_to_model(model, sparsity=0.5, update_freq=100)
        
        for epoch in range(epochs):
            for batch in dataloader:
                loss = criterion(model(x), y)
                loss.backward()
                rigl.step()  # Update masks if needed
                optimizer.step()
                optimizer.zero_grad()
    """
    exclude = []
    if exclude_first_last:
        # Patterns for first/last layers across different architectures:
        # - UnetModel: first, final, conv.0
        # - FastMRI Unet: down_sample_layers.0.layers.0, up_conv.3.1
        exclude = [
            'first', 'initial', 'final', 'last', 'output',  # Generic patterns
            'down_sample_layers.0.layers.0',  # FastMRI Unet first conv
            'up_conv.3.1',  # FastMRI Unet final conv
        ]
    
    T_end = int(total_iters * 0.75) if total_iters else None
    
    scheduler = RigLScheduler(
        model=model,
        sparsity=sparsity,
        update_freq=update_freq,
        T_end=T_end,
        exclude_layers=exclude,
    )
    
    return scheduler

