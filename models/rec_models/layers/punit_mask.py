"""
PUN-IT & SNIP: Pruning at Initialization for MRI Reconstruction

This module implements two approaches for finding masks BEFORE training:

1. PUN-IT: Optimizes learnable mask probabilities using Gumbel-Softmax
   - Requires multiple optimization steps
   - Can struggle with random weights (no meaningful gradient signal)

2. SNIP (Single-shot Network Pruning): Uses gradient × weight as importance score
   - Single forward-backward pass
   - Much faster and often more effective
   - Naturally finds which weights contribute to loss reduction

Reference: 
- PUN-IT paper methodology
- Lee et al., "SNIP: Single-shot Network Pruning based on Connection Sensitivity" (ICLR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import math


class PUNITMaskFinder:
    """
    PUN-IT: Pruning Unrolled Networks at Initialization.
    
    Implements the mask-finding algorithm from:
    "Pruning Unrolled Networks (PUN) at Initialization for MRI Reconstruction 
     Improves Generalization" (arXiv:2412.18668v1)
    
    Key components following the paper:
    1. Frozen θ_init: Original weights never change during mask search
    2. Learnable p: Bernoulli probability parameters optimized via gradient descent
    3. Gumbel-Softmax (Eq. 7): Two independent Gumbel samples for differentiable sampling
    4. KL Divergence (Eq. 5): Regularizes p towards target sparsity p_0 = s/d
    5. Binarization C(p*) (Eq. 6): Keep top-s entries based on final probabilities
    """
    
    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        temperature: float = 5.0,  # Start higher for smoother gradients
        temperature_anneal: float = 0.995,  # Slower decay for stability
        min_temperature: float = 0.1,
        exclude_layers: List[str] = None,
        device: str = 'cuda',
        init_noise: float = 0.0,  # Noise to add to initial logits (breaks symmetry)
    ):
        self.model = model
        self.sparsity = sparsity
        self.temperature = temperature
        self.temperature_anneal = temperature_anneal
        self.min_temperature = min_temperature
        self.exclude_layers = exclude_layers or []
        self.device = device
        self.init_noise = init_noise
        
        # Shadow parameters (probabilities)
        self.prob_params: Dict[str, nn.Parameter] = {}
        self.layer_info: Dict[str, dict] = {}
        
        # Optimization: Cache references to original weights
        self.original_weights: Dict[str, torch.Tensor] = {}
        
        self._freeze_weights()
        self._initialize_probabilities()
        
    def _is_excluded(self, name: str) -> bool:
        for pattern in self.exclude_layers:
            if pattern in name:
                return True
        return False

    def _classify_layer(self, name: str) -> str:
        name_lower = name.lower()
        if 'down' in name_lower or 'encoder' in name_lower:
            return 'encoder'
        elif 'up' in name_lower or 'decoder' in name_lower:
            return 'decoder'
        elif 'bottleneck' in name_lower or 'conv_' in name_lower:
            return 'bottleneck'
        return 'unknown'
    
    def _freeze_weights(self):
        """Freeze model and cache original weights."""
        print(f"\n{'='*80}")
        print(f"PUN-IT Initialization: Target {self.sparsity*100:.0f}% sparsity")
        print(f"{'='*80}")
        
        # 1. Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. Cache weights for fast patching
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if self._is_excluded(name):
                    print(f"[SKIP]  {name}")
                    continue
                
                # Store reference to the original frozen weight tensor
                self.original_weights[name] = module.weight.detach().clone().to(self.device)
                self.layer_info[name] = {
                    'module': module,
                    'type': self._classify_layer(name)
                }
        print("✓ Model frozen and weights cached")
    
    def _initialize_probabilities(self):
        """Initialize logits such that initial probability ~= density, with optional noise."""
        init_density = 1.0 - self.sparsity
        # Inverse sigmoid: log(p / (1-p))
        init_logit = math.log(init_density / (1 - init_density + 1e-8))
        
        total_params = 0
        for name, weight in self.original_weights.items():
            # Create logits with gradients enabled
            # Add noise to break symmetry (important for optimization!)
            if self.init_noise > 0:
                logits = torch.full_like(weight, init_logit) + self.init_noise * torch.randn_like(weight)
            else:
                logits = torch.full_like(weight, init_logit)
            logits.requires_grad = True
            self.prob_params[name] = nn.Parameter(logits)
            total_params += weight.numel()
        
        noise_str = f" (noise std={self.init_noise})" if self.init_noise > 0 else ""
        print(f"Initialized {len(self.prob_params)} learnable masks ({total_params:,} params){noise_str}")

    def _gumbel_softmax_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Gumbel-Softmax sampling as per PUN-IT paper Eq. 7.
        
        m̂_j = exp((log(p_j) + G_l) / T) / [exp((log(p_j) + G_l) / T) + exp((log(1-p_j) + G_k) / T)]
        
        Uses TWO independent Gumbel samples (G_l, G_k).
        """
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        
        # Sample TWO independent Gumbel(0,1) random variables
        # G ~ -log(-log(U)) where U ~ Uniform(0,1)
        u_l = torch.rand_like(probs, device=self.device).clamp(1e-8, 1 - 1e-8)
        u_k = torch.rand_like(probs, device=self.device).clamp(1e-8, 1 - 1e-8)
        G_l = -torch.log(-torch.log(u_l))
        G_k = -torch.log(-torch.log(u_k))
        
        # Eq. 7: Gumbel-Softmax for binary (Bernoulli)
        log_p = torch.log(probs)
        log_1_minus_p = torch.log(1 - probs)
        
        # Clamp exponents to prevent overflow (exp(88) ≈ max float32)
        exp_arg_1 = ((log_p + G_l) / self.temperature).clamp(-80, 80)
        exp_arg_2 = ((log_1_minus_p + G_k) / self.temperature).clamp(-80, 80)
        
        numerator = torch.exp(exp_arg_1)
        denominator = numerator + torch.exp(exp_arg_2)
        
        y = numerator / (denominator + 1e-8)
        return y
    
    def _compute_hard_masks_for_step(self) -> Dict[str, torch.Tensor]:
        """Compute current hard masks from logits (for Hamming distance tracking)."""
        all_probs = []
        for name, logits in self.prob_params.items():
            all_probs.append(torch.sigmoid(logits).detach().flatten())
        all_probs = torch.cat(all_probs)
        
        target_density = 1.0 - self.sparsity
        k = int(all_probs.numel() * target_density)
        if k > 0:
            threshold = torch.topk(all_probs, k)[0][-1].item()
        else:
            threshold = 1.0
        
        masks = {}
        for name, logits in self.prob_params.items():
            probs = torch.sigmoid(logits).detach()
            masks[name] = (probs >= threshold).float()
        return masks
    
    def _compute_hamming_distance(
        self, 
        masks_prev: Dict[str, torch.Tensor], 
        masks_curr: Dict[str, torch.Tensor]
    ) -> Tuple[int, int]:
        """Compute Hamming distance between two mask sets."""
        total_diff = 0
        total_params = 0
        for name in masks_curr:
            if name in masks_prev:
                diff = (masks_prev[name] != masks_curr[name]).sum().item()
                total_diff += diff
                total_params += masks_curr[name].numel()
        return int(total_diff), total_params

    def find_masks(
        self,
        dataloader: DataLoader,
        criterion: nn.Module = None,
        num_steps: int = 5000,
        lr: float = 0.01,
        sparsity_weight: float = 100.0,  # Increased weight helps convergence
        print_freq: int = 50,
    ) -> Dict[str, torch.Tensor]:
        
        if criterion is None:
            criterion = nn.MSELoss()
        
        # Setup Optimizer
        optimizer = torch.optim.Adam(self.prob_params.values(), lr=lr)
        
        print(f"\n{'='*80}")
        print(f"PUN-IT Mask Search: {num_steps} steps")
        print(f"{'='*80}")
        
        data_iter = iter(dataloader)
        self.model.train()  # Ensure BN stats update if needed
        
        # Track previous masks for Hamming distance
        prev_masks = None
        
        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                input_data, target = batch[0], batch[1]
            else:
                input_data = target = batch
            
            input_data, target = input_data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # 1. Sample Soft Masks
            soft_masks = {name: self._gumbel_softmax_sample(logits) 
                         for name, logits in self.prob_params.items()}
            
            # 2. PATCH WEIGHTS (The "Perfect" Fix)
            # We temporarily replace the weight parameter with a computed tensor.
            # This maintains the gradient graph from Loss -> Weight -> Mask -> Logits
            for name, info in self.layer_info.items():
                module = info['module']
                del module.weight  # Delete the parameter attribute
                # Assign the tensor (original_frozen * soft_mask)
                module.weight = self.original_weights[name] * soft_masks[name]
            
            # 3. Forward & Loss
            output = self.model(input_data)
            recon_loss = criterion(output, target)
            
            # 4. Sparsity Loss: KL divergence as per PUN-IT paper Eq. 5
            # KL(Ber(p) || Ber(p_0)) on PROBABILITIES p, not on samples m
            # where p_0 = s/d = target density
            p_0 = 1.0 - self.sparsity  # target density (paper sets p_0 = s/d)
            
            kl_loss = 0.0
            total_params = 0
            prob_density_sum = 0.0
            for name, logits in self.prob_params.items():
                p = torch.sigmoid(logits)  # TRUE probabilities, not Gumbel samples!
                # KL divergence for each Bernoulli: KL(Ber(p) || Ber(p_0))
                kl = p * torch.log((p + 1e-8) / (p_0 + 1e-8)) + \
                     (1 - p) * torch.log((1 - p + 1e-8) / (1 - p_0 + 1e-8))
                kl_loss += kl.sum()
                total_params += logits.numel()
                prob_density_sum += p.sum()
            
            # Average KL per parameter
            sparsity_loss = kl_loss / total_params
            # Actual density from probabilities (not samples)
            actual_density = prob_density_sum / total_params
            
            total_loss = recon_loss + sparsity_weight * sparsity_loss
            
            # 5. Backward
            total_loss.backward()
            
            # 6. RESTORE WEIGHTS
            # We must restore the original Parameters before the optimizer step
            # or the next iteration
            for name, info in self.layer_info.items():
                module = info['module']
                del module.weight  # Delete the computed tensor
                # Restore the frozen parameter
                module.weight = nn.Parameter(self.original_weights[name], requires_grad=False)
            
            optimizer.step()
            
            # Anneal temperature
            self.temperature = max(self.min_temperature, self.temperature * self.temperature_anneal)
            
            # Compute Hamming distance at print steps
            if step % print_freq == 0 or step == num_steps - 1:
                curr_masks = self._compute_hard_masks_for_step()
                
                if prev_masks is not None:
                    hamming_dist, total_params = self._compute_hamming_distance(prev_masks, curr_masks)
                    hamming_pct = hamming_dist / total_params * 100 if total_params > 0 else 0
                    hamming_str = f"Hamming: {hamming_dist:,} ({hamming_pct:.2f}%)"
                else:
                    hamming_str = "Hamming: N/A (first step)"
                
                print(f"Step {step:4d} | Recon: {recon_loss.item():.4f} | "
                      f"KL: {sparsity_loss.item():.4f} | "
                      f"Density: {actual_density.item()*100:.2f}% (Target: {p_0*100:.0f}%) | "
                      f"Temp: {self.temperature:.3f} | {hamming_str}")
                
                prev_masks = curr_masks

        # Get final masks and print stats
        final_masks = self._get_hard_masks()
        self._print_mask_statistics(final_masks)
        return final_masks

    def _get_hard_masks(self) -> Dict[str, torch.Tensor]:
        """Convert logits to binary masks using global thresholding."""
        # Collect all probs
        all_probs = []
        for name, logits in self.prob_params.items():
            all_probs.append(torch.sigmoid(logits).detach().flatten())
        all_probs = torch.cat(all_probs)
        
        # Find global threshold
        target_density = 1.0 - self.sparsity
        k = int(all_probs.numel() * target_density)
        threshold = torch.topk(all_probs, k)[0][-1].item()
        
        print(f"\nBinarization Threshold: {threshold:.4f}")
        
        masks = {}
        for name, logits in self.prob_params.items():
            probs = torch.sigmoid(logits).detach()
            masks[name] = (probs >= threshold).float()
            
        return masks

    def _print_mask_statistics(self, masks: Dict[str, torch.Tensor]):
        """Print statistics about the found masks."""
        print(f"\n{'='*80}")
        print(f"PUN-IT Final Mask Statistics")
        print(f"{'='*80}")
        print(f"{'Type':<10} {'Layer Name':<40} {'Active':>12} {'Total':>12} {'Density':>8}")
        print(f"{'-'*80}")
        
        encoder_active, encoder_total = 0, 0
        decoder_active, decoder_total = 0, 0
        bottleneck_active, bottleneck_total = 0, 0
        
        for name, mask in masks.items():
            info = self.layer_info[name]
            active = int(mask.sum().item())
            total = mask.numel()
            density = active / total * 100
            layer_type = info['type']
            
            print(f"{layer_type.upper():<10} {name:<40} {active:>12,} {total:>12,} {density:>7.1f}%")
            
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
        
        total_active = encoder_active + decoder_active + bottleneck_active
        total_all = encoder_total + decoder_total + bottleneck_total
        
        print(f"\nDensity by Region:")
        if encoder_total > 0:
            print(f"  Encoder:    {encoder_active:>10,} / {encoder_total:>10,} ({encoder_active/encoder_total*100:.1f}%)")
        if decoder_total > 0:
            print(f"  Decoder:    {decoder_active:>10,} / {decoder_total:>10,} ({decoder_active/decoder_total*100:.1f}%)")
        if bottleneck_total > 0:
            print(f"  Bottleneck: {bottleneck_active:>10,} / {bottleneck_total:>10,} ({bottleneck_active/bottleneck_total*100:.1f}%)")
        
        print(f"  TOTAL:      {total_active:>10,} / {total_all:>10,} ({total_active/total_all*100:.1f}%)")
        print(f"  Target was: {(1-self.sparsity)*100:.1f}% density ({self.sparsity*100:.1f}% sparse)")
        print(f"{'='*80}\n")

    def apply_masks_to_model(self, masks: Dict[str, torch.Tensor] = None):
        if masks is None:
            masks = self._get_hard_masks()
        
        # Unfreeze and apply
        for param in self.model.parameters():
            param.requires_grad = True
            
        for name, mask in masks.items():
            module = self.layer_info[name]['module']
            with torch.no_grad():
                module.weight.data *= mask.to(self.device)
                
        print("✓ Masks applied and weights unfrozen")
        return masks


def find_punit_masks(
    model: nn.Module,
    dataloader: DataLoader,
    sparsity: float = 0.5,
    num_steps: int = 5000,
    lr: float = 0.01,
    device: str = 'cuda',
    exclude_first_last: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to find PUN-IT masks.
    
    Args:
        model: Model to find masks for
        dataloader: Training data
        sparsity: Target sparsity
        num_steps: Optimization steps
        lr: Learning rate
        device: Device
        exclude_first_last: Exclude first/last layers from pruning
    
    Returns:
        Dictionary of binary masks
    
    Example:
        model = UNet(...)
        masks = find_punit_masks(model, train_loader, sparsity=0.6)
        # Now train with these fixed masks
    """
    exclude = []
    if exclude_first_last:
        exclude = [
            'first', 'initial', 'final', 'last', 'output',
            'down_sample_layers.0.layers.0',
            'up_conv.3.1',
        ]
    
    finder = PUNITMaskFinder(
        model=model,
        sparsity=sparsity,
        exclude_layers=exclude,
        device=device,
    )
    
    masks = finder.find_masks(
        dataloader=dataloader,
        num_steps=num_steps,
        lr=lr,
    )
    
    finder.apply_masks_to_model(masks)
    
    return masks


# Alias for easier import
PunitMaskFinder = PUNITMaskFinder


# ==============================================================================
# SNIP: Single-shot Network Pruning (RECOMMENDED)
# ==============================================================================

class SNIPMaskFinder:
    """
    SNIP: Single-shot Network Pruning based on Connection Sensitivity.
    
    Much simpler and more effective than PUN-IT:
    1. Do ONE forward-backward pass
    2. Compute importance score = |gradient × weight| for each weight
    3. Keep top-k weights globally based on importance
    
    This naturally finds which weights contribute most to loss reduction!
    
    Usage:
        finder = SNIPMaskFinder(model, sparsity=0.6)
        masks = finder.find_masks(dataloader)
        finder.apply_masks_to_model(masks)
    """
    
    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        exclude_layers: List[str] = None,
        device: str = 'cuda',
    ):
        self.model = model
        self.sparsity = sparsity
        self.exclude_layers = exclude_layers or []
        self.device = device
        
        self.layer_info: Dict[str, dict] = {}
        self.excluded_layers: Dict[str, dict] = {}
        
        self._collect_layers()
    
    def _is_excluded(self, name: str) -> bool:
        for pattern in self.exclude_layers:
            if pattern in name:
                return True
        return False
    
    def _classify_layer(self, name: str) -> str:
        name_lower = name.lower()
        if 'down' in name_lower or 'encoder' in name_lower:
            return 'encoder'
        elif 'up' in name_lower or 'decoder' in name_lower:
            return 'decoder'
        elif 'bottleneck' in name_lower or 'conv_' in name_lower:
            return 'bottleneck'
        return 'unknown'
    
    def _collect_layers(self):
        """Collect all layers that will be pruned."""
        print(f"\n{'='*80}")
        print(f"SNIP Initialization: Target {self.sparsity*100:.0f}% sparsity")
        print(f"{'='*80}")
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                numel = weight.numel()
                
                if self._is_excluded(name):
                    self.excluded_layers[name] = {
                        'type': self._classify_layer(name),
                        'numel': numel,
                        'module': module,
                    }
                    print(f"[SKIP]  {name}: {numel:,} params")
                    continue
                
                self.layer_info[name] = {
                    'type': self._classify_layer(name),
                    'numel': numel,
                    'module': module,
                }
                print(f"[SNIP]  {name}: {numel:,} params")
        
        total = sum(info['numel'] for info in self.layer_info.values())
        print(f"\nTotal prunable parameters: {total:,}")
        print(f"{'='*80}\n")
    
    def find_masks(
        self,
        dataloader: DataLoader,
        criterion: nn.Module = None,
        num_batches: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Find masks using SNIP (gradient × weight importance).
        
        Args:
            dataloader: Training data
            criterion: Loss function (default: MSE)
            num_batches: Number of batches to average gradients over
        
        Returns:
            Dictionary of binary masks
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.model.to(self.device)
        self.model.train()
        
        # Enable gradients
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Accumulate gradient × weight scores
        scores = {name: torch.zeros_like(info['module'].weight) 
                  for name, info in self.layer_info.items()}
        
        print(f"Computing SNIP scores over {num_batches} batches...")
        
        data_iter = iter(dataloader)
        for batch_idx in range(num_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                input_data, target = batch[0], batch[1]
            else:
                input_data = target = batch
            
            input_data = input_data.to(self.device)
            target = target.to(self.device)
            
            # Forward
            self.model.zero_grad()
            output = self.model(input_data)
            loss = criterion(output, target)
            
            # Backward
            loss.backward()
            
            # Accumulate |gradient × weight|
            for name, info in self.layer_info.items():
                module = info['module']
                if module.weight.grad is not None:
                    scores[name] += (module.weight.grad * module.weight).abs()
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Average scores
        for name in scores:
            scores[name] /= num_batches
        
        # Global pruning: find threshold for target sparsity
        all_scores = torch.cat([s.flatten() for s in scores.values()])
        target_density = 1 - self.sparsity
        k = int(len(all_scores) * target_density)
        
        if k > 0:
            threshold = torch.topk(all_scores, k, largest=True).values[-1].item()
        else:
            threshold = float('inf')
        
        print(f"\nGlobal threshold: {threshold:.6f} (keeping top {target_density*100:.0f}%)")
        
        # Create masks
        masks = {}
        for name, score in scores.items():
            masks[name] = (score >= threshold).float()
        
        # Print statistics
        self._print_mask_statistics(masks)
        
        return masks
    
    def _print_mask_statistics(self, masks: Dict[str, torch.Tensor]):
        """Print mask statistics by layer and region."""
        print(f"\n{'='*80}")
        print(f"SNIP Final Mask Statistics")
        print(f"{'='*80}")
        print(f"{'Type':<10} {'Layer Name':<40} {'Active':>10} {'Total':>10} {'Density':>8}")
        print(f"{'-'*80}")
        
        encoder_active, encoder_total = 0, 0
        decoder_active, decoder_total = 0, 0
        bottleneck_active, bottleneck_total = 0, 0
        
        for name, mask in masks.items():
            info = self.layer_info[name]
            active = int(mask.sum().item())
            total = mask.numel()
            density = active / total * 100
            layer_type = info['type']
            
            print(f"{layer_type.upper():<10} {name:<40} {active:>10,} {total:>10,} {density:>7.1f}%")
            
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
        
        total_active = encoder_active + decoder_active + bottleneck_active
        total_all = encoder_total + decoder_total + bottleneck_total
        
        print(f"\nDensity by Region:")
        if encoder_total > 0:
            print(f"  Encoder:    {encoder_active:>10,} / {encoder_total:>10,} ({encoder_active/encoder_total*100:.1f}%)")
        if decoder_total > 0:
            print(f"  Decoder:    {decoder_active:>10,} / {decoder_total:>10,} ({decoder_active/decoder_total*100:.1f}%)")
        if bottleneck_total > 0:
            print(f"  Bottleneck: {bottleneck_active:>10,} / {bottleneck_total:>10,} ({bottleneck_active/bottleneck_total*100:.1f}%)")
        
        print(f"  TOTAL:      {total_active:>10,} / {total_all:>10,} ({total_active/total_all*100:.1f}%)")
        print(f"  Target was: {(1-self.sparsity)*100:.1f}% density ({self.sparsity*100:.1f}% sparse)")
        print(f"{'='*80}\n")
    
    def apply_masks_to_model(self, masks: Dict[str, torch.Tensor] = None):
        """Apply masks to model weights."""
        if masks is None:
            raise ValueError("Must provide masks")
        
        for name, mask in masks.items():
            module = self.layer_info[name]['module']
            with torch.no_grad():
                module.weight.data *= mask.to(module.weight.device)
        
        print("✓ SNIP masks applied to model")
        return masks


def find_snip_masks(
    model: nn.Module,
    dataloader: DataLoader,
    sparsity: float = 0.5,
    num_batches: int = 10,
    device: str = 'cuda',
    exclude_first_last: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to find SNIP masks (RECOMMENDED over PUN-IT).
    
    Example:
        model = UNet(...)
        masks = find_snip_masks(model, train_loader, sparsity=0.6)
        # Model is now pruned and ready to train
    """
    exclude = []
    if exclude_first_last:
        exclude = [
            'first', 'initial', 'final', 'last', 'output',
            'down_sample_layers.0.layers.0',
            'up_conv.3.1',
        ]
    
    finder = SNIPMaskFinder(
        model=model,
        sparsity=sparsity,
        exclude_layers=exclude,
        device=device,
    )
    
    masks = finder.find_masks(
        dataloader=dataloader,
        num_batches=num_batches,
    )
    
    finder.apply_masks_to_model(masks)
    
    return masks

