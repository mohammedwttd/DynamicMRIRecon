"""
STAMP U-Net for MRI Reconstruction

Full official implementation from: https://github.com/nkdinsdale/STAMP.git
Paper: "STAMP: Simultaneous Training and Model Pruning for Low Data Regimes 
        in Medical Image Segmentation" (Medical Image Analysis, 2022)

Key components from official code:
- UNet2D: 2D adaptation of official STAMP UNet with per-layer targeted dropout
- FilterPrunner2D: Computes filter importance (Taylor/L1/L2/Random)
- PruningController2D: Manages pruning iterations
- new_weights_rankval: Computes new dropout probabilities from filter ranks
- update_droplayer: Updates dropout layers with new probabilities
"""

import torch
import torch.nn as nn
import numpy as np

# Import official STAMP 2D components
from models.rec_models.stamp.models.unet_model_targeted_dropout_2d import (
    UNet2D,
    update_droplayer,
    get_default_drop_probs,
)
from models.rec_models.stamp.pruning_tools import (
    FilterPrunner2D,
    PruningController2D,
    prune_conv_layer2d,
)


def new_weights_rankval(d, base_prob=0.10):
    """
    Official STAMP function to compute new dropout probabilities from filter ranks.
    
    This implements Algorithm 1 from the paper:
    - Filters with lower importance get higher dropout probability
    - Filters with higher importance get lower dropout probability
    
    From: https://github.com/nkdinsdale/STAMP/blob/main/pruning_functions.py
    """
    new_d = {}
    for key in d:
        array = d[key].numpy() if hasattr(d[key], 'numpy') else np.array(d[key])
        for i in array:
            if i not in new_d.keys():
                new_d[i] = key
            else:
                i += 1e-12
                new_d[i] = key
    o_keys = sorted(new_d.keys())

    i = len(o_keys)
    prune_dict = {j: 0 for j in range(17)}  # 17 dropout layers in UNet
    for val in o_keys:
        i -= 1
        index = new_d[val]
        prune_dict[index] += i

    for key in prune_dict:
        if key in d:
            arr = d[key].numpy() if hasattr(d[key], 'numpy') else np.array(d[key])
            prune_dict[key] = prune_dict[key] / len(arr)
        else:
            prune_dict[key] = 0
            
    max_val = max(prune_dict.values()) if max(prune_dict.values()) > 0 else 1
    for key in prune_dict:
        prune_dict[key] = prune_dict[key] / max_val

    for key in prune_dict:
        prune_dict[key] = prune_dict[key] * base_prob
    return prune_dict


class STAMPUnet(nn.Module):
    """
    STAMP UNet wrapper for DynamicMRIRecon compatibility.
    
    Uses the official STAMP UNet2D with targeted dropout.
    """
    
    def __init__(self, in_chans=1, out_chans=1, chans=32, num_pool_layers=4,
                 drop_prob=0.0, b_drop=0.1, mode='Taylor'):
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.b_drop = b_drop
        self.mode = mode
        
        # Initialize dropout probabilities
        self.drop_probs = get_default_drop_probs(b_drop)
        
        # Create the UNet
        self.unet = UNet2D(
            drop_probs=self.drop_probs,
            in_channels=in_chans,
            out_channels=out_chans,
            init_features=chans
        )
    
    def forward(self, x):
        return self.unet(x)
    
    def update_dropout_probs(self, new_probs):
        """Update dropout probabilities."""
        self.drop_probs = new_probs
        update_droplayer(self.unet, new_probs)


class STAMPScheduler:
    """
    Official STAMP scheduler for training with adaptive targeted dropout and pruning.
    
    Implements the full STAMP algorithm:
    1. Train network with targeted dropout
    2. Compute filter importance using FilterPrunner (on RECONSTRUCTED images)
    3. Update dropout probabilities based on importance (Algorithm 1)
    4. Optionally prune least important filters
    5. Recovery training after pruning
    
    Based on: https://github.com/nkdinsdale/STAMP
    
    IMPORTANT: Call collect_batch() during training to collect post-subsampling images.
    The images must be AFTER the subsampling layer (image domain), not raw k-space!
    """
    
    def __init__(self, model, b_drop=0.1, prune_epochs=None, prune_percentage=10,
                 prune_ratio=None, recovery_epochs=5, total_epochs=50, mode='Taylor',
                 max_collected_batches=20):
        """
        Args:
            model: STAMPUnet model
            b_drop: Base dropout probability (default: 0.1 = 10%)
            prune_epochs: List of epochs to prune at (None = auto-generate)
            prune_percentage: Number of filters to prune per iteration (official STAMP, default: 10)
            prune_ratio: Keep ratio per iteration (0.5 = keep 50%, DynamicMRIRecon style)
                         If provided, overrides prune_percentage calculation
            recovery_epochs: Training epochs between prune iterations
            total_epochs: Total training epochs
            mode: Importance computation mode ('Taylor', 'L1', 'L2', 'Random')
            max_collected_batches: Max batches to store for importance computation
        """
        self.model = model
        self.b_drop = b_drop
        self.prune_ratio = prune_ratio  # Store for dynamic calculation
        # Official STAMP uses fixed number of filters per iteration
        self.prune_percentage = prune_percentage
        self.recovery_epochs = recovery_epochs
        self.total_epochs = total_epochs
        self.mode = mode
        self.current_epoch = 0
        self.pruning_done = []
        self.use_cuda = torch.cuda.is_available()
        
        # Image collection for importance computation
        # These should be POST-SUBSAMPLING images (after k-space to image conversion)
        self.collected_inputs = []  # Input images (undersampled)
        self.collected_targets = []  # Target images (ground truth)
        self.max_collected_batches = max_collected_batches
        
        # Auto-generate prune epochs if not provided
        if prune_epochs is None or len(prune_epochs) == 0:
            self.prune_epochs = []
            epoch = recovery_epochs
            while epoch < int(total_epochs * 0.8):
                self.prune_epochs.append(epoch)
                epoch += recovery_epochs
        else:
            self.prune_epochs = list(prune_epochs)
        
        # Initialize pruning controller
        self.prunner = None
        
        # Track if architecture changed (for optimizer recreation)
        self.architecture_changed = False
        
        print(f"[STAMP] Initialized official STAMP scheduler")
        print(f"[STAMP] Mode: {mode}, Base dropout: {b_drop}")
        print(f"[STAMP] Prune epochs: {self.prune_epochs}")
        print(f"[STAMP] Will collect {max_collected_batches} batches for importance computation")
        if prune_ratio is not None:
            print(f"[STAMP] Keep ratio per iteration: {prune_ratio:.2f} (prune {(1-prune_ratio)*100:.0f}%)")
        else:
            print(f"[STAMP] Filters to prune per iteration: {prune_percentage} (official STAMP style)")
    
    def collect_batch(self, input_img, target_img):
        """
        Collect a batch of images for importance computation.
        
        IMPORTANT: Call this with POST-SUBSAMPLING images!
        - input_img: Output of subsampling layer (undersampled image) [B, H, W] or [B, 1, H, W]
        - target_img: Ground truth image [B, H, W] or [B, 1, H, W]
        
        Example usage in train_epoch:
            # After subsampling, before reconstruction
            subsampled_img = model.module.subsampling(kspace_input)
            if stamp_scheduler is not None:
                stamp_scheduler.collect_batch(subsampled_img.detach(), target.detach())
            output = model.module.reconstruction_model(subsampled_img)
        """
        if len(self.collected_inputs) >= self.max_collected_batches:
            return  # Already collected enough
        
        # Ensure correct shape [B, 1, H, W]
        if input_img.dim() == 3:
            input_img = input_img.unsqueeze(1)
        if target_img.dim() == 3:
            target_img = target_img.unsqueeze(1)
        
        # Store on CPU to save GPU memory
        self.collected_inputs.append(input_img.detach().cpu())
        self.collected_targets.append(target_img.detach().cpu())
    
    def clear_collected(self):
        """Clear collected batches (call at start of each epoch)."""
        self.collected_inputs = []
        self.collected_targets = []
    
    def compute_importance(self, criterion):
        """
        Compute filter importance using collected images.
        
        Returns:
            dict: Filter ranks per layer
        """
        if len(self.collected_inputs) == 0:
            print("[STAMP] WARNING: No images collected! Call collect_batch() during training.")
            return None, None
        
        # Get the inner UNet for pruning
        unet = self.model.unet if hasattr(self.model, 'unet') else self.model
        
        # Create pruning controller
        controller = PruningController2D(
            unet=unet,
            criterion=criterion,
            prune_percentage=self.prune_percentage,
            mode=self.mode,
            use_cuda=self.use_cuda
        )
        
        # Compute ranks over collected images
        device = next(unet.parameters()).device
        controller.pruner.reset()
        unet.train()
        
        print(f"[STAMP] Computing importance using {len(self.collected_inputs)} collected batches...")
        
        for i, (input_batch, target_batch) in enumerate(zip(self.collected_inputs, self.collected_targets)):
            try:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                controller.compute_ranks(input_batch, target_batch)
            except Exception as e:
                print(f"[STAMP] Batch {i} failed: {e}")
                continue
        
        ranks = controller.get_ranks()
        return ranks, controller
    
    def step(self, epoch, criterion=None, data_loader=None):
        """
        STAMP step at end of each epoch.
        
        1. Compute filter importance (using collected images)
        2. Update dropout probabilities (Algorithm 1)
        3. Prune if at prune epoch
        4. Clear collected images for next epoch
        
        Args:
            epoch: Current epoch number
            criterion: Loss criterion for importance computation (default: L1Loss)
            data_loader: DEPRECATED - not used, kept for backward compatibility
        """
        self.current_epoch = epoch
        
        if criterion is None:
            criterion = nn.L1Loss()
        
        if len(self.collected_inputs) == 0:
            print(f"[STAMP] Epoch {epoch}: No images collected, skipping importance update")
            print(f"[STAMP] TIP: Call stamp_scheduler.collect_batch(input, target) during training")
            return self.b_drop
        
        try:
            # Compute filter importance from collected images
            ranks, controller = self.compute_importance(criterion)
            
            if ranks is not None and len(ranks) > 0:
                # Update dropout probabilities based on importance (Algorithm 1)
                new_drop = new_weights_rankval(ranks, self.b_drop)
                self.model.update_dropout_probs(new_drop)
                
                if epoch % 10 == 0:
                    print(f"[STAMP] Epoch {epoch}: Updated dropout probabilities")
                    print(f"[STAMP] Sample probs: {list(new_drop.values())[:5]}")
                
                # Check if we should prune
                if epoch in self.prune_epochs and epoch not in self.pruning_done:
                    print(f"\n[STAMP] === Physical Filter Pruning at epoch {epoch} ===")
                    
                    # Calculate number of filters to prune
                    # If prune_ratio is provided, use it to calculate
                    # Otherwise use prune_percentage directly (official STAMP)
                    if self.prune_ratio is not None:
                        total_filters = controller.total_num_filters()
                        filters_to_remove = int(total_filters * (1 - self.prune_ratio))
                        filters_to_remove = max(1, filters_to_remove)  # At least 1
                        print(f"[STAMP] Keep ratio: {self.prune_ratio:.2f} -> removing {filters_to_remove} of {total_filters} filters")
                    else:
                        filters_to_remove = self.prune_percentage
                    
                    # Perform actual physical filter pruning (removes filters from network)
                    returned_ranks = controller.prune(num_filters=filters_to_remove, physical_prune=True)
                    
                    # Update the model reference (architecture has changed!)
                    if hasattr(self.model, 'unet'):
                        self.model.unet = controller.unet
                    else:
                        self.model = controller.unet
                    
                    # Mark that architecture changed - optimizer needs to be recreated
                    self.architecture_changed = True
                    
                    # Update dropout after pruning
                    if returned_ranks is not None:
                        try:
                            new_drop = new_weights_rankval(returned_ranks, self.b_drop)
                            self.model.update_dropout_probs(new_drop)
                        except Exception as e:
                            print(f"[STAMP] Warning: Could not update dropout after pruning: {e}")
                    
                    self.pruning_done.append(epoch)
                    
                    # Count remaining parameters
                    total_params = sum(p.numel() for p in self.model.parameters())
                    print(f"[STAMP] Pruning complete. Total pruning iterations: {len(self.pruning_done)}")
                    print(f"[STAMP] Model now has {total_params:,} parameters")
                    print(f"[STAMP] WARNING: Optimizer should be recreated for the pruned model!")
                    
        except Exception as e:
            print(f"[STAMP] Warning: Importance computation failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue training with current dropout probabilities
        
        # Clear collected images for next epoch
        self.clear_collected()
        
        return self.b_drop
    
    def get_status(self):
        """Get current scheduler status."""
        return {
            'epoch': self.current_epoch,
            'b_drop': self.b_drop,
            'mode': self.mode,
            'prune_epochs': self.prune_epochs,
            'prunings_done': len(self.pruning_done),
            'next_prune': next((e for e in self.prune_epochs if e > self.current_epoch), None),
            'architecture_changed': self.architecture_changed,
        }
    
    def check_and_reset_architecture_changed(self):
        """
        Check if architecture changed and reset the flag.
        
        Returns:
            bool: True if architecture changed since last check
        """
        changed = self.architecture_changed
        self.architecture_changed = False
        return changed
    
    def get_model(self):
        """Get the current model (may have changed architecture after pruning)."""
        return self.model


# Alias for compatibility
STAMPReconUnet = STAMPUnet

__all__ = ['STAMPUnet', 'STAMPReconUnet', 'STAMPScheduler', 'new_weights_rankval']
