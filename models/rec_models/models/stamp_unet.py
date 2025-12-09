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
    2. Compute filter importance using FilterPrunner
    3. Update dropout probabilities based on importance (Algorithm 1)
    4. Optionally prune least important filters
    5. Recovery training after pruning
    
    Based on: https://github.com/nkdinsdale/STAMP
    """
    
    def __init__(self, model, b_drop=0.1, prune_epochs=None, prune_percentage=10,
                 prune_ratio=None, recovery_epochs=5, total_epochs=50, mode='Taylor'):
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
        
        print(f"[STAMP] Initialized official STAMP scheduler")
        print(f"[STAMP] Mode: {mode}, Base dropout: {b_drop}")
        print(f"[STAMP] Prune epochs: {self.prune_epochs}")
        if prune_ratio is not None:
            print(f"[STAMP] Keep ratio per iteration: {prune_ratio:.2f} (prune {(1-prune_ratio)*100:.0f}%)")
        else:
            print(f"[STAMP] Filters to prune per iteration: {prune_percentage} (official STAMP style)")
    
    def compute_importance(self, data_loader, criterion):
        """
        Compute filter importance using official FilterPrunner.
        
        Returns:
            dict: Filter ranks per layer
        """
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
        
        # Compute ranks over data
        ranks = controller.compute_ranks_epoch(data_loader, num_batches=10)
        
        return ranks, controller
    
    def step(self, epoch, data_loader=None, criterion=None):
        """
        STAMP step at end of each epoch.
        
        1. Compute filter importance
        2. Update dropout probabilities (Algorithm 1)
        3. Prune if at prune epoch
        
        Args:
            epoch: Current epoch number
            data_loader: Training data loader
            criterion: Loss criterion for importance computation
        """
        self.current_epoch = epoch
        
        if data_loader is None or criterion is None:
            # Can't compute importance without data
            return self.b_drop
        
        try:
            # Compute filter importance
            ranks, controller = self.compute_importance(data_loader, criterion)
            
            if ranks is not None and len(ranks) > 0:
                # Update dropout probabilities based on importance (Algorithm 1)
                new_drop = new_weights_rankval(ranks, self.b_drop)
                self.model.update_dropout_probs(new_drop)
                
                if epoch % 10 == 0:
                    print(f"[STAMP] Epoch {epoch}: Updated dropout probabilities")
                    print(f"[STAMP] Sample probs: {list(new_drop.values())[:5]}")
                
                # Check if we should prune
                if epoch in self.prune_epochs and epoch not in self.pruning_done:
                    print(f"\n[STAMP] === Pruning at epoch {epoch} ===")
                    
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
                    
                    # Perform actual filter pruning
                    returned_ranks = controller.prune(num_filters=filters_to_remove)
                    
                    # Update the model reference
                    if hasattr(self.model, 'unet'):
                        self.model.unet = controller.unet
                    else:
                        self.model = controller.unet
                    
                    # Update dropout after pruning
                    if returned_ranks is not None:
                        new_drop = new_weights_rankval(returned_ranks, self.b_drop)
                        self.model.update_dropout_probs(new_drop)
                    
                    self.pruning_done.append(epoch)
                    print(f"[STAMP] Pruning complete. Total pruning iterations: {len(self.pruning_done)}")
                    
        except Exception as e:
            print(f"[STAMP] Warning: Importance computation failed: {e}")
            # Continue training with current dropout probabilities
        
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
        }


# Alias for compatibility
STAMPReconUnet = STAMPUnet

__all__ = ['STAMPUnet', 'STAMPReconUnet', 'STAMPScheduler', 'new_weights_rankval']
