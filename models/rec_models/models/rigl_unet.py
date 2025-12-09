"""
RigL U-Net: Just use the existing Unet with RigL sparse training.

RigL works with ANY model - the RigLScheduler wraps it and handles sparsity.
No need for a separate architecture!

Usage in train.py:
    model = Unet(...)  # or any model
    if args.use_rigl:
        rigl_scheduler = RigLScheduler(model, sparsity=0.5, ...)
        # In training loop:
        loss.backward()
        rigl_scheduler.step()  # Updates sparse masks
        optimizer.step()
"""

# Just re-export the existing Unet - RigL works with any model!
from .unet_model import UnetModel as RigLUnet
from .unet_model import UnetModel as RigLLightUnet

# Note: The difference between RigL and RigLLight is just the num_chans
# passed from exp.py (32 vs 20). The model class is the same.

