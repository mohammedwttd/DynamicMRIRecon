"""
RigL U-Net: Use the local UnetModel with RigL sparse training.

RigL works with ANY model - the RigLScheduler wraps it and handles sparsity.
No need for a separate architecture!

Usage in train.py:
    model = UnetModel(...)  # or any model
    if args.use_rigl:
        rigl_scheduler = RigLScheduler(model, sparsity=0.5, ...)
        # In training loop:
        loss.backward()
        rigl_scheduler.step()  # Updates sparse masks
        optimizer.step()

Local UnetModel (~3.35M params with num_chans=128):
- Uses ReLU activation
- Uses max_pool2d for downsampling  
- Uses bilinear interpolation for upsampling
- Standard Conv2d with bias
"""

# Use local UnetModel for RigL - consistent with mask finding scripts
from models.rec_models.models.unet_model import UnetModel as RigLUnet
from models.rec_models.models.unet_model import UnetModel as RigLLightUnet

# Note: The difference between RigL and RigLLight is just the num_chans
# passed from exp.py (32 vs 20). The model class is the same.

