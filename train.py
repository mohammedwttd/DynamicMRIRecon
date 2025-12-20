import logging
import pathlib
import random
import shutil
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
# sys.path.insert(0, '/home/tomerweiss/multiPILOT2')

import fastmri
import numpy as np
# np.seterr('raise')
import torch

# Disable cuFFT plan caching to prevent OOM/internal errors
torch.backends.cuda.cufft_plan_cache[0].max_size = 0
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as tv_transforms
from PIL import Image
from common.args import Args
from data import transforms
from data.mri_data import SliceData
import matplotlib

from models.rec_models.models.recon_net import ReconNet
from models.rec_models.models.vit_model import VisionTransformer

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.subsampling_model import Subsampling_Model, SubsamplingBinary
from scipy.spatial import distance_matrix
#from tsp_solver.greedy import solve_tsp
import scipy.io as sio
from common.utils import get_vel_acc
from common.evaluate import psnr, ssim
from fastmri.losses import SSIMLoss
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import random

import torch.nn as nn
import torch.optim as optim
import logging
import torchvision.models as models


def _get_train_dataset_name(args) -> str:
    """
    Human-readable training dataset name to embed in run directories.
    - Defaults to 'fastmri' for MRI training (args.dataset == 'mri')
    - Defaults to 'imagenet' for ImageNet training
    - Can be overridden via --train-dataset-name
    """
    override = getattr(args, "train_dataset_name", None)
    if isinstance(override, str) and override.strip():
        return override.strip()
    dataset_type = getattr(args, "dataset", "mri")
    if dataset_type == "imagenet":
        return "imagenet"
    return "fastmri"


def _count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _format_param_count(n: int) -> str:
    # Path-safe, readable.
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


class DecoderOnlyMaskTrainer:
    """
    Decoder-only inverse mask training: 
    - Only DECODER's pruned (zero) weights are TRAINABLE
    - Encoder and bottleneck pruned weights stay FROZEN at 0
    - All active (non-zero) weights from RigL are FROZEN
    """
    def __init__(self, model, masks, device='cuda', reinit='xavier'):
        self.model = model
        self.masks = {}
        self.inverse_masks = {}  # Only decoder layers
        self.device = device
        self.frozen_weights = {}
        self.reinit = reinit
        
        encoder_trainable = 0
        decoder_trainable = 0
        bottleneck_trainable = 0
        total_active = 0
        
        # Store masks and compute inverse only for decoder layers
        for name, mask in masks.items():
            self.masks[name] = mask.to(device)
            layer_type = self._classify_layer(name)
            
            # Count statistics
            active = (mask == 1).sum().item()
            pruned = (mask == 0).sum().item()
            total_active += active
            
            if layer_type == 'decoder':
                # Decoder: pruned weights are trainable
                self.inverse_masks[name] = (1 - mask).to(device)
                decoder_trainable += pruned
            else:
                # Encoder/bottleneck: all weights frozen (including pruned ones)
                self.inverse_masks[name] = torch.zeros_like(mask).to(device)
                if layer_type == 'encoder':
                    encoder_trainable += 0  # Not trainable
                else:
                    bottleneck_trainable += 0  # Not trainable
        
        # Save the frozen (active) weights before any training
        self._save_frozen_weights()
        
        # Xavier reinitialize only decoder's pruned weights
        if reinit == 'xavier':
            self._xavier_reinit_pruned_weights()
        
        total_params = sum(m.numel() for m in self.masks.values())
        print(f"\n{'='*70}")
        print(f"Decoder-Only Mask Training: Only train DECODER's pruned weights")
        print(f"{'='*70}")
        print(f"  FROZEN (all active weights): {total_active:,}")
        print(f"  FROZEN (encoder/bottleneck pruned): kept at 0")
        print(f"  TRAINABLE (decoder pruned): {decoder_trainable:,}")
        print(f"  Reinit method: {reinit}")
        print(f"{'='*70}\n")
    
    def _classify_layer(self, name):
        """Classify layer as encoder, decoder, or bottleneck."""
        name_lower = name.lower()
        if 'down' in name_lower or 'encoder' in name_lower:
            return 'encoder'
        elif 'up' in name_lower or 'decoder' in name_lower:
            return 'decoder'
        else:
            return 'bottleneck'
    
    def _save_frozen_weights(self):
        """Save a copy of the frozen (active) weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                if mask_key in self.masks:
                    # Save only the active weights (masked by original mask)
                    self.frozen_weights[mask_key] = (param.data * self.masks[mask_key]).clone()
    
    def _xavier_reinit_pruned_weights(self):
        """Xavier reinitialize only decoder's pruned (trainable) weights."""
        reinit_count = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                if mask_key in self.inverse_masks and name.endswith('.weight'):
                    inverse_mask = self.inverse_masks[mask_key]
                    
                    # Only reinit if there are trainable positions (decoder only)
                    if inverse_mask.sum() > 0:
                        if len(param.shape) >= 2:
                            xavier_init = torch.empty_like(param.data)
                            torch.nn.init.xavier_uniform_(xavier_init)
                        else:
                            xavier_init = torch.randn_like(param.data) * 0.01
                        
                        # Apply only to decoder's pruned positions
                        param.data.copy_(
                            self.frozen_weights[mask_key] +  # Frozen (active) weights
                            xavier_init * inverse_mask       # Xavier init for decoder pruned
                        )
                        reinit_count += 1
        
        print(f"  [Xavier Reinit] Reinitialized decoder pruned weights in {reinit_count} layers")
    
    def zero_frozen_gradients(self):
        """Zero out gradients for all frozen weights - only train decoder pruned."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                if mask_key in self.inverse_masks:
                    # Only allow gradients where inverse_mask = 1 (decoder pruned only)
                    param.grad.mul_(self.inverse_masks[mask_key])
    
    def step(self):
        """After optimizer step, restore frozen weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                if mask_key in self.frozen_weights:
                    # Combine: frozen_weights (active) + new_weights (decoder pruned only)
                    param.data.copy_(
                        self.frozen_weights[mask_key] + 
                        param.data * self.inverse_masks[mask_key]
                    )
    
    def print_sparsity(self):
        """Print current model statistics."""
        total = 0
        zeros = 0
        for name, param in self.model.named_parameters():
            if 'reconstruction_model' in name and 'weight' in name:
                total += param.numel()
                zeros += (param == 0).sum().item()
        print(f"  [DecoderOnlyMask] Zeros: {zeros:,}/{total:,} ({100*zeros/total:.1f}%)")


class FrozenMaskTrainer:
    """
    Inverse mask training: Train the PRUNED weights, freeze the ACTIVE weights.
    - RigL's learned (non-zero) weights are FROZEN
    - RigL's pruned (zero) weights are TRAINABLE (Xavier reinitialized)
    """
    def __init__(self, model, masks, device='cuda', reinit='xavier'):
        self.model = model
        self.masks = {}
        self.inverse_masks = {}
        self.device = device
        self.frozen_weights = {}
        self.reinit = reinit
        
        # Store masks and compute inverse
        for name, mask in masks.items():
            self.masks[name] = mask.to(device)
            self.inverse_masks[name] = (1 - mask).to(device)
        
        # Save the frozen (active) weights before any training
        self._save_frozen_weights()
        
        # Xavier reinitialize the trainable (pruned) weights
        if reinit == 'xavier':
            self._xavier_reinit_pruned_weights()
        
        # Count trainable vs frozen
        total_params = 0
        frozen_params = 0
        trainable_params = 0
        
        for name, mask in self.masks.items():
            total_params += mask.numel()
            frozen_params += (mask == 1).sum().item()
            trainable_params += (mask == 0).sum().item()
        
        print(f"\n{'='*70}")
        print(f"Inverse Mask Training: Train PRUNED weights, Freeze ACTIVE weights")
        print(f"{'='*70}")
        print(f"  FROZEN (RigL learned): {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"  TRAINABLE (RigL pruned): {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Reinit method: {reinit}")
        print(f"{'='*70}\n")
    
    def _save_frozen_weights(self):
        """Save a copy of the frozen (active) weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                if mask_key in self.masks:
                    # Save only the active weights (masked by original mask)
                    self.frozen_weights[mask_key] = (param.data * self.masks[mask_key]).clone()
    
    def _xavier_reinit_pruned_weights(self):
        """Xavier reinitialize the pruned (trainable) weights."""
        reinit_count = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                if mask_key in self.inverse_masks and name.endswith('.weight'):
                    inverse_mask = self.inverse_masks[mask_key]
                    
                    # Only reinit if there are pruned positions
                    if inverse_mask.sum() > 0:
                        # Create Xavier initialized tensor
                        if len(param.shape) >= 2:
                            # For conv/linear weights: use xavier_uniform
                            xavier_init = torch.empty_like(param.data)
                            torch.nn.init.xavier_uniform_(xavier_init)
                        else:
                            # For 1D params (rare): use normal init
                            xavier_init = torch.randn_like(param.data) * 0.01
                        
                        # Apply only to pruned positions (inverse_mask = 1)
                        # Keep frozen weights unchanged (mask = 1)
                        param.data.copy_(
                            self.frozen_weights[mask_key] +  # Frozen (active) weights
                            xavier_init * inverse_mask       # Xavier init for pruned weights
                        )
                        reinit_count += 1
        
        print(f"  [Xavier Reinit] Reinitialized pruned weights in {reinit_count} layers")
    
    def zero_frozen_gradients(self):
        """Zero out gradients for frozen (active) weights - only train pruned weights."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                if mask_key in self.inverse_masks:
                    # Only allow gradients where inverse_mask = 1 (pruned positions)
                    param.grad.mul_(self.inverse_masks[mask_key])
    
    def step(self):
        """After optimizer step, restore frozen weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                if mask_key in self.frozen_weights:
                    # Combine: frozen_weights (active) + new_weights (pruned)
                    param.data.copy_(
                        self.frozen_weights[mask_key] + 
                        param.data * self.inverse_masks[mask_key]
                    )
    
    def print_sparsity(self):
        """Print current model statistics."""
        total = 0
        zeros = 0
        for name, param in self.model.named_parameters():
            if 'reconstruction_model' in name and 'weight' in name:
                total += param.numel()
                zeros += (param == 0).sum().item()
        print(f"  [InverseMask] Zeros: {zeros:,}/{total:,} ({100*zeros/total:.1f}%)")
    
    def single_shot_adapt(self, input_data, target, criterion, num_steps=20, lr=1e-4):
        """
        Single-shot learning: Adapt model on a single example with multiple gradient steps.
        Only trains the pruned weights (frozen weights stay fixed).
        Saves and restores best weights (lowest loss) during adaptation.
        
        Args:
            input_data: Single input tensor (B, C, H, W) or (B, C, H, W, 2)
            target: Target tensor
            criterion: Loss function
            num_steps: Number of gradient steps (default: 20)
            lr: Learning rate for adaptation (default: 1e-4)
        
        Returns:
            losses: List of losses during adaptation
            final_output: Model output after adaptation
        """
        import torch.optim as optim
        
        # Create optimizer for all params (we'll mask gradients manually)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        losses = []
        
        # Track best weights
        best_loss = float('inf')
        best_weights = None
        best_step = 0
        
        print(f"  [SingleShot] Adapting on single example with {num_steps} steps (lr={lr})...")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(input_data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Zero gradients for frozen weights (only train pruned weights)
            self.zero_frozen_gradients()
            
            # Update weights
            optimizer.step()
            
            # Restore frozen weights
            self.step()
            
            current_loss = loss.item()
            losses.append(current_loss)
            
            # Save best weights
            if current_loss < best_loss:
                best_loss = current_loss
                best_step = step + 1
                # Save only the pruned weights (the ones we're training)
                best_weights = {}
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                        if mask_key in self.inverse_masks:
                            # Save the pruned portion of weights
                            best_weights[name] = (param.data * self.inverse_masks[mask_key]).clone()
            
            if (step + 1) % 5 == 0:
                print(f"    Step {step+1}/{num_steps}: Loss = {current_loss:.6f} (best: {best_loss:.6f} @ step {best_step})")
        
        # Restore best weights
        if best_weights is not None:
            print(f"  [SingleShot] Restoring best weights from step {best_step} (loss: {best_loss:.6f})")
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in best_weights:
                        mask_key = name.replace('.weight', '') if name.endswith('.weight') else name
                        if mask_key in self.frozen_weights:
                            # Combine: frozen weights + best pruned weights
                            param.data.copy_(self.frozen_weights[mask_key] + best_weights[name])
        
        # Get final output
        self.model.eval()
        with torch.no_grad():
            final_output = self.model(input_data)
        
        print(f"  [SingleShot] Done. Initial: {losses[0]:.6f}, Best: {best_loss:.6f} (step {best_step}), Final: {losses[-1]:.6f}")
        
        return losses, final_output


def load_rigl_checkpoint(checkpoint_path, model, device='cuda'):
    """Load RigL model weights and masks from a checkpoint file."""
    print(f"\n{'='*70}")
    print(f"Loading RigL checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    # Allow passing either:
    # - a direct .pt file path
    # - a checkpoint directory (containing best_model.pt/model.pt)
    # - an old-style directory/file prefix that now has a suffix like:
    #   *_train-<dataset>_params-<N>/
    ckpt_path = pathlib.Path(str(checkpoint_path))
    if ckpt_path.is_dir():
        if (ckpt_path / "best_model.pt").exists():
            ckpt_path = ckpt_path / "best_model.pt"
        elif (ckpt_path / "model.pt").exists():
            ckpt_path = ckpt_path / "model.pt"

    if not ckpt_path.exists():
        # If user passed ".../<run>/best_model.pt" but run dir now has a suffix, try glob.
        parent = ckpt_path.parent.parent if ckpt_path.suffix == ".pt" else ckpt_path.parent
        base_dir = ckpt_path.parent.name if ckpt_path.suffix == ".pt" else ckpt_path.name
        file_name = ckpt_path.name if ckpt_path.suffix == ".pt" else "best_model.pt"

        try:
            matches = sorted(parent.glob(base_dir + "_train-*_params-*"))
        except Exception:
            matches = []

        if matches:
            candidate = matches[-1] / file_name
            if candidate.exists():
                ckpt_path = candidate

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    
    # Load model weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"  Loaded model weights")
    else:
        raise ValueError(f"Checkpoint does not contain model weights: {checkpoint_path}")
    
    # Load masks
    if 'rigl_scheduler' not in checkpoint or 'masks' not in checkpoint['rigl_scheduler']:
        raise ValueError(f"Checkpoint does not contain RigL masks: {checkpoint_path}")
    
    masks = checkpoint['rigl_scheduler']['masks']
    print(f"  Loaded {len(masks)} masks")
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_psnr_mean' in checkpoint:
        print(f"  Best PSNR: {checkpoint['best_psnr_mean']:.2f}")
    
    print(f"{'='*70}\n")
    
    return masks


def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
def save_image(source, folder_path, image_name):
    source = source.clone()
    for i in range(source.size(0)):  # Iterate over the batch dimension
        image = source[i]
        image -= image.min()
        max_val = image.max()
        if max_val > 0:
            image /= max_val
        source[i] = image

    if source.dim() == 3:
        source = source.unsqueeze(1)

    grid = torchvision.utils.make_grid(source, nrow=4, pad_value=1)
    numpy_image = grid.permute(1, 2, 0).cpu().detach().numpy()

    os.makedirs(folder_path, exist_ok=True)

    save_path = os.path.join(folder_path, f'{image_name}.png')
    plt.imsave(save_path, numpy_image)
    return


# Example usage
folder_path = "output_images"
image_name = "example_image.png"


# save_image_without_pillow(tensor_image, folder_path, image_name)



class DataTransform:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice):
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        target = normalize(transforms.to_tensor(target))
        mean = std = 0

        if target.shape[1] != self.resolution:
            target = transforms.center_crop(target, (self.resolution, self.resolution))
        return fastmri.rss(image), target, mean, std, attrs['norm'].astype(np.float32)


class ImageNetTransform:
    """
    Transform for ImageNet data to be compatible with MRI reconstruction model.
    Converts RGB images to grayscale with pseudo-complex format (H, W, 2).
    """
    def __init__(self, resolution=320):
        self.resolution = resolution
        self.transform = tv_transforms.Compose([
            tv_transforms.Resize((resolution, resolution)),
            tv_transforms.Grayscale(num_output_channels=1),
            tv_transforms.ToTensor(),  # Converts to (1, H, W) in [0, 1]
        ])
    
    def __call__(self, img):
        # Apply transforms
        img_tensor = self.transform(img)  # (1, H, W)
        
        # Convert to pseudo-complex format (H, W, 2) for model compatibility
        # Real part = image, Imaginary part = zeros
        img_gray = img_tensor.squeeze(0)  # (H, W)
        img_complex = torch.stack([img_gray, torch.zeros_like(img_gray)], dim=-1)  # (H, W, 2)
        
        # Target should be (H, W) to match model output shape after batching
        # MRI target is (H, W), model outputs (B, H, W), so ImageNet target must be (H, W) too
        target = normalize(img_gray)  # (H, W) - squeezed to match MRI format
        
        mean = std = 0
        norm = np.float32(1.0)
        
        return img_complex, target, mean, std, norm


class ImageNetDataset(torch.utils.data.Dataset):
    """
    ImageNet dataset wrapper that returns data in MRI reconstruction format.
    """
    def __init__(self, root, split='train', resolution=320):
        self.resolution = resolution
        self.transform = ImageNetTransform(resolution)
        
        # Use torchvision ImageFolder
        self.dataset = datasets.ImageFolder(root=root)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img)


def create_datasets(args):
    dataset_type = getattr(args, 'dataset', 'mri')
    
    if dataset_type == 'imagenet':
        # ImageNet dataset
        imagenet_path = getattr(args, 'imagenet_path', pathlib.Path('../ImageNet'))
        print(f"Using ImageNet dataset from: {imagenet_path}", flush=True)
        print(f"  Train: {imagenet_path / 'train'}", flush=True)
        print(f"  Val: {imagenet_path / 'val'}", flush=True)
        
        train_data = ImageNetDataset(
            root=str(imagenet_path / 'train'),
            split='train',
            resolution=args.resolution
        )
        
        dev_data = ImageNetDataset(
            root=str(imagenet_path / 'val'),
            split='val',
            resolution=args.resolution
        )
        
        # Apply sample rate if specified
        if args.sample_rate < 1.0:
            num_train = int(len(train_data) * args.sample_rate)
            indices = list(range(num_train))
            train_data = torch.utils.data.Subset(train_data, indices)
        
        return dev_data, train_data
    
    else:
        # MRI dataset (default)
        print(args.data_path / f'multicoil_train', flush=True)
        print(args.data_path / f'multicoil_val', flush=True)
        train_data = SliceData(
            root=args.data_path / f'multicoil_train',
            transform=DataTransform(args.resolution),
            sample_rate=args.sample_rate)

        dev_data = SliceData(
            root=args.data_path / f'multicoil_val',
            transform=DataTransform(args.resolution),
            sample_rate=1)

        return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args) 
    display_data = [dev_data[i] for i in range(0, len(dev_data), 1 if len(dev_data) // 16 == 0 else len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=20,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=args.batch_size,
        num_workers=20,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader

from torch.optim import SGD
import torch
import torch.nn.functional as F


def pgd_attack_on_trajectory(model, input_tensor, target_tensor, epsilon, alpha=1, steps=1, norm='linf'):
    """
    PGD attack directly on the trajectory perturbation (not on the trajectory itself).
    """
    from pytorch_nufft.nufft2 import nufft, nufft_adjoint
    device = input_tensor.device
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    # Clone original trajectory
    original_trajectory = model.module.subsampling.x.detach().clone()

    # Initialize perturbation as the variable we optimize
    perturbation = torch.zeros_like(original_trajectory, device=device, requires_grad=True)

    lowest_psnr = float('inf')
    best_perturbation = torch.zeros_like(perturbation)

    for step in range(steps + 1):
        # Apply current perturbation to trajectory
        perturbed_trajectory = torch.clamp(original_trajectory + perturbation, -160, 160)

        # Inject into model
        x_full = perturbed_trajectory.reshape(-1, 2)

        input = input_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)
        sub_ksp = nufft(input, x_full)
        output = nufft_adjoint(sub_ksp, x_full, input.shape)
        output = output.permute(0, 1, 3, 4, 2)
        output = transforms.complex_abs(output)
        output = normalize(output)

        # Forward pass
        output = model.module.reconstruction_model(output)
        target = target_tensor.view_as(output) if output.shape != target_tensor.shape else target_tensor
        loss = F.l1_loss(output, target)

        # Track lowest PSNR
        current_psnr = psnr(target.detach().cpu().numpy(), output.detach().cpu().numpy())
        if current_psnr < lowest_psnr:
            lowest_psnr = current_psnr
            best_perturbation = perturbation.detach().clone()

        if step == steps:
            break

        # Backward: compute gradient w.r.t. perturbation
        if perturbation.grad is not None:
            perturbation.grad.zero_()
        loss.backward()

        # Update perturbation using its own gradient
        grad = perturbation.grad

        if norm == 'linf':
            perturbation.data += alpha * grad.sign()
            perturbation.data = project_linf(perturbation.data, epsilon)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    # Restore original trajectory
    model.module.subsampling.attack_trajectory_radial = best_perturbation


# Projection functions (same as provided)
def project_linf(perturbation, epsilon):
    return torch.clamp(perturbation, -epsilon, epsilon)


def project_l2(perturbation, epsilon):
    flat = perturbation.view(perturbation.shape[0], -1)
    norm = flat.norm(p=2, dim=1, keepdim=True)
    factor = torch.min(torch.ones_like(norm), epsilon / (norm + 1e-10))
    projected = flat * factor
    return projected.view_as(perturbation)


def project_l1(perturbation, epsilon):
    original_shape = perturbation.shape
    x_flat = perturbation.view(perturbation.shape[0], -1)
    abs_x = torch.abs(x_flat)

    sorted_x, _ = torch.sort(abs_x, descending=True, dim=1)
    cumsum = torch.cumsum(sorted_x, dim=1)

    rho = (sorted_x * torch.arange(1, x_flat.shape[1] + 1, device=x_flat.device)) > (cumsum - epsilon)
    rho_idx = rho.sum(dim=1) - 1
    theta = (cumsum.gather(1, rho_idx.unsqueeze(1)) - epsilon) / (rho_idx + 1).float().unsqueeze(1)
    theta = torch.clamp(theta, min=0)

    projected = torch.sign(x_flat) * torch.clamp(abs_x - theta, min=0)
    return projected.view(original_shape)


def project_linf(perturbation, epsilon):
    return torch.clamp(perturbation, -epsilon, epsilon)


def project_l2(perturbation, epsilon):
    flat = perturbation.view(perturbation.shape[0], -1)
    norm = flat.norm(p=2, dim=1, keepdim=True)
    factor = torch.min(torch.ones_like(norm), epsilon / (norm + 1e-10))
    projected = flat * factor
    return projected.view_as(perturbation)


def project_l1(perturbation, epsilon):
    original_shape = perturbation.shape
    x_flat = perturbation.view(perturbation.shape[0], -1)
    abs_x = torch.abs(x_flat)

    sorted_x, _ = torch.sort(abs_x, descending=True, dim=1)
    cumsum = torch.cumsum(sorted_x, dim=1)

    rho = (sorted_x * torch.arange(1, x_flat.shape[1] + 1, device=x_flat.device)) > (cumsum - epsilon)
    rho_idx = rho.sum(dim=1) - 1
    theta = (cumsum.gather(1, rho_idx.unsqueeze(1)) - epsilon) / (rho_idx + 1).float().unsqueeze(1)
    theta = torch.clamp(theta, min=0)

    projected = torch.sign(x_flat) * torch.clamp(abs_x - theta, min=0)
    return projected.view(original_shape)

def select_top_perturbations_balanced(model, image, target, mask_module, loss_fn, num_bits=3):
    """
    Perform one gradient ascent step to find the most harmful perturbations in XOR-space.

    Args:
        model: Reconstruction model (e.g., autoencoder).
        image: Input image tensor.
        mask_module: BinaryMask instance with .binary_mask.
        loss_fn: Loss function (e.g., MSELoss).
        lr: Learning rate for ascent.
        num_bits: Number of positions to flip in perturbation mask.

    Returns:
        perturbation_mask: Float tensor with 1.0 at selected perturbation locations (rest 0.0).
    """
    # Create perturbation mask initialized to 0 (no flips)
    perturbation_mask = torch.zeros_like(mask_module, dtype=torch.float32, requires_grad=True)
    # Get the "new" mask: XOR between current binary mask and perturbation mask
    def xor_masks(m1, m2):
        return (1 - m1.float()) * m2.float() + m1.float() * (1 - m2.float())  # Differentiable XOR

    new_mask = xor_masks(mask_module.detach(), perturbation_mask).view(1, 1, 320, 1, 1)

    # Apply the new mask to the image
    def apply_mask(mask, x):
        input_c = fastmri.fft2c(x)
        print(input_c.shape, mask.shape)
        input_c_masked = input_c.unsqueeze(0) * mask
        input_c_masked = fastmri.ifft2c(input_c_masked)
        input_c_masked = transforms.complex_abs(input_c_masked)
        min_val = input_c_masked.amin(dim=(1, 2, 3), keepdim=True)
        max_val = input_c_masked.amax(dim=(1, 2, 3), keepdim=True)
        input_c_masked = (input_c_masked - min_val) / (max_val - min_val + 1e-8)  # Avoid divide-by-zero
        output = model(input_c_masked)
        return output

    masked_recon_image = apply_mask(new_mask, image)

    # Compute reconstruction loss
    loss = loss_fn(masked_recon_image, target)
    loss.backward()

    # Get gradient of perturbation_mask
    grad = perturbation_mask.grad.detach().view(-1)
    binary_mask_flat = mask_module.view(-1)

    # Find top num_bits from where original mask == 1
    ones_mask = (binary_mask_flat == 1.0)
    ones_grad = grad.clone()
    ones_grad[~ones_mask] = float('-inf')  # exclude non-1s
    top_vals, top_indices = torch.topk(ones_grad, int(num_bits))

    # Find bottom num_bits from where original mask == 0
    zeros_mask = (binary_mask_flat == 0.0)
    zeros_grad = grad.clone()
    zeros_grad[~zeros_mask] = float('inf')  # exclude non-0s
    bottom_vals, bottom_indices = torch.topk(-zeros_grad, int(num_bits))  # negate for lowest

    # Merge indices
    all_indices = torch.cat([top_indices, bottom_indices])

    # Create final perturbation mask
    final_mask = torch.zeros_like(perturbation_mask).flatten()
    final_mask[all_indices] = 1.0
    return final_mask.view_as(perturbation_mask)


def train_epoch(args, epoch, model, data_loader, optimizer, optimizer_sub, writer, scheduler, scheduler_sub, adv_mask = None, rigl_scheduler = None, frozen_mask_trainer = None, stamp_scheduler = None):
    model.train()
    avg_loss = 0.

    import re
    mode = args.inter_gap_mode  # e.g., "changing_downwards_20"
    match = re.search(r'changing_downwards_(\d+)', mode)
    if match:
        end_epoch = int(match.group(1))
        print(f'end_epoch = {end_epoch}')
    else:
        raise ValueError(f"Invalid inter_gap_mode format: {mode}")

    if "changing_downwards" in args.inter_gap_mode:
        # Define start and end values
        start_epoch = 0
        start_gap = args.num_epochs
        end_gap = 1

        if end_epoch == 0:
            model.module.subsampling.interp_gap = end_gap

        elif epoch <= end_epoch:
            # Linear interpolation
            interp_gap = start_gap + (end_gap - start_gap) * (epoch - start_epoch) / (end_epoch - start_epoch)
            model.module.subsampling.interp_gap = max(int(interp_gap), end_gap)
        else:
            model.module.subsampling.interp_gap = end_gap


    print("\n", "epochs: ", epoch ," model.module.subsampling.interp_gap: ", model.module.subsampling.interp_gap)

    psnr_l = []
    ssim_l = []
    start_epoch = start_iter = time.perf_counter()
    print(f'a_max={args.a_max}, v_max={args.v_max}')

    done = False
    for iter, data in enumerate(data_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        optimizer_sub.zero_grad()
        
        # RigL: Apply masks to zero out inactive weights before forward pass
        if rigl_scheduler is not None:
            rigl_scheduler.apply_masks()
            # Print sparsity every 100 iterations (matching RigL update frequency)
            if iter % 100 == 0:
                total_params = 0
                zero_params = 0
                for param in model.module.reconstruction_model.parameters():
                    total_params += param.numel()
                    zero_params += (param.data == 0).sum().item()
                print(f"  [RigL] Model sparsity: {zero_params:,}/{total_params:,} zeros ({100*zero_params/total_params:.1f}% sparse)")
        
        # Frozen mask training: no action needed before forward pass
        # (frozen weights are preserved, trainable weights can be any value)
            
        input, target, mean, std, norm = data
        if input is None:
            print("skipping")
            continue

        noise = torch.zeros_like(input)
        if ("image" in args.noise_behaviour) and random.random() <= (args.noise_p):
            print("applied image! ", model.module.noise_model.get_noise())
            noise = torch.randn_like(input) * model.module.noise_model.get_noise()

        input = input + noise
        input = input.to(args.device)
        target = target.to(args.device)

        if ('pgd' in args.noise_behaviour) and ('radial' in args.initialization) and (random.random() <= (args.noise_p)):
            pgd_attack_on_trajectory(model, input, target, model.module.noise_model.get_noise())
        elif ('pgd' in args.noise_behaviour) and ('radial' in args.initialization):
            model.module.subsampling.attack_trajectory_radial = None

        if ('pgd' in args.noise_behaviour) and ('cartesian' in args.initialization) and (random.random() <= (args.noise_p)):
            model.module.subsampling.attack_trajectory_cartesian = select_top_perturbations_balanced(model.module.reconstruction_model, input, target, model.module.subsampling.get_mask(), F.l1_loss, num_bits=model.module.noise_model.get_noise()).float()
        elif ('pgd' in args.noise_behaviour) and ('cartesian' in args.initialization):
            model.module.subsampling.attack_trajectory_cartesian = None


        output = model(input.unsqueeze(1))
        
        # STAMP: Collect subsampled images for importance computation
        # The subsampling layer converts k-space to image domain
        if stamp_scheduler is not None and iter % 50 == 0:  # Collect every 50th batch
            try:
                # Get the subsampled image (after k-space to image)
                with torch.no_grad():
                    subsampled_input = model.module.subsampling(input.unsqueeze(1))
                    if subsampled_input.dim() == 3:
                        subsampled_input = subsampled_input.unsqueeze(1)
                    stamp_scheduler.collect_batch(subsampled_input, target.view_as(subsampled_input))
            except Exception as e:
                pass  # Silently skip if collection fails
        
        x = model.module.get_trajectory()
        # v, a = get_vel_acc(x)
        # acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max).abs() + 1e-8, 2)))
        # vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max).abs() + 1e-8, 2)))
        resolution = target.shape[-1]

        if epoch <= 10:
            vel_weight = 1e-3
        elif epoch <= 20:
            vel_weight = 1e-2
        elif epoch <= 30:
            vel_weight = 1e-1

        if epoch <= 10:
            acc_weight = 1e-3
        elif epoch <= 20:
            acc_weight = 1e-2
        elif epoch <= 30:
            acc_weight = 1e-1

        if not done:
            done = True
            # Global images directory: Images/<model>/<run>/<epoch_###>/
            # (keeps all images under one root, then per-model)
            save_dir = os.path.join("Images", str(args.model), str(args.test_name), f"epoch_{epoch:03d}")
            os.makedirs(save_dir, exist_ok=True)
            # NOTE: save_image expects (folder_path, image_name) — previously we passed a filename
            # which created nested ".../0.png/0.png" directories.
            save_image(
                torch.stack(
                    [
                        output[0].view(resolution, resolution),
                        target[0].view(resolution, resolution),
                    ]
                ),
                save_dir,
                f"{iter}",
            )

        data_min = target.min()
        data_max = target.max()
        target_normalized = (target - data_min) / (data_max - data_min)
        output_normalized = (output - data_min) / (data_max - data_min)

        loss_l1 = F.l1_loss(output, target)
        psnr_l.append(psnr(target.detach().cpu().numpy(), output.detach().cpu().numpy()))
        ssim_l.append(ssim(target.detach().cpu().numpy(), output.detach().cpu().numpy()))
        rec_loss = loss_l1 #+ dcLoss # SSIMLoss().to(args.device)(output, target, data_range) # F.l1_loss(output, target)
        if args.TSP and epoch < args.TSP_epoch:
            loss = args.rec_weight * rec_loss
        else:
            loss = args.rec_weight * rec_loss #+ args.vel_weight * vel_loss + args.acc_weight * acc_loss

        #if vel_loss + acc_loss > 1e-3:
        #    optimize_trajectory(args, model)

        #print("before backprop:", model.module.subsampling.x)
        loss.backward()
        #print("after backprop:", model.module.subsampling.x)
        
        # RigL: update sparse masks based on gradients
        if rigl_scheduler is not None:
            rigl_scheduler.step()
        
        # Frozen mask training: zero out gradients for frozen (active) weights
        if frozen_mask_trainer is not None:
            frozen_mask_trainer.zero_frozen_gradients()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1.)
        optimizer.step()
        optimizer_sub.step()
        
        # Frozen mask training: re-apply masks after optimizer step to ensure zeros stay zero
        if frozen_mask_trainer is not None:
            frozen_mask_trainer.step()
            if iter % 500 == 0:
                frozen_mask_trainer.print_sparsity()
        
        if args.initialization == 'cartesian':
            model.module.subsampling.apply_binary_grad(optimizer_sub.param_groups[0]['lr'])
        model.module.subsampling.attack_trajectory = None
        
        # Dynamic U-Net: perform channel swap if it's time
        if args.model == 'DynamicUnet':
            try:
                model.module.reconstruction_model.maybe_swap_channels()
            except Exception as e:
                logging.warning(f"Failed to perform channel swap at iteration {iter}: {e}")

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        # writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'PSNR: {np.mean(psnr_l):.2f} +- {np.std(psnr_l):.2f}, SSIM: {np.mean(ssim_l):.4f} +- {np.std(ssim_l):.4f}'
            )
        start_iter = time.perf_counter()
    if scheduler:
        scheduler.step()

    if scheduler_sub is not None:
        try:
            scheduler_sub.step()
        except ValueError as e:
            print(f"Skipping scheduler step: {e}")

    model.module.noise_model.step()
    print("noise level = ", model.module.noise_model.get_noise())
    print(optimizer.param_groups[0]['lr'], (optimizer_sub.param_groups[0]['lr']))
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, adv_mask = None):
    model.eval()
    losses = []
    psnr_l = []
    ssim_l = []

    print("Halo", len(data_loader))
    start = time.perf_counter()
    with torch.no_grad():
        if epoch > 0:
            for iter, data in enumerate(data_loader):
                input, target, mean, std, norm = data
                input = input.to(args.device)
                resolution = target.shape[-1]
                target = target.to(args.device)
                output = model(input.unsqueeze(1))
                recons = output.to('cpu').squeeze(1).view(target.shape)
                recons = recons.squeeze()
                if output.shape != target.shape:
                    target = target.view_as(output)
                loss = F.l1_loss(output, target)
                losses.append(loss.item())
                target = target.view(-1,resolution,resolution)
                recons = recons.view(target.shape)
                psnr_l.append(psnr(target.detach().cpu().numpy(), recons.detach().cpu().numpy()))
                ssim_l.append(ssim(target.detach().cpu().numpy(), recons.detach().cpu().numpy()))

        print(
            f'PSNR: {np.mean(psnr_l):.2f} +- {np.std(psnr_l):.2f}, SSIM: {np.mean(ssim_l):.4f} +- {np.std(ssim_l):.4f}')
        x = model.module.get_trajectory()
        # v, a = get_vel_acc(x)
        # acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max), 2)))
        # vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max), 2)))
        rec_loss = np.mean(losses)

        writer.add_scalar('Rec_Loss', rec_loss, epoch)
        # writer.add_scalar('Acc_Loss', acc_loss.detach().cpu().numpy(), epoch)
        # writer.add_scalar('Vel_Loss', vel_loss.detach().cpu().numpy(), epoch)
        #writer.add_scalar('Total_Loss',
        #                  rec_loss + acc_loss.detach().cpu().numpy() + vel_loss.detach().cpu().numpy(), epoch)

        psnr_mean, psnr_std = np.mean(psnr_l), np.std(psnr_l)
        ssim_mean, ssim_std = np.mean(ssim_l), np.std(ssim_l)
        x = model.module.get_trajectory()
        #v, a = get_vel_acc(x)
        if args.TSP and epoch < args.TSP_epoch:
            writer.add_figure('Scatter', plot_scatter(x.detach().cpu().numpy()), epoch)
        else:
            trajectory = plot_trajectory(x.detach().cpu().numpy())
            ax = trajectory.gca()  # Get the current axis from the figure
            text_str = f'PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}\nSSIM: {ssim_mean:.4f} ± {ssim_std:.4f}'
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            save_dir = f"trajectory/{args.exp_dir}_{args.model}"
            os.makedirs(save_dir, exist_ok=True)
            trajectory.savefig(f"{save_dir}/trajectory_epoch_{epoch}.png")
            plt.close(trajectory)
            writer.add_figure('Trajectory', plot_trajectory(x.detach().cpu().numpy()), epoch)
            writer.add_figure('Scatter', plot_scatter(x.detach().cpu().numpy()), epoch)

            if adv_mask:
                trajectory = plot_trajectory(adv_mask.get_trajectory().detach().cpu().numpy())
                ax = trajectory.gca()  # Get the current axis from the figure
                text_str = f'PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}\nSSIM: {ssim_mean:.4f} ± {ssim_std:.4f}'
                ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                save_dir = f"trajectory/{args.exp_dir}_{args.model}"
                os.makedirs(save_dir, exist_ok=True)
                trajectory.savefig(f"{save_dir}/trajectory_adv_epoch_{epoch}.png")
                plt.close(trajectory)

        # writer.add_figure('Accelerations_plot', plot_acc(a.cpu().numpy(), args.a_max), epoch)
        # writer.add_figure('Velocity_plot', plot_acc(v.cpu().numpy(), args.v_max), epoch)
        #writer.add_text('Coordinates', str(x.detach().cpu().numpy()).replace(' ', ','), epoch)
    if epoch == 0:
        return None, time.perf_counter() - start
    else:
        return np.mean(losses), time.perf_counter() - start, psnr_mean, ssim_mean


def plot_scatter(x):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1], '.')
    return fig


def plot_trajectory(x):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1])
    return fig


def plot_acc(a, a_max=None):
    fig, ax = plt.subplots(2, sharex=True)
    for i in range(a.shape[0]):
        ax[0].plot(a[i, :, 0])
        ax[1].plot(a[i, :, 1])
    if a_max is not None:
        limit = np.ones(a.shape[1]) * a_max
        ax[1].plot(limit, color='red')
        ax[1].plot(-limit, color='red')
        ax[0].plot(limit, color='red')
        ax[0].plot(-limit, color='red')
    return fig


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        print("entered visualize: ")
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.to(args.device)
            target = target.unsqueeze(1).to(args.device)

            save_image(target, 'Target')
            if epoch != 0:
                print(input.unsqueeze(1).shape)
                output = model(input.unsqueeze(1))
                save_image(output, 'Reconstruction')
                save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, scheduler, best_dev_loss, best_psnr_mean, best_ssim_mean, is_new_best, metrics, rigl_scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_dev_loss': best_dev_loss,
        'best_psnr_mean': best_psnr_mean,
        'best_ssim_mean': best_ssim_mean,
        'exp_dir': exp_dir,
        'metrics': metrics
    }
    
    # Save RigL scheduler state (masks) if present
    if rigl_scheduler is not None:
        checkpoint['rigl_scheduler'] = rigl_scheduler.state_dict()
        # Log sparsity info
        total_masked = 0
        total_params = 0
        for name, mask in rigl_scheduler.masks.items():
            total_masked += (mask == 0).sum().item()
            total_params += mask.numel()
        print(f"  [RigL] Saving model with masks: {total_masked:,}/{total_params:,} masked ({100*total_masked/total_params:.1f}% sparse)")
    
    torch.save(checkpoint, f=exp_dir + '/model.pt')
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')


def calculate_model_flops(model, input_shape=(1, 1, 320, 320, 2), device='cuda'):
    """
    Calculate FLOPs for the model using available libraries.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W, 2) for complex images
        device: Device to run on
    
    Returns:
        tuple: (flops, macs, method_used)
    """
    # Try thop first (most compatible)
    try:
        from thop import profile
        model.eval()
        inputs = torch.randn(input_shape).to(device)
        with torch.no_grad():
            macs, params = profile(model, inputs=(inputs,), verbose=False)
        model.train()
        flops = macs * 2  # FLOPs ≈ 2 × MACs
        return flops, macs, 'thop'
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        inputs = torch.randn(input_shape).to(device)
        with torch.no_grad():
            flops_counter = FlopCountAnalysis(model, inputs)
            flops = flops_counter.total()
        model.train()
        macs = flops / 2
        return flops, macs, 'fvcore'
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try ptflops
    try:
        from ptflops import get_model_complexity_info
        # ptflops doesn't handle complex 5D tensors well, skip
        pass
    except ImportError:
        pass
    
    return None, None, None


def format_flops(num):
    """Format FLOPs in human-readable format."""
    if num is None:
        return "N/A"
    if num >= 1e12:
        return f"{num/1e12:.2f} TFLOPs"
    elif num >= 1e9:
        return f"{num/1e9:.2f} GFLOPs"
    elif num >= 1e6:
        return f"{num/1e6:.2f} MFLOPs"
    elif num >= 1e3:
        return f"{num/1e3:.2f} KFLOPs"
    else:
        return f"{num:.2f} FLOPs"


def build_model(args):
    print("\n" + "=" * 80)
    print(f"Building model: {args.model}")
    print("=" * 80)

    model = Subsampling_Model(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        res=args.resolution,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization,
        SNR=args.SNR,
        n_shots=args.n_shots,
        interp_gap=args.interp_gap,
        type=args.model,
        img_size=args.img_size,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        sample_per_shot=args.sample_per_shot,
        epsilon = args.epsilon,
        noise_p = 0 if ("noise" not in args.noise_behaviour) else args.noise_p,
        std = args.std,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        noise = args.noise_behaviour,
        epochs = args.num_epochs,
        # Dynamic U-Net parameters
        swap_frequency=args.swap_frequency if hasattr(args, 'swap_frequency') else 10,
        # CondUnet parameters
        num_experts=args.num_experts if hasattr(args, 'num_experts') else 8,
        # FDUnet parameters
        fd_kernel_num=args.fd_kernel_num if hasattr(args, 'fd_kernel_num') else 64,
        fd_use_simple=args.fd_use_simple if hasattr(args, 'fd_use_simple') else False,
        # HybridSnakeFDUnet parameters
        snake_layers=args.snake_layers if hasattr(args, 'snake_layers') else 2,
        snake_kernel_size=args.snake_kernel_size if hasattr(args, 'snake_kernel_size') else 9,
        # ConfigurableUNet parameters (LightUNet, LightDCN, LightFD, LightDCNFD)
        use_dcn=args.use_dcn if hasattr(args, 'use_dcn') else False,
        use_fdconv=args.use_fdconv if hasattr(args, 'use_fdconv') else False,
    ).to(args.device)
    
    # Count and display parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    recon_params = sum(p.numel() for p in model.reconstruction_model.parameters() if p.requires_grad)
    subsampling_params = total_params - recon_params
    
    print(f"\n📊 Parameter Count:")
    print(f"  Reconstruction Model: {recon_params:,} parameters ({recon_params/1e6:.2f}M)")
    print(f"  Subsampling Model:    {subsampling_params:,} parameters ({subsampling_params/1e6:.2f}M)")
    print(f"  Total:                {total_params:,} parameters ({total_params/1e6:.2f}M)")
    print(f"  Model size:           {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Calculate and display FLOPs
    print(f"\n⚡ Computational Complexity:")
    input_shape = (1, 1, 320, 320, 2)  # Standard input shape for MRI
    flops, macs, method = calculate_model_flops(model, input_shape, args.device)
    
    if flops is not None:
        print(f"  FLOPs:  {format_flops(flops)} ({flops:,})")
        print(f"  MACs:   {format_flops(macs)} ({macs:,})")
        print(f"  Method: {method}")
        print(f"  Input:  {input_shape[0]}×{input_shape[1]}×{input_shape[2]}×{input_shape[3]}×{input_shape[4]}")
    else:
        print(f"  FLOPs calculation unavailable (install 'thop' or 'fvcore')")
        print(f"  To enable: pip install thop")
    
    print("=" * 80 + "\n")
    
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer, args


def build_optim(args, model):
    optimizer_sub = torch.optim.Adam([{'params': model.module.subsampling.parameters(), 'lr': args.sub_lr}])
    optimizer = torch.optim.Adam([{'params': model.module.reconstruction_model.parameters()}], args.lr)
    return optimizer, optimizer_sub

def build_scheduler(optimizer, optimizer_sub, args):
    # For small number of epochs (e.g., ImageNet experiments), use StepLR to avoid OneCycleLR issues
    # OneCycleLR requires total_steps > pct_start * total_steps for warmup phase
    if args.num_epochs < 20:
        # Use StepLR for short training runs
        scheduler_sub = torch.optim.lr_scheduler.StepLR(
            optimizer_sub,
            step_size=max(1, args.num_epochs // 3),
            gamma=0.1
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.num_epochs // 3),
            gamma=0.1
        )
    else:
        # Use OneCycleLR for longer training runs
        total_steps = args.num_epochs
        scheduler_sub = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_sub,
            max_lr=[args.sub_lr],  # One per param group
            total_steps=(total_steps) if "vit-l-pretrained-radial" in args.model else total_steps,
            pct_start=0.1,  # = 0.1 for warmup
            anneal_strategy='linear',       # linear decay after warmup
            cycle_momentum=False,           # disable momentum scheduling
            div_factor=10,                 # base_lr = max_lr / div_factor (i.e., start at 0)
            final_div_factor=1e9            # decay linearly to ~0
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[args.lr],  # One per param group
            total_steps=total_steps,
            pct_start=0.1,  # = 0.1 for warmup
            anneal_strategy='linear',       # linear decay after warmup
            cycle_momentum=False,           # disable momentum scheduling
            div_factor=10,                 # base_lr = max_lr / div_factor (i.e., start at 0)
            final_div_factor=1e9            # decay linearly to ~0
        )
    return scheduler, scheduler_sub


def eval(args, model, data_loader):
    model.eval()
    psnr_l = []
    ssim_l = []
    with torch.no_grad():
        for (input, target, mean, std, norm) in data_loader:
            input = input.to(args.device)
            recons = model(input.unsqueeze(1)).to('cpu').squeeze(1)
            # recons = transforms.complex_abs(recons)  # complex to real
            recons = recons.squeeze()
            target = target.to('cpu')

            psnr_l.append(psnr(target.numpy(), recons.numpy()))
            ssim_l.append(ssim(target.numpy(), recons.numpy()))

    print(f'PSNR: {np.mean(psnr_l):.2f} +- {np.std(psnr_l):.2f}, SSIM: {np.mean(ssim_l):.4f} +- {np.std(ssim_l):.4f}')
    return

def train():
    import torch
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")  # 0 is the index of the first GPU
    else:
        print("CUDA device not available")
    torch.cuda.empty_cache()
    print("started training", flush=True)
    args = create_arg_parser().parse_args()

    args.v_max = args.gamma * args.G_max * args.FOV * args.dt

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    optimizer = None
    g_params = None
    g_optimizer = None
    d_optimizer = None
    d_scheduler = None
    g_scheduler = None

    rigl_state = None  # Will be loaded from checkpoint if resuming with RigL
    if args.resume:
        # Resume uses the already-established run directory name (do not rename).
        args.exp_dir = f'summary/{args.test_name}'
        pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=args.exp_dir)
        with open(args.exp_dir + '/args.txt', "w") as text_file:
            print(vars(args), file=text_file)
        print(args.test_name, flush=True)
        args.checkpoint = f'summary/{args.test_name}/model.pt'

        checkpoint, model, optimizer = load_model(args.checkpoint)
        # args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        best_psnr_mean = checkpoint['best_psnr_mean']
        best_ssim_mean = checkpoint['best_ssim_mean']
        start_epoch = checkpoint['epoch'] + 1
        rigl_state = checkpoint.get('rigl_scheduler', None)  # Save for later
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        if "pretrain" in args.model:
            path = ""
            if "cartesian" in args.model and "vit-l" in args.model:
                path = "pretrained_models/equidist"
            elif "radial" in args.model and "vit-l" in args.model:
                path = "pretrained_models/radial"
            checkpoint = torch.load(path)
            model.module.reconstruction_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer, optimizer_sub = build_optim(args, model)
        scheduler, scheduler_sub = build_scheduler(optimizer, optimizer_sub, args)

        # Finalize run naming AFTER the model is built so we can embed parameter count.
        train_ds_name = _get_train_dataset_name(args)
        param_count = _count_parameters(model)
        args.num_parameters = param_count
        args.train_dataset_name = train_ds_name

        suffix = f"_train-{train_ds_name}_params-{param_count}"
        if suffix not in str(args.test_name):
            args.test_name = f"{args.test_name}{suffix}"

        args.exp_dir = f"summary/{args.test_name}"
        pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=args.exp_dir)
        with open(args.exp_dir + '/args.txt', "w") as text_file:
            print(vars(args), file=text_file)
        print(args.test_name, flush=True)
        args.checkpoint = f'summary/{args.test_name}/model.pt'

        best_dev_loss = 1e9
        best_psnr_mean = 0
        best_ssim_mean = 0
        start_epoch = 0
    inter_gap_mode = args.inter_gap_mode
    noise_behaviour = args.noise_behaviour
    logging.info(args)
    adv_mask = None
    if "cartesian" and "pgd" in args.noise_behaviour:
        adv_mask = SubsamplingBinary(320, 100, 0, adv=True).to("cuda")
    train_loader, dev_loader, display_loader = create_data_loaders(args)
    
    # RigL sparse training scheduler (created after data loaders to know total iterations)
    rigl_scheduler = None
    if args.use_rigl:
        from models.rec_models.layers.rigl_sparse import RigLScheduler
        iters_per_epoch = len(train_loader)
        total_iters = args.num_epochs * iters_per_epoch
        # Update masks twice per epoch (restructure at mid-epoch and end-of-epoch)
        rigl_update_freq = iters_per_epoch // 2
        rigl_scheduler = RigLScheduler(
            model=model,
            sparsity=args.rigl_sparsity,
            update_freq=rigl_update_freq,
            T_end=int(total_iters * 0.75),  # Stop mask updates at 75% of training
            delta=args.rigl_delta,
            exclude_layers=[],  # Don't skip any layers - apply RigL to all
            initial_masks_path=args.rigl_initial_mask,
            static_mask=args.rigl_static_mask,
        )
        print(f"\n*** RigL Enabled: {args.rigl_sparsity*100:.0f}% sparsity, update every {rigl_update_freq} iters (2x per epoch) ***")
        print(f"*** Iters per epoch: {iters_per_epoch}, Total iterations: {total_iters}, T_end: {int(total_iters * 0.75)} ***\n")
        
        # Restore RigL state when resuming
        if args.resume:
            if rigl_state is not None:
                # Full restore from saved state
                rigl_scheduler.load_state_dict(rigl_state)
                print(f"*** RigL: Restored from checkpoint (iter {rigl_scheduler.iteration}) ***\n")
            else:
                # Reconstruct masks from weights (backward compatible)
                start_iter = start_epoch * len(train_loader)
                rigl_scheduler.reconstruct_from_weights(start_iter)
                print(f"*** RigL: Reconstructed masks from weights (iter {start_iter}) ***\n")
    
    # Frozen mask training (train pruned weights, freeze learned weights)
    frozen_mask_trainer = None
    if args.freeze_masked or getattr(args, 'freeze_masked_decoder_only', False):
        if args.rigl_checkpoint is None:
            raise ValueError("--freeze-masked or --freeze-masked-decoder-only requires --rigl-checkpoint")
        
        # Load both model weights AND masks from RigL checkpoint
        masks = load_rigl_checkpoint(args.rigl_checkpoint, model, device='cuda')
        
        if getattr(args, 'freeze_masked_decoder_only', False):
            # Decoder-only: only train decoder's pruned weights
            frozen_mask_trainer = DecoderOnlyMaskTrainer(model, masks, device='cuda')
        else:
            # Full: train all pruned weights
            frozen_mask_trainer = FrozenMaskTrainer(model, masks, device='cuda')
    
    # STAMP scheduler for channel-wise structured pruning
    # Based on: https://github.com/nkdinsdale/STAMP.git
    stamp_scheduler = None
    if hasattr(args, 'use_stamp') and args.use_stamp:
        from models.rec_models.models.stamp_unet import STAMPScheduler
        prune_epochs = args.stamp_prune_epochs if args.stamp_prune_epochs else None  # None = auto
        stamp_mode = getattr(args, 'stamp_mode', 'Taylor')  # Default to Taylor if not specified
        stamp_scheduler = STAMPScheduler(
            model=model.module.reconstruction_model,
            b_drop=args.stamp_channel_drop_rate,
            prune_epochs=prune_epochs,
            prune_ratio=args.stamp_prune_ratio,
            recovery_epochs=args.stamp_recovery_epochs,
            total_epochs=args.num_epochs,
            mode=stamp_mode
        )
        print(f"\n*** STAMP Enabled (https://github.com/nkdinsdale/STAMP.git) ***")
        print(f"*** Mode: {stamp_mode} | b_drop: {args.stamp_channel_drop_rate*100:.0f}% | Keep ratio: {args.stamp_prune_ratio*100:.0f}% ***")
        print(f"*** Recovery epochs: {args.stamp_recovery_epochs} | Total epochs: {args.num_epochs} ***\n")
    
    dev_loss, dev_time = evaluate(args, 0, model, dev_loader, writer)
    print("started mid point", flush=True)
    best_epoch = 0
    metrics = []
    for epoch in range(start_epoch, args.num_epochs):
        if noise_behaviour == "log":
            model.module.subsampling.epsilon = np.logspace(np.log10(args.epsilon), np.log10(args.end_epsilon), num=args.num_epochs)[epoch]
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, optimizer_sub, writer, scheduler, scheduler_sub, adv_mask, rigl_scheduler, frozen_mask_trainer, stamp_scheduler)
        
        # STAMP: Update dropout probabilities based on filter importance (uses collected images)
        if stamp_scheduler is not None:
            # Uses images collected during train_epoch (post-subsampling images)
            current_drop_rate = stamp_scheduler.step(
                epoch, 
                criterion=nn.L1Loss()
            )
            if epoch % 10 == 0:
                print(f"  [STAMP] Epoch {epoch}: b_drop = {current_drop_rate*100:.1f}%")
            
            # Check if physical pruning occurred - need to recreate optimizer
            if stamp_scheduler.check_and_reset_architecture_changed():
                print(f"  [STAMP] Architecture changed! Recreating optimizer for pruned model...")
                # Update the model in the wrapper if needed
                if hasattr(model.module, 'reconstruction_model'):
                    model.module.reconstruction_model = stamp_scheduler.get_model()
                # Recreate optimizer with new model parameters
                optimizer, optimizer_sub = build_optim(args, model)
                # Recreate scheduler with remaining epochs
                remaining_epochs = args.num_epochs - epoch
                if remaining_epochs > 3:
                    scheduler, scheduler_sub = build_scheduler(optimizer, optimizer_sub, args)
                print(f"  [STAMP] Optimizer recreated successfully")
        
        # Dynamic U-Net: channel swapping happens automatically during training batches
        # (see train_epoch function where maybe_swap_channels() is called)
        
        dev_loss, dev_time, psnr_mean, ssim_mean = evaluate(args, epoch + 1, model, dev_loader, writer, adv_mask)
        metrics += (dev_loss, dev_time, psnr_mean, ssim_mean)
        if best_psnr_mean < psnr_mean:
            is_new_best = True
            best_psnr_mean = psnr_mean
            best_ssim_mean = ssim_mean
            best_epoch = epoch
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, model, optimizer, scheduler, best_dev_loss, best_psnr_mean, best_ssim_mean, is_new_best, metrics, rigl_scheduler)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )

        if noise_behaviour == "linear_PGD":
            model.module.increase_noise_linearly()

    print(args.test_name)
    eval(args, model, dev_loader)
    print(f'Training done, best epoch: {best_epoch}')
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='gaussiantsp-d24-a1e-3-v1e-3', help='name for the output dir')
    parser.add_argument(
        '--train-dataset-name',
        type=str,
        default=None,
        help='Optional: override the training dataset name embedded in run/checkpoint directories '
             '(default: fastmri for --dataset=mri, imagenet for --dataset=imagenet).',
    )
    parser.add_argument('--exp-dir', type=pathlib.Path, default='summary/testepi',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default='summary/test/model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--report-interval', type=int, default=1, help='Period of loss reporting')

    # model parameters
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--decimation-rate', default=10, type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for each volume.')

    # optimization parameters
    parser.add_argument('--batch-size', default=9, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.01,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--sub-lr', type=float, default=1e-1, help='lerning rate of the sub-samping layel')

    # trajectory learning parameters
    parser.add_argument('--trajectory-learning', default=1,
                        help='trajectory_learning, if set to False, fixed trajectory, only reconstruction learning.')
    parser.add_argument('--acc-weight', type=float, default=1e-2, help='weight of the acceleration loss')
    parser.add_argument('--vel-weight', type=float, default=1e-1, help='weight of the velocity loss')
    parser.add_argument('--rec-weight', type=float, default=1, help='weight of the reconstruction loss')
    parser.add_argument('--gamma', type=float, default=42576, help='gyro magnetic ratio - kHz/T')
    parser.add_argument('--G-max', type=float, default=40, help='maximum gradient (peak current) - mT/m')
    parser.add_argument('--S-max', type=float, default=200, help='maximum slew-rate - T/m/s')
    parser.add_argument('--FOV', type=float, default=0.2, help='Field Of View - in m')
    parser.add_argument('--dt', type=float, default=1e-5, help='sampling time - sec')
    parser.add_argument('--a-max', type=float, default=0.17, help='maximum acceleration')
    parser.add_argument('--v-max', type=float, default=3.4, help='maximum velocity')
    parser.add_argument('--TSP', action='store_true', default=False,
                        help='Using the PILOT-TSP algorithm,if False using PILOT.')
    parser.add_argument('--TSP-epoch', default=20, type=int, help='Epoch to preform the TSP reorder at')
    parser.add_argument('--initialization', type=str, default='radial',
                        help='Trajectory initialization when using PILOT (spiral, EPI, rosette, uniform, gaussian).')
    parser.add_argument('--SNR', action='store_true', default=False,
                        help='add SNR decay')
    parser.add_argument('--n-shots', type=int, default=16,
                        help='Number of shots')
    parser.add_argument('--interp_gap', type=int, default=10,
                        help='number of interpolated points between 2 parameter points in the trajectory')
    parser.add_argument('--model', type=str, default='Unet',
                        help='the model type and params of the reconstruction net')
    
    # Dynamic U-Net parameters
    parser.add_argument('--swap-frequency', type=int, default=10,
                        help='Perform channel swap every N batches (for DynamicUnet)')
    
    # CondUnet parameters
    parser.add_argument('--num-experts', type=int, default=8,
                        help='Number of expert kernels per CondConv layer (for CondUnet)')
    
    # FDUnet parameters
    parser.add_argument('--fd-kernel-num', type=int, default=64,
                        help='Number of frequency-diverse kernels (for FDUnet)')
    parser.add_argument('--fd-use-simple', action='store_true', default=False,
                        help='Use simplified FDConv for faster computation (for FDUnet)')
    
    # HybridSnakeFDUnet parameters
    parser.add_argument('--snake-layers', type=int, default=2,
                        help='Number of encoder layers to use Snake Conv (for HybridSnakeFDUnet)')
    parser.add_argument('--snake-kernel-size', type=int, default=9,
                        help='Kernel size for Snake Convolution (for HybridSnakeFDUnet)')
    
    # ConfigurableUNet parameters (LightUNet, LightDCN, LightFD, LightDCNFD)
    parser.add_argument('--use-dcn', action='store_true', default=False,
                        help='Use DCNv2 for skip connection alignment')
    parser.add_argument('--use-fdconv', action='store_true', default=False,
                        help='Use FDConv at bottleneck')
    
    parser.add_argument('--inter-gap-mode', type=str, default='constant',
                        help='How the interpolated gap will change during the training')
    parser.add_argument('--img-size', type=int, nargs=2, default=[320, 320], help='Image size (height, width)')
    parser.add_argument('--in-chans', type=int, default=1, help='Number of input channels')
    parser.add_argument('--out-chans', type=int, default=1, help='Number of output channels')
    parser.add_argument('--num-blocks', type=int, default=1, help='Number of blocks in the model')
    parser.add_argument('--window-size', type=int, default=10, help='Window size for the model')
    parser.add_argument('--embed-dim', type=int, default=66, help='Embedding dimension for the model')
    parser.add_argument('--sample-per-shot', type=int, default=3001, help='Number of samples per shot')
    parser.add_argument('--noise-mode', type=str, default='ones',
                        help='Type of noise to be added (e.g., "ones" or "random")')
    parser.add_argument('--noise-behaviour', type=str, default='constant',
                        help='How the noise should behave (e.g., "constant" or "linear")')
    parser.add_argument('--epsilon', type=float, default=0, help='Starting value of epsilon for noise scaling')
    parser.add_argument('--end-epsilon', type=float, default=1e7, help='End value of epsilon for noise scaling')
    parser.add_argument('--noise-type', type=str, default='l1', help='Type of noise to be added (e.g., "l1", "l2")')
    parser.add_argument('--noise-p', type=float, default=0, help='Probability of applying noise during training')
    parser.add_argument('--noise-steps', type=int, default=0, help='The number of PGD steps to apply noise during training')
    parser.add_argument('--std', type=int, default=0, help='The std of the normal noise')
    parser.add_argument('--std-image', type=float, default=0, help='The std of the normal noise on the image')
    parser.add_argument('--acceleration', type=int, default=4, help='The Cartesian Acceleration')
    parser.add_argument('--center-fraction', type=float, default=0.08, help='The Cartesian Center Fraction')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='mri', choices=['mri', 'imagenet'],
                        help='Dataset to use: mri (FastMRI) or imagenet')
    parser.add_argument('--imagenet-path', type=pathlib.Path, default=pathlib.Path('../ImageNet'),
                        help='Path to ImageNet dataset (only used if --dataset=imagenet)')
    
    # RigL sparse training arguments
    parser.add_argument('--use-rigl', action='store_true', default=False,
                        help='Enable RigL dynamic sparse training')
    parser.add_argument('--rigl-sparsity', type=float, default=0.5,
                        help='Target sparsity for RigL (0.5 = 50% zeros)')
    parser.add_argument('--rigl-update-freq', type=int, default=100,
                        help='How often to update sparse masks (in iterations)')
    parser.add_argument('--rigl-delta', type=float, default=0.3,
                        help='Fraction of weights to reallocate each update')
    parser.add_argument('--rigl-initial-mask', type=str, default=None,
                        help='Path to pre-computed masks (.pt file from SNIP/PUN-IT) for RigL initialization')
    parser.add_argument('--rigl-static-mask', action='store_true',
                        help='Keep mask static (no RigL updates) - pure static sparse training')
    
    # Frozen mask training (sparse fine-tuning)
    parser.add_argument('--freeze-masked', action='store_true', default=False,
                        help='Train only non-masked (active) weights, keep masked weights frozen at zero')
    parser.add_argument('--freeze-masked-decoder-only', action='store_true', default=False,
                        help='Train only DECODER pruned weights, encoder/bottleneck pruned stay at 0')
    parser.add_argument('--rigl-checkpoint', type=str, default=None,
                        help='Path to RigL checkpoint to load masks from (for --freeze-masked)')
    
    # STAMP channel-wise pruning arguments (paper Section 2.5.1 defaults)
    # Based on: https://github.com/nkdinsdale/STAMP.git
    parser.add_argument('--use-stamp', action='store_true', default=False,
                        help='Enable STAMP simultaneous training and channel pruning')
    parser.add_argument('--stamp-channel-drop-rate', type=float, default=0.1,
                        help='b_drop: Channel dropout rate (paper default: 0.1)')
    parser.add_argument('--stamp-prune-ratio', type=float, default=0.5,
                        help='Fraction of channels to keep at each prune (0.5 = keep 50%)')
    parser.add_argument('--stamp-recovery-epochs', type=int, default=5,
                        help='Recovery epochs between prunings (paper default: 5)')
    parser.add_argument('--stamp-prune-epochs', type=int, nargs='+', default=[],
                        help='Specific epochs to prune (empty = auto-generate)')
    parser.add_argument('--stamp-mode', type=str, default='Taylor',
                        choices=['Taylor', 'L1', 'L2', 'Random'],
                        help='Importance scoring mode: Taylor (gradient*activation, default), L1, L2, Random')
    return parser


if __name__ == '__main__':
    train()
