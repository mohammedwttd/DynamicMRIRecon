import torch
from torch import nn
from torch.nn import functional as F
import json
import fastmri.models
from models.rec_models.unet_model import UnetModel
from models.rec_models.complex_unet import ComplexUnetModel
import data.transforms as transforms
from pytorch_nufft.nufft2 import nufft, nufft_adjoint
import numpy as np
from WaveformProjection.run_projection import proj_handler
import matplotlib.pylab as P
from models.rec_models.vision_transformer import VisionTransformer
from models.rec_models.recon_net import ReconNet
from models.rec_models.humus_net import HUMUSNet, HUMUSBlock
from  models.VarBlock import VarNet
from typing import Tuple
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import apply_mask
from models.rec_models.dcg import Generator, Discriminator
import random
import fastmri

import os
import torch
import numpy as np

import torch
import torchvision
import matplotlib.pyplot as plt

class NoiseScheduler:
    def __init__(
        self,
        max_noise: float,
        warmup_steps: int = 4,
        total_steps: int = 0,
        final_noise: float = 0.0,
        constant_steps: int = 0,
        mode: str = "linear",
    ):
        """
        max_noise: Peak noise level.
        warmup_steps: Steps to linearly warm up from 0 → max_noise.
        total_steps: Total number of steps (including warmup and decay).
        final_noise: Noise level at the end of decay.
        constant_steps: Number of steps to keep noise constant after warmup.
        mode: Type of schedule ("constant", "linear_decay", "cosine_decay").
        static: If True, noise stays at max_noise always (ignores warmup/decay).
        """
        self.max_noise = max_noise
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.total_steps = total_steps
        self.final_noise = final_noise
        self.mode = mode
        self.step_count = 0
        self.static = False
        if "linear" not in mode and "constant" not in mode:
            self.static = True

    def step(self):
        self.step_count += 1

    def get_noise(self):
        if self.static:  # <-- Static mode
            return self.max_noise

        t = self.step_count

        # 1. Warmup phase
        if t < self.warmup_steps:
            return self.max_noise * (t / self.warmup_steps)

        # 2. Constant phase
        elif t < self.warmup_steps + self.constant_steps:
            return self.max_noise

        # 3. Decay phase
        else:
            decay_t = t - (self.warmup_steps + self.constant_steps)
            decay_steps = max(1, self.total_steps - (self.warmup_steps + self.constant_steps))

            if "constant" in self.mode:
                return self.max_noise
            elif "linear" in self.mode:
                ratio = max(0.0, 1 - decay_t / decay_steps)
                return self.final_noise + (self.max_noise - self.final_noise) * ratio
            elif "cosine" in self.mode:
                from math import cos, pi
                ratio = 0.5 * (1 + cos(pi * decay_t / decay_steps))
                return self.final_noise + (self.max_noise - self.final_noise) * ratio
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

def perturb_Bimask(Bimask, move_std=20):
    _, _, H, _, _ = Bimask.shape
    Bimask = Bimask.clone()
    perturbed = Bimask.clone()

    selected_rows = (Bimask[0, 0, :, 0, 0] == 1).nonzero(as_tuple=False).squeeze(-1)
    occupied = set(selected_rows.tolist())

    for y in selected_rows.tolist():
        for _ in range(10):
            eps = torch.randn((1,))
            sample = move_std * eps
            shift = int(round(sample.item()))
            y_new = y + shift
            if 0 <= y_new < H and y_new not in occupied:
                perturbed[0, 0, y, 0, 0] = 0
                perturbed[0, 0, y_new, 0, 0] = 1
                occupied.remove(y)
                occupied.add(y_new)
                break

    noise_mask = (perturbed != Bimask).float()
    return noise_mask
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

class SubsamplingBinary(nn.Module):
    def initialize_random_mask(self, num_cols, acceleration, center_fraction):
        rng = np.random.RandomState(42)
        num_low_freqs = int(round(num_cols * center_fraction))
        num_high_freq = int(num_cols / acceleration) - num_low_freqs
        high_freq_mask = rng.uniform(size=(num_cols - num_low_freqs))
        sorted_vals = np.sort(high_freq_mask)
        threshold = sorted_vals[-num_high_freq]
        high_freq_mask[high_freq_mask >= threshold] = 1
        high_freq_mask[high_freq_mask < threshold] = 0
        pad = (num_cols - num_low_freqs + 1) // 2
        low_freq_mask = np.ones(num_low_freqs)
        mask = np.concatenate((high_freq_mask[:pad], low_freq_mask, high_freq_mask[pad:]))
        mask = mask * 0.5 + 0.5 * rng.uniform(size=num_cols)
        return torch.tensor(mask, dtype=torch.float)

    def __init__(self, res, acceleration=4, center_fraction=0.08, momentum=0.9, use_random=False, trajectory_learning = True, noise_cartesian = 0, noise_p = 0, noise_model = None, adv = False):
        super().__init__()
        self.res = res
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.momentum = momentum
        self.use_random = use_random
        # Initialize velocity as a tensor with the same shape as mask
        mask = self.initialize_random_mask(res, self.acceleration, self.center_fraction)
        self.velocity = torch.zeros_like(mask)
        self.mask = torch.nn.Parameter(mask, requires_grad=False)
        mask_shape = [1, 1, res, 1, 1]
        Bimask_1 = torch.tensor(mask)
        self.Bimask = torch.nn.Parameter(torch.reshape(Bimask_1, mask_shape), requires_grad=bool(int(trajectory_learning)))
        self.trajectory_learning = bool(int(trajectory_learning))
        self.noise_cartesian = noise_cartesian
        self.noise_p = noise_p
        self.noise_model = noise_model
        self.attack_trajectory_cartesian = None
        self.adv = adv

    def get_mask(self):
        res = self.res
        sorted_parameters, _ = torch.sort(self.mask)
        threshold = sorted_parameters[res - int(res / self.acceleration)]
        return self.mask >= threshold

    def make_mask(self, shape):
        res = shape[2]
        if self.use_random:
            with torch.no_grad():
                self.mask.data = self.initialize_random_mask(res, self.acceleration, self.center_fraction).to('cuda')

        sorted_parameters, _ = torch.sort(self.mask)
        threshold = sorted_parameters[res - int(res / self.acceleration)]
        with torch.no_grad():
            # Create a new tensor instead of in-place update
            new_bimask = torch.zeros_like(self.Bimask)
            new_bimask[0, 0, :, 0, 0] = (self.mask >= threshold).float()
            self.Bimask.data = new_bimask  # Update data safely
        return

    def transform(self, input):
        input = fastmri.ifft2c(input)
        input = transforms.complex_abs(input)
        input = normalize(input)
        return input

    def forward(self, input):
        # input: [B, 1, H, W, 2] (complex image domain)
        self.make_mask(input.shape)
        input_c = fastmri.fft2c(input).clone()  # Clone to avoid view issues
        if input_c.device != self.Bimask.device:
            self.Bimask = self.Bimask.to(input_c.device)

        mask = None
        noise_mask_rand = None
        if self.attack_trajectory_cartesian is not None:
            mask = self.attack_trajectory_cartesian
            mask = mask.view(1, 1, 320, 1, 1)
            print("applied cartesian noise adv", self.noise_model.get_noise())

        elif random.random() < self.noise_p and (self.training) and self.noise_model.get_noise() > 0:
            print("applied cartesian noise ", self.noise_model.get_noise())
            noise = self.noise_model.get_noise()
            mask = perturb_Bimask(self.Bimask, noise)

        if mask is not None:
            noise_mask_rand = self.Bimask * (1 - mask) + (1 - self.Bimask) * mask

        if noise_mask_rand != None:
            input_c_m = input_c * noise_mask_rand
        else:
            input_c_m = input_c * self.Bimask

        return self.transform(input_c_m)

    def apply_binary_grad(self, lr):
        if not self.trajectory_learning:
            return

        self.velocity = self.momentum * self.velocity.to("cuda") + (1 - self.momentum) * self.Bimask.grad[0, 0, :, 0, 0]
        if self.adv:
            new_mask = self.mask + lr * self.velocity
        else:
            new_mask = self.mask - lr * self.velocity
        new_mask = torch.clamp(new_mask, -1, 1)
        self.mask.data = new_mask

    def get_trajectory(self):
        def mask_to_trajectory(mask):
            mask = np.squeeze(self.get_mask())
            x_values = [x for x in range(-160, 160)]

            trajectory = []
            for i in range(320):
                if mask[i].item() == True:
                    for x in x_values:
                        trajectory.append([x, i - 160])
            return torch.tensor(trajectory).reshape(-1, 320, 2)

        return mask_to_trajectory(self.Bimask.data)


class Subsampling_Layer(nn.Module):    
    def initilaize_trajectory(self, trajectory_learning, initialization, n_shots):
        # x = torch.zeros(self.num_measurements, 2)
        sample_per_shot = self.sample_per_shot # TODO, try to increase this (instead of #of shots)
        print(sample_per_shot)
        if initialization == 'spiral':
            x = np.load(f'spiral/{n_shots}int_spiral_low.npy')
            x = torch.tensor(x[:, :sample_per_shot, :]).float()
        elif initialization == 'spiral_high':
            x = np.load(f'spiral/{n_shots}int_spiral.npy') * 10
            x = torch.tensor(x[:, :sample_per_shot, :]).float()
        elif initialization == 'EPI':
            x = torch.zeros(n_shots, sample_per_shot, 2)
            v_space = self.res // n_shots
            for i in range(n_shots):
                index = 0
                for j in range(sample_per_shot):
                    x[i, index, 1] = (i + 0.5) * v_space - 160
                    x[i, index, 0] = j * 320 / sample_per_shot - 160
                    index += 1
        elif initialization == 'radial':
            x = torch.zeros(n_shots, sample_per_shot, 2)
            theta = np.pi / n_shots
            for i in range(n_shots):
                L = torch.arange(-160, 160, 320 / sample_per_shot).float()
                x[i, :, 0] = L * np.cos(theta * i)
                x[i, :, 1] = L * np.sin(theta * i)
        elif initialization == 'uniform':
            x = (torch.rand(n_shots, sample_per_shot, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(n_shots, sample_per_shot, 2) * self.res / 6
        else:
            print('Wrong initialization')
        self.x = torch.nn.Parameter(x, requires_grad=bool(int(trajectory_learning)))
        self.trajectory_learning = bool(int(trajectory_learning))
        return

    def __init__(self, decimation_rate, res, trajectory_learning, initialization, n_shots, interp_gap, SNR=False,
                 projection_iters=10, sample_per_shot = 3001, interp_gap_mode = 0, epsilon = 0, noise_p = 0, noise_model = None):
        super().__init__()

        random.seed(42)
        self.sample_per_shot = sample_per_shot
        self.decimation_rate = decimation_rate
        self.res = res
        self.num_measurements = res ** 2 // decimation_rate
        self.initilaize_trajectory(trajectory_learning, initialization, n_shots)
        self.SNR = SNR
        self.interp_gap = interp_gap
        self.iters = projection_iters
        self.epsilon = epsilon
        self.noise_p = noise_p
        self.attack_trajectory_radial = None
        self.noise_model = noise_model

    def add_normed_noise(self, x_full):
        if not self.training:
            return x_full

        if self.attack_trajectory_radial != None:
            print("applied PGD!")
            return  torch.clamp(self.attack_trajectory_radial.reshape(-1, 2) + x_full, min=-160, max=160)

        if self.noise_p >= 0 and (random.random() <= (1 - self.noise_p)):
            return x_full

        elif self.noise_model.get_noise() > 0:
            print("applied std path! ", self.noise_model.get_noise())
            scaled_noise = torch.randn_like(x_full) * self.noise_model.get_noise()
            noisy_x = x_full + scaled_noise
            noisy_x = torch.clamp(noisy_x, min=-160, max=160)
            return noisy_x

        return x_full

    def forward(self, input):
        self.x.requires_grad_(self.trajectory_learning)
        if self.interp_gap > 1:
            t = torch.arange(0, self.x.shape[1], device=self.x.device).float()
            t1 = t[::self.interp_gap]
            x_short = self.x[:, ::self.interp_gap, :]
            for shot in range(x_short.shape[0]):
                for d in range(2):
                    self.x.data[shot, :, d] = self.interp(t1, x_short[shot, :, d], t)

        x_full = torch.clamp(self.add_normed_noise(self.x.reshape(-1,2)), min=-160, max=160)

        input = input.permute(0, 1, 4, 2, 3)
        sub_ksp = nufft(input, x_full)
        if self.SNR:
            noise_amp = 0.001
            noise = noise_amp * torch.randn(sub_ksp.shape)
            sub_ksp = sub_ksp + noise.to(sub_ksp.device)
        output = nufft_adjoint(sub_ksp, x_full, input.shape)
        output = output.permute(0, 1, 3, 4, 2)
        output = transforms.complex_abs(output)
        output = normalize(output)
        return output

    def get_trajectory(self):
        return self.x

    def h_poly(self, t):
        tt = [None for _ in range(4)]
        tt[0] = 1
        for i in range(1, 4):
            tt[i] = tt[i - 1] * t
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=tt[-1].dtype)
        return [
            sum(A[i, j] * tt[j] for j in range(4))
            for i in range(4)]

    def interp(self, x, y, xs):
        m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
        I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
        I = torch.clamp(I, 0, x.shape[0] - 2) 
        dx = (x[I + 1] - x[I])
        hh = self.h_poly((xs - x[I]) / dx)
        return hh[0] * y[I] + hh[1] * m[I] * dx + hh[2] * y[I + 1] + hh[3] * m[I + 1] * dx


    def __repr__(self):
        return f'Subsampling_Layer'


class Subsampling_Model(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, decimation_rate, res,
                 trajectory_learning, initialization, n_shots, interp_gap, SNR=False, type=None,
                 img_size=(320, 320) ,window_size=10, embed_dim=66, num_blocks=1, sample_per_shot=3001,
                 epsilon = 0, epsilon_step = None, noise_p = 0, std = 0, acceleration=4, center_fraction=0.08, noise ="", epochs = 40):
        super().__init__()
        self.type = type
        self.noise_model = NoiseScheduler(epsilon,warmup_steps=4, total_steps=epochs, final_noise=0, constant_steps=0, mode = noise)
        if initialization != "cartesian":
            self.subsampling = Subsampling_Layer(decimation_rate, res, trajectory_learning, initialization, n_shots,
                                                 interp_gap, SNR, sample_per_shot = sample_per_shot, epsilon = epsilon, noise_p = noise_p, noise_model = self.noise_model)
        else:
            self.subsampling = SubsamplingBinary(img_size[0], acceleration, center_fraction, trajectory_learning = trajectory_learning, noise_cartesian=std, noise_p=noise_p, noise_model = self.noise_model).to("cuda")

        if 'vit' in type:
            if "vit-l" in type:
                avrg_img_size = 320 if "pretrain" in type else 320
                patch_size = 10
                depth = 10
                num_heads = 16
                embed_dim = 44
            elif type == "vit-m":
                avrg_img_size = 320
                patch_size = 10
                depth = 8
                num_heads = 9
                embed_dim = 64
            else:
                avrg_img_size = 320
                patch_size = 10
                depth = 4
                num_heads = 9
                embed_dim = 44

            vitNet = VisionTransformer(
                avrg_img_size=avrg_img_size,
                patch_size=patch_size,
                in_chans=1, embed_dim=embed_dim,
                depth=depth, num_heads=num_heads,
            )
            self.reconstruction_model = ReconNet(vitNet)

        elif type == 'Unet':
            self.reconstruction_model = UnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob)
        elif type == 'humus':
            HUMUSReconstructionNet = HUMUSBlock(img_size=img_size, in_chans =in_chans, out_chans = out_chans, num_blocks = num_blocks, window_size = window_size, embed_dim = embed_dim)
            self.reconstruction_model = HUMUSReconstructionNet
        elif type == 'DCG':
            self.reconstruction_model = {"G": Generator().to("cuda"), "D": Discriminator().to("cuda")}
        

        self.iter = 0

    def forward(self, input):
        if "vit" in self.type:
            input = self.subsampling(input)
            output = self.reconstruction_model(input).reshape(-1, 320, 320)
            return output
        elif self.type == 'Unet':
            input = self.subsampling(input)
            output = self.reconstruction_model(input).reshape(-1, 320, 320)
            return output
        elif self.type == "humus":
            input = self.subsampling(input).squeeze(0).permute(0,3,1,2)
            out = self.reconstruction_model(input)
            out = out.permute(0,2,3,1)
            output = transforms.root_sum_of_squares(transforms.complex_abs(out.unsqueeze(0)), dim=1)
            return output.reshape(-1, 320, 320)
        elif self.type == "DCG":
            input = self.subsampling(input)
            input = transforms.root_sum_of_squares(transforms.complex_abs(input), dim=1).unsqueeze(1)
            output = self.reconstruction_model["G"](input).reshape(-1, 320, 320)
            return output
        return None

    def get_trajectory(self):
        return self.subsampling.get_trajectory()

    def increase_noise_linearly(self):
        self.subsampling.increase_noise_linearly()
