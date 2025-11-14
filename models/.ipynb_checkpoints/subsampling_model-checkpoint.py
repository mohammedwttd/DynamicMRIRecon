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
    def initilaize_random_mask(self,num_cols,acceleration,center_fraction):
        rng = np.random.RandomState()
        num_low_freqs = int(round(num_cols * center_fraction))
        num_high_freq = int(num_cols / acceleration) - num_low_freqs
        high_freq_mask = rng.uniform(size=(num_cols - num_low_freqs))
        sorted = np.sort(high_freq_mask)
        threshold = sorted[-num_high_freq]
        high_freq_mask[high_freq_mask >= threshold] = 1
        high_freq_mask[high_freq_mask < threshold] = 0
        pad = (num_cols - num_low_freqs + 1) // 2
        low_freq_mask = np.ones(num_low_freqs)
        mask = np.concatenate((high_freq_mask[:pad], low_freq_mask, high_freq_mask[pad:]))
        #num_high_freq = int(num_cols / acceleration)
        #high_freq_mask = rng.uniform(size=num_cols)
        #sorted = np.sort(high_freq_mask)
        #threshold = sorted[-num_high_freq]
        #high_freq_mask[high_freq_mask >= threshold] = 1
        #high_freq_mask[high_freq_mask < threshold] = 0
        #mask=high_freq_mask
    
        mask = mask * 0.5 + 0.5 * rng.uniform(size=num_cols)
        # sorted_parameters = np.sort(mask)
        # threshold = sorted_parameters[320 - int(320 / acceleration)]
        # bimask = mask >= threshold
        # plt.imshow(np.ones((320,320))*bimask)
        # plt.show()
        # print(bimask)
        return torch.tensor(mask,dtype=torch.float)

    def __init__(self, res, acceleration=4, center_fraction=0.08, momentum=0.9, use_random=False):
        super().__init__()
        self.res = res 
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.momentum = momentum
        self.use_random = use_random
        self.velocity = 0.
        self.use_random=False

        mask= self.initilaize_random_mask(res, self.acceleration,self.center_fraction)
        self.mask = torch.nn.Parameter(mask, requires_grad=False)
        mask_shape = [1,1,res,1,1]
        Bimask_1=torch.tensor(mask)
        self.Bimask = torch.nn.Parameter(torch.reshape(Bimask_1, mask_shape),requires_grad=True)
            
    def make_mask(self, shape):
        res=shape[2]
        if self.use_random:
            with torch.no_grad():
                self.mask.data = self.initilaize_random_mask(res, self.acceleration,self.center_fraction).to('cuda')
               
        sorted_parameters, _ = torch.sort(self.mask)
        threshold = sorted_parameters[res - int(res / self.acceleration)]
        with torch.no_grad():
            self.Bimask[0,0,:,0,0]=self.mask >= threshold
        return

    def transform(self, input):
        input = fastmri.ifft2c(input)
        input = transforms.complex_abs(input)
        input = normalize(input, eps=1e-11)
        input = input.clamp(-6, 6)
        return input
        
    def forward(self, input):
        # input: [B, 1, H, W, 2] (complex image domain)
        self.make_mask(input.shape)
        input=fastmri.fft2c(input)*self.Bimask
        input=self.transform(input)
        return input


    def apply_binary_grad(self, lr):
        print(self.Bimask.grad.shape, self.Bimask.shape)
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * self.Bimask.grad[0, 0, :, 0, 0]
        self.mask -= lr * self.velocity
        self.mask.clamp_(-1, 1)
        
    def get_trajectory(self):
        def mask_to_trajectory(mask):
            mask = np.squeeze(mask)
    
            x_values = [x for x in range(-160, 160)]
    
            trajectory = []
            for i in range(mask.shape[0]):
                #print(mask[i].item())
                if mask[i].item() == 1:
                    for x in x_values:
                        trajectory.append([x, i - 160])
            return torch.tensor(trajectory)

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
                 projection_iters=10, sample_per_shot = 3001, interp_gap_mode = 0, epsilon = 0, epsilon_step = None, noise_p = 0, std = 0):
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
        self.epsilon_step = epsilon_step
        self.noise_p = noise_p
        self.attack_trajectory = None
        self.std = std
        print(noise_p)

    def increase_noise_linearly(self):
        """
        Linearly increases the noise magnitude epsilon over the specified steps.

        The magnitude increases from the initial value up to the target value at the final step.
        """
        if self.epsilon == 0:
            return

        self.epsilon += self.epsilon_step
        return self.epsilon


    def add_normed_noise(self, x_full):
        if self.noise_p >= 0 and (not self.training or random.random() <= (1 - self.noise_p)):
            return x_full

        if self.epsilon > 0 and self.attack_trajectory != None:
            print("applied PGD!")
            return  torch.clamp(self.attack_trajectory.reshape(-1, 2) + x_full, min=-160, max=160)

        elif self.std > 0:
            print("applied std path!")
            scaled_noise = torch.randn_like(x_full) * self.std
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

        x_full = self.add_normed_noise(self.x.reshape(-1,2))

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
                 epsilon = 0, epsilon_step = None, noise_p = 0, std = 0, acceleration=4, center_fraction=0.08):
        super().__init__()
        self.type = type
        if initialization != "cartesian":
            self.subsampling = Subsampling_Layer(decimation_rate, res, trajectory_learning, initialization, n_shots,
                                                 interp_gap, SNR, sample_per_shot = sample_per_shot, epsilon = epsilon,
                                                 epsilon_step = epsilon_step, noise_p = noise_p, std = std)
        else:
            self.subsampling = SubsamplingBinary(img_size[0], acceleration, center_fraction)
            
        if type == 'vit':
            avrg_img_size = 340
            patch_size = 10
            depth = 10
            num_heads = 16
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
        if self.type == "vit":
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
