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
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.args import Args
from data import transforms
from data.mri_data import SliceData
import matplotlib

from models.rec_models.recon_net import ReconNet
from models.rec_models.vit_model import VisionTransformer

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


def create_datasets(args):
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


def train_epoch(args, epoch, model, data_loader, optimizer, optimizer_sub, writer, scheduler, scheduler_sub, adv_mask = None):
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
            save_dir = f"Images_{args.model}/{args.test_name}/{epoch}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{iter}.png"
            save_image(torch.stack([output[0].view(resolution, resolution),
                                    target[0].view(resolution, resolution)]), save_path, f"{iter}")

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1.)
        optimizer.step()
        optimizer_sub.step()
        if args.initialization == 'cartesian':
            model.module.subsampling.apply_binary_grad(optimizer_sub.param_groups[0]['lr'])
        model.module.subsampling.attack_trajectory = None

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


def save_model(args, exp_dir, epoch, model, optimizer, scheduler, best_dev_loss, best_psnr_mean, best_ssim_mean, is_new_best, metrics):
    torch.save(
        {
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
        },
        f=exp_dir + '/model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')


def build_model(args):

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
    ).to(args.device)
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
    scheduler_sub = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_sub,
        max_lr=[args.sub_lr],  # One per param group
        total_steps=(args.num_epochs) if "vit-l-pretrained-radial" in args.model else args.num_epochs,    # = 40
        pct_start= 0.1,  # = 0.1 for 4 warmup epochs
        anneal_strategy='linear',       # linear decay after warmup
        cycle_momentum=False,           # disable momentum scheduling
        div_factor=10,                 # base_lr = max_lr / div_factor (i.e., start at 0)
        final_div_factor=1e9            # decay linearly to ~0
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr],  # One per param group
        total_steps=args.num_epochs,    # = 40
        pct_start= 0.1,  # = 0.1 for 4 warmup epochs
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
    args.exp_dir = f'summary/{args.test_name}'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir)
    with open(args.exp_dir + '/args.txt', "w") as text_file:
        print(vars(args), file=text_file)

    print(args.test_name, flush=True)
    args.checkpoint = f'summary/{args.test_name}/model.pt'
    optimizer = None
    g_params = None
    g_optimizer = None
    d_optimizer = None
    d_scheduler = None
    g_scheduler = None

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        # args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        best_psnr_mean = checkpoint['best_psnr_mean']
        best_ssim_mean = checkpoint['best_ssim_mean']
        start_epoch = checkpoint['epoch'] + 1
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
    dev_loss, dev_time = evaluate(args, 0, model, dev_loader, writer)
    print("started mid point", flush=True)
    best_epoch = 0
    metrics = []
    for epoch in range(start_epoch, args.num_epochs):
        if noise_behaviour == "log":
            model.module.subsampling.epsilon = np.logspace(np.log10(args.epsilon), np.log10(args.end_epsilon), num=args.num_epochs)[epoch]
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, optimizer_sub, writer, scheduler, scheduler_sub, adv_mask)
        dev_loss, dev_time, psnr_mean, ssim_mean = evaluate(args, epoch + 1, model, dev_loader, writer, adv_mask)
        metrics += (dev_loss, dev_time, psnr_mean, ssim_mean)
        if best_psnr_mean < psnr_mean:
            is_new_best = True
            best_psnr_mean = psnr_mean
            best_ssim_mean = ssim_mean
            best_epoch = epoch
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, model, optimizer, scheduler, best_dev_loss, best_psnr_mean, best_ssim_mean, is_new_best, metrics)
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
    return parser


if __name__ == '__main__':
    train()
