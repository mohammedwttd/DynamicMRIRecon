#!/home/mohammed-wa/miniconda3/envs/mpilot/bin/python
import os
import json
import shlex
import sys

from sympy.physics.units import acceleration

data_path = '/mnt/walkure_public/users/mohammedw/fastmri_downloads/'

lr = {
    'Unet': {
        'rec_lr': 5e-4,
        'sub_lr': {
            'cartesian': 0.025,
            'radial': 0.005
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'vit-l-pretrained-cartesian-decoder': {
        'rec_lr': 1e-4,
        'sub_lr': {
            'cartesian': 0.05,
            'radial': 0.0025
        }
    },
    'vit-l-pretrained-cartesian': {
        'rec_lr': 1e-4,
        'sub_lr': {
            'cartesian': 0.5,
            'radial': 0.01 * 3
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'vit-l-pretrained-radial': {
        'rec_lr': 1e-4,
        'sub_lr': {
            'cartesian': 0.5,
            'radial': 0.01
        },
        'noise': {
            'cartesian': 10,
            'radial': 30,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
    'vit-l': {
        'rec_lr': 5e-4,
        'sub_lr': {
            'cartesian': 0.1,
            'radial': 0.01
        },
        'noise': {
            'cartesian': 20,
            'radial': 80,
            'image': 6e-5,
            'radial_pgd': 1,
            'cartesian_pgd': 4,
            'none': 0
        }
    },
}



acc_weight = 0.005
vel_weight = 0.001
batch_size = 1
n_shots = 16

model = 'vit-l-pretrained-radial'
init = 'radial'
noise = ''
noise_behaviour = ''

num_epochs = 30
trajectory_learning = 1
sample_rate = 0.25
inter_gap_mode = "changing_downwards_15"

clr = lr[model]
sub_lr = clr['sub_lr'][init]
rec_lr = clr['rec_lr']
noise_std = clr['noise'][noise] if noise != '' else 0

TSP = ''
SNR = ''
weight_decay = 0
interp_gap = 10
acceleration = 4
center_fraction = 0.08

#model settings
img_size = [320, 320]
in_chans = 1
out_chans = 1
num_blocks = 1
sample_per_shot = 1600
drop_prob = 0.1

#relevant only for humus
window_size = 10
embed_dim = 66


#noise
noise_mode = None
epsilon = 0
noise_p = 0

if 'pgd' in noise:
    noise_behaviour += "_" + noise
    epsilon = noise_std
    noise_p = 0.5

if init == 'radial' and noise == 'radial':
    epsilon = noise_std
    noise_behaviour += "_noise"
    noise_p = 0.5

if init == 'cartesian' and noise == 'cartesian':
    epsilon = noise_std
    noise_behaviour += "_noise"
    noise_p = 0.5

if noise == 'image':
    epsilon = noise_std
    noise_behaviour += "_image"
    noise_p = 0.5


noise_type = "linf"
test_name = f'{n_shots}/{init}_{sample_rate}_'

if init == "cartesian":
    test_name += f'{acceleration}_{center_fraction}_'
else:
    test_name += f'{n_shots}_{sample_per_shot}_'

if trajectory_learning == 1:
    test_name += f'{rec_lr}_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}_{interp_gap}_{model}_{num_epochs}'
else:
    test_name += f'{rec_lr}_fixed_{model}_{num_epochs}'

if TSP == '--TSP':
    test_name += f'{rec_lr}_TSP_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}'

if SNR == '--SNR':
    test_name += '_SNR_flat_0.01'

if epsilon != 0:
    test_name += f"_{noise_behaviour}"
    test_name += f"_intensity_{epsilon}"
    test_name += f"_noise_p_{noise_p}"

command = f'python3 train.py --test-name={test_name} ' \
          f'--n-shots={n_shots} ' \
          f'--trajectory-learning={trajectory_learning} ' \
          f'--sub-lr={sub_lr} ' \
          f'--initialization={init} ' \
          f'--batch-size={batch_size} ' \
          f'--lr={rec_lr} ' \
          f'--num-epochs={num_epochs} ' \
          f'--acc-weight={acc_weight} ' \
          f'--vel-weight={vel_weight} ' \
          f'--data-path={data_path} ' \
          f'--sample-rate={sample_rate} ' \
          f'--data-parallel {TSP} ' \
          f'--weight-decay={weight_decay} ' \
          f'--inter-gap-mode={inter_gap_mode} ' \
          f'--model={model} ' \
          f'--in-chans={in_chans} ' \
          f'--out-chans={out_chans} ' \
          f'--num-blocks={num_blocks} ' \
          f'--window-size={window_size} ' \
          f'--embed-dim={embed_dim} ' \
          f'--interp_gap={interp_gap} ' \
          f'--drop-prob={drop_prob} '\
          f'--sample-per-shot={sample_per_shot} ' \
          f'--noise-mode={noise_mode} ' \
          f'--noise-behaviour={noise_behaviour} ' \
          f'--epsilon={epsilon} ' \
          f'--noise-type={noise_type} '  \
          f'--noise-p={noise_p} '  \
          f'--acceleration={acceleration} '  \
          f'--center-fraction={center_fraction}'

os.system(command)
