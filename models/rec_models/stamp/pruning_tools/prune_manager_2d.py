# Adapted from Nicola Dinsdale 2020
# 2D version of pruning manager for STAMP
# Original: https://github.com/nkdinsdale/STAMP.git
########################################################################################################################
import torch
from torch.autograd import Variable
import numpy as np
from .prune_unet import prune_conv_layer2d, replace_layer_new2d
from operator import itemgetter
from heapq import nsmallest

########################################################################################################################
class FilterPrunner2D():
    """
    2D version of FilterPrunner for computing filter importance.
    Adapted from official STAMP implementation.
    
    Computes importance scores for each convolutional filter using:
    - Taylor: gradient * activation (default in paper)
    - L1: L1 norm of activations
    - L2: L2 norm of activations
    - Random: random values (baseline)
    """
    def __init__(self, unet, mode='Taylor', use_cuda=True):
        self.unet = unet
        self.mode = mode
        self.use_cuda = use_cuda
        self.return_ranks = None
        self.reset()

    def reset(self):
        self.filter_ranks = {}
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

    def forward(self, x):
        """
        Forward pass that registers hooks to compute filter importance.
        Follows official STAMP UNet structure:
        - encoder1-4: Sequential blocks with Conv2d layers
        - pool1-4: MaxPool2d
        - bottleneck: Sequential block
        - upconv4-1: ConvTranspose2d
        - decoder4-1: Sequential blocks
        - conv: final Conv2d
        """
        self.reset()
        self.activations = []
        activation_index = 0
        
        # Store encoder outputs for skip connections
        enc1 = enc2 = enc3 = enc4 = None
        
        # Traverse the model following exact UNet structure
        for block_name, block_module in self.unet._modules.items():
            if isinstance(block_module, torch.nn.Sequential):
                # Process sequential blocks (encoders, bottleneck, decoders)
                for layer_idx, (layer_name, layer) in enumerate(block_module._modules.items()):
                    x = layer(x)
                    
                    if isinstance(layer, torch.nn.Conv2d):
                        # Register hook for importance computation
                        if self.mode == 'Taylor':
                            x.register_hook(self.compute_rank_taylor)
                        elif self.mode == 'Random':
                            x.register_hook(self.compute_rank_random)
                        elif self.mode == 'L1':
                            x.register_hook(self.compute_rank_l1)
                        elif self.mode == 'L2':
                            x.register_hook(self.compute_rank_l2)
                        
                        self.activations.append(x)
                        # Store mapping: activation_index -> (block_name, layer_idx)
                        self.activation_to_layer[activation_index] = (block_name, layer_idx, layer_name)
                        activation_index += 1
                
                # Store encoder outputs
                if block_name == 'encoder1':
                    enc1 = x
                elif block_name == 'encoder2':
                    enc2 = x
                elif block_name == 'encoder3':
                    enc3 = x
                elif block_name == 'encoder4':
                    enc4 = x
                    
            elif isinstance(block_module, torch.nn.MaxPool2d):
                x = block_module(x)
                
            elif isinstance(block_module, torch.nn.ConvTranspose2d):
                x = block_module(x)
                # Concatenate with skip connection
                if block_name == 'upconv4':
                    x = torch.cat((x, enc4), dim=1)
                elif block_name == 'upconv3':
                    x = torch.cat((x, enc3), dim=1)
                elif block_name == 'upconv2':
                    x = torch.cat((x, enc2), dim=1)
                elif block_name == 'upconv1':
                    x = torch.cat((x, enc1), dim=1)
                    
            elif isinstance(block_module, torch.nn.Conv2d):
                # Final output conv
                x = block_module(x)
        
        return x

    def compute_rank_taylor(self, grad):
        """Taylor importance: gradient * activation."""
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Average over batch, height, width -> per-filter importance
        taylor = taylor.mean(dim=(0, 2, 3)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def compute_rank_l1(self, grad):
        """L1 norm importance."""
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        # L1 norm per filter
        l1 = torch.norm(activation, p=1, dim=(2, 3))  # sum over H, W
        l1 = l1.mean(dim=0).data  # average over batch

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += l1
        self.grad_index += 1

    def compute_rank_l2(self, grad):
        """L2 norm importance."""
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        # L2 norm per filter
        l2 = torch.norm(activation, p=2, dim=(2, 3))  # over H, W
        l2 = l2.mean(dim=0).data  # average over batch

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += l2
        self.grad_index += 1

    def compute_rank_random(self, grad):
        """Random importance (baseline)."""
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        if activation.size(1) != 1:
            random = torch.rand(activation.size(1))
        else:
            random = torch.ones(activation.size(1)) * 100

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()

        self.filter_ranks[activation_index] += random.cpu()
        self.grad_index += 1

    def normalize_ranks_per_layer(self):
        """Normalize ranks per layer (official STAMP)."""
        if self.mode != 'Random':
            for i in self.filter_ranks:
                v = torch.abs(self.filter_ranks[i])
                v = v / (torch.sqrt(torch.sum(v * v)) + 1e-8)
                self.filter_ranks[i] = v.cpu()
        self.return_ranks = self.filter_ranks

    def lowest_ranking_filters(self, num):
        """Get the num lowest ranking filters."""
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                block_info = self.activation_to_layer.get(i, (i, 0, ''))
                data.append((block_info, j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))

    def get_prunning_plan(self, num_filters_to_prune):
        """Get pruning plan: list of (block_info, layer_idx, filter_idx)."""
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        
        # Group by block
        filters_to_prune_per_block = {}
        for (block_info, filter_idx, _) in filters_to_prune:
            block_name = block_info[0] if isinstance(block_info, tuple) else block_info
            layer_idx = block_info[1] if isinstance(block_info, tuple) else 0
            
            if block_name not in filters_to_prune_per_block:
                filters_to_prune_per_block[block_name] = {}
            if layer_idx not in filters_to_prune_per_block[block_name]:
                filters_to_prune_per_block[block_name][layer_idx] = []
            filters_to_prune_per_block[block_name][layer_idx].append(filter_idx)
        
        # Sort and adjust indices
        for block in filters_to_prune_per_block:
            for layer in filters_to_prune_per_block[block]:
                filters_to_prune_per_block[block][layer] = sorted(filters_to_prune_per_block[block][layer])
                for i in range(len(filters_to_prune_per_block[block][layer])):
                    filters_to_prune_per_block[block][layer][i] -= i
        
        return filters_to_prune_per_block


class PruningController2D():
    """
    2D version of PruningController.
    Manages the pruning process for STAMP.
    """
    def __init__(self, unet, criterion, prune_percentage=10, mode='Taylor', use_cuda=True):
        self.unet = unet
        self.mode = mode
        self.pruner = FilterPrunner2D(self.unet, self.mode, use_cuda)
        self.criterion = criterion
        self.use_cuda = use_cuda
        self.prune_percentage = prune_percentage

    def compute_ranks(self, data, target):
        """Compute filter ranks for a single batch."""
        device = next(self.unet.parameters()).device
        data = data.to(device)
        target = target.to(device)

        self.unet.zero_grad()
        output = self.pruner.forward(data)
        loss = self.criterion(output, target)
        loss.backward()

    def compute_ranks_epoch(self, data_loader, num_batches=None):
        """Compute filter ranks over multiple batches."""
        self.pruner.reset()
        self.unet.train()
        
        for i, batch in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            
            # Handle different batch formats from DynamicMRIRecon
            if isinstance(batch, dict):
                # Typical fastMRI format
                if 'input' in batch:
                    data = batch['input']
                    target = batch.get('target', batch.get('target_img', data))
                elif 'kspace' in batch:
                    # Skip k-space data, need preprocessed input
                    continue
                else:
                    continue
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    data, target = batch[0], batch[1]
                else:
                    data = batch[0]
                    target = batch[0]
            else:
                data = batch
                target = batch
            
            try:
                self.compute_ranks(data, target)
            except Exception as e:
                print(f"[FilterPrunner2D] Batch {i} failed: {e}")
                continue
        
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.return_ranks

    def total_num_filters(self):
        """Count total convolutional filters."""
        filters = 0
        for name, module in self.unet._modules.items():
            if isinstance(module, torch.nn.Sequential):
                for layer in module.modules():
                    if isinstance(layer, torch.nn.Conv2d):
                        filters += layer.out_channels
        return filters

    def get_candidates_to_prune(self, num_filters_to_prune):
        """Get pruning candidates."""
        return self.pruner.get_prunning_plan(num_filters_to_prune), self.pruner.return_ranks

    def prune(self, num_filters=None):
        """
        Prune filters from the network.
        
        Note: For simplicity, this version updates dropout probabilities
        rather than physically removing filters, which is more stable
        for MRI reconstruction tasks.
        
        Returns:
            returned_ranks: Filter importance ranks for updating dropout
        """
        for param in self.unet.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        if num_filters is None:
            num_filters = int(self.prune_percentage)
        
        print(f'[STAMP Prune] Number of filters to prune: {num_filters}')
        print(f'[STAMP Prune] Total filters: {number_of_filters}')

        # Get pruning plan and ranks
        prune_plan, returned_ranks = self.get_candidates_to_prune(num_filters)
        
        print(f'[STAMP Prune] Pruning plan created for {len(prune_plan)} blocks')
        
        # Note: Physical pruning (prune_conv_layer2d) changes the network architecture
        # which can cause issues with the training loop. The main STAMP benefit
        # comes from the targeted dropout based on importance.
        # 
        # If you want actual filter removal, uncomment the following:
        # 
        # unet = self.unet.cpu()
        # for block_name, layers in prune_plan.items():
        #     for layer_idx, filter_indices in layers.items():
        #         for filter_idx in filter_indices:
        #             unet, _ = prune_conv_layer2d(unet, None, block_idx, layer_idx, filter_idx, use_cuda=False)
        # self.unet = unet
        # if self.use_cuda:
        #     self.unet = self.unet.cuda()

        return returned_ranks
