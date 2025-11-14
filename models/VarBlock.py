import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.transforms import ifft2_regular, fft2


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        #print(x.shape)
        return fft2(x)

    def sens_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return ifft2_regular(x)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
    ) -> torch.Tensor:
        soft_dc = (current_kspace - ref_kspace) * self.dc_weight
        input = self.sens_reduce(current_kspace).squeeze(0).squeeze(0).permute(0, 3, 1, 2)
        model_term = self.sens_expand(
            self.model(input)
        )
        return current_kspace - soft_dc - model_term


class VarNet(nn.Module):
    """
    Full variational network composed of multiple VarNetBlock cascades.
    """

    def __init__(self, num_cascades: int, reg_model_fn: callable):
        """
        Args:
            num_cascades: Number of VarNetBlock cascades.
            reg_model_fn: A callable that returns a new instance of the
                regularizer model for each cascade.
        """
        super().__init__()
        self.num_cascades = num_cascades
        self.cascades = nn.ModuleList(
            [VarNetBlock(reg_model_fn(i)) for i in range(num_cascades)]
        )

    def forward(self, kspace: torch.Tensor, ref_kspace: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kspace: Input k-space data of shape [batch_size, num_coils, height, width].
            ref_kspace: Reference k-space data of the same shape.

        Returns:
            Refined k-space data after processing through all cascades.
        """
        current_kspace = kspace
        for cascade in self.cascades:
            current_kspace = cascade(current_kspace, ref_kspace)

        return current_kspace

