"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torchvision.transforms as tt

def to_tensor(data):
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)

def rfft2(data):
    data = ifftshift(data, dim=(-2, -1))
    data = torch.view_as_complex(data)
    data = torch.fft.rfft2(data,onesided=False)
    data = torch.view_as_real(data)
    data = fftshift(data, dim=(-3, -2))
    data=data.permute(0,1,4,2,3)
    data=data.squeeze(1)
    # data=data.reshape(1,2,320,320)
    return data
    
def rfft2_regular(data):
    data = ifftshift(data, dim=(-2, -1))
    data = torch.view_as_complex(data)
    data = torch.fft.rfft2(data,onesided=False)
    data = torch.view_as_real(data)
    data = fftshift(data, dim=(-3, -2))
    return data

def irfft2(data):
    data=data.unsqueeze(1)
    data = data.permute(0, 1, 3, 4, 2)
    # data=data.reshape(1,1,320,320,2)
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.irfft2(data,onesided=False)
    data = torch.view_as_real(data)
    data = fftshift(data, dim=(-2, -1))
    return data

def irfft2_regular(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.irfft2(data,onesided=False)
    data = torch.view_as_real(data)
    data = fftshift(data, dim=(-2, -1))
    return data

def fft2(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.fft2(data)
    data = torch.view_as_real(data)
    data = fftshift(data, dim=(-3, -2))
    return data

def ifft2(data):
    data=data.unsqueeze(1)
    data = data.permute(0, 1, 3, 4, 2)
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.ifft2(data)
    data = torch.view_as_real(data)
    data = fftshift(data, dim=(-3, -2))
    data = data.permute(0, 1, 4, 2, 3)
    data = data.squeeze(1)
    return data
    
    
def ifft2_regular(data):
    assert data.size(-1) == 2
    # Convert from two channels to complex
    data = ifftshift(data, dim=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.ifft2(data, norm="ortho")  # or norm=None
    data = torch.view_as_real(data)
    #data = torch.fft.ifft2(data)
    data = fftshift(data, dim=(-3, -2))

    return data
    
def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    if 0 >= shape[1] or shape[1] > data.shape[-1]:
        print(shape[0], shape[1], data.shape[-1], data.shape[-2], flush=True)
    if not  0 < shape[0] <= data.shape[-2]:
        print(data.shape)
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


import torch

def reflect_pad_to_shape(data, shape):
    """
    Pads an image to a target shape using reflection padding.

    Args:
        data (torch.Tensor): The input tensor to be padded.
        shape (int, int): The desired output shape.

    Returns:
        torch.Tensor: The padded image tensor.
    """

    # Calculate padding needed on each side
    pad_left = (shape[1] - data.shape[-1]) // 2
    pad_right = shape[1] - data.shape[-1] - pad_left
    pad_top = (shape[0] - data.shape[-2]) // 2
    pad_bottom = shape[0] - data.shape[-2] - pad_top

    # Reflect pad the image
    padded_data = torch.nn.functional.pad(data, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    return padded_data

def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std

def normalize_instance_per_channel(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    for i in range(data.shape[0]):
        for j in range(2):
            mean = data[i, :, :, j].mean()
            std = data[i, :, :, j].std()
            data[i, :, :, j] = normalize(data[i, :, :, j], mean, std, eps)
    return data, mean, std

# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)
