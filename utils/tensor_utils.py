import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gpu_device_config import device


def normalize(x, minmax=False, eps=1e-12):
    """Normalize each image in a batch to either unit energy or [0, 1]."""
    if x.dim() == 2:
        x = x[None, None, :, :]
    elif x.dim() == 3:
        x = x[:, None, :, :]

    batch_size, channels, height, width = x.shape
    x = x.reshape(batch_size, channels * height * width)
    if minmax:
        x = x - x.min(1, keepdim=True)[0]
        x = x / (x.max(1, keepdim=True)[0] + eps)
    else:
        x = x / (torch.sum(x, -1, keepdim=True) + eps)
    return x.reshape(batch_size, channels, height, width)


def tensor_to_list(x):
    """Convert a tensor with batch dimension into a list of per-item tensors."""
    return [item[None].to(device) for item in x]


def total_variation(img, reduction="sum"):
    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]
    res1 = pixel_dif1.abs()
    res2 = pixel_dif2.abs()

    if reduction == "mean":
        res1 = res1.mean(dim=(-2, -1))
        res2 = res2.mean(dim=(-2, -1))
    elif reduction == "sum":
        res1 = res1.sum(dim=(-2, -1))
        res2 = res2.sum(dim=(-2, -1))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
    return res1 + res2


def shift_and_crop(img, two_d_shift, centersize):
    x_center = int(img.shape[-2] / 2)
    y_center = int(img.shape[-1] / 2)
    half_centersize = int(centersize / 2)
    return img[
        ...,
        (x_center - half_centersize) + two_d_shift[-2] : (x_center + half_centersize) + two_d_shift[-2],
        y_center - half_centersize + two_d_shift[-1] : y_center + half_centersize + two_d_shift[-1],
    ]


def central_crop(variable, tw=None, th=None, dim=2):
    if dim == 2:
        w = variable.shape[-2]
        h = variable.shape[-1]
        if w >= tw and h >= th:
            x1 = int(round((w - tw) / 2.0))
            y1 = int(math.ceil((h - th) / 2.0))
            return variable[..., x1 : x1 + tw, y1 : y1 + th]
    elif dim == 1:
        h = variable.shape[-1]
        if h >= th:
            y1 = int(round((h - th) / 2.0))
            return variable[..., y1 : y1 + th]
    raise NotImplementedError


class InterpolateComplex2d(nn.Module):
    def __init__(self, input_dx, input_field_shape, output_dx, output_field_shape=None, mode="bicubic", del_intermediate_var=False) -> None:
        super().__init__()
        self.mode = mode
        self.input_pad_scale = self.get_input_pad_scale(input_dx, input_field_shape, output_dx, output_field_shape)
        self.interpolated_input_field_shape = [
            int(input_dx * side_length * self.input_pad_scale / output_dx)
            for side_length in input_field_shape[-2:]
        ]
        self.output_field_shape = output_field_shape if output_field_shape is not None else self.interpolated_input_field_shape
        self.del_intermediate_var = del_intermediate_var
        self.scale_factor = input_dx / output_dx

    def get_input_pad_scale(self, input_dx, input_field_shape, output_dx, output_field_shape):
        if output_field_shape is None:
            return 1

        if input_dx * input_field_shape[-2] <= output_dx * output_field_shape[-2]:
            input_pad_scale_x = (output_dx * output_field_shape[-2]) / (input_dx * input_field_shape[-2])
        else:
            input_pad_scale_x = 1

        if input_dx * input_field_shape[-1] <= output_dx * output_field_shape[-1]:
            input_pad_scale_y = (output_dx * output_field_shape[-1]) / (input_dx * input_field_shape[-1])
        else:
            input_pad_scale_y = 1
        return max(input_pad_scale_y, input_pad_scale_x)

    def interp_complex(self, x):
        x_in_real_imag = torch.view_as_real(x)
        x_real_interpolated = F.interpolate(
            x_in_real_imag[..., 0],
            (self.interpolated_input_field_shape[-2], self.interpolated_input_field_shape[-1]),
            mode=self.mode,
            align_corners=False,
        )
        x_imag_interpolated = F.interpolate(
            x_in_real_imag[..., 1],
            (self.interpolated_input_field_shape[-2], self.interpolated_input_field_shape[-1]),
            mode=self.mode,
            align_corners=False,
        )
        x_interpolated = torch.stack([x_real_interpolated, x_imag_interpolated], dim=-1)

        if self.del_intermediate_var:
            del x_real_interpolated
            del x_imag_interpolated
        return torch.view_as_complex(x_interpolated)

    def circular_pad_or_crop(self, x):
        binary_outputs = torch.tensor(x.shape[-2:]) < torch.tensor(self.output_field_shape)
        intermediate_size = (
            binary_outputs * (torch.tensor(self.output_field_shape) - torch.tensor(x.shape[-2:]))
            + torch.tensor(x.shape[-2:])
        )

        x = circular_pad(x, w_padded=intermediate_size[-2].item(), h_padded=intermediate_size[-1].item())
        return central_crop(x, tw=self.output_field_shape[-2], th=self.output_field_shape[-1])

    def forward(self, x):
        x = circular_pad(x, pad_scale=self.input_pad_scale)
        x_interpolated = self.interp_complex(x)
        x = x_interpolated / self.scale_factor

        if self.del_intermediate_var:
            del x_interpolated

        if torch.prod(torch.tensor(x.shape[-2:]) >= torch.tensor(self.output_field_shape)):
            output = central_crop(x, tw=self.output_field_shape[-2], th=self.output_field_shape[-1])
        elif torch.prod(torch.tensor(x.shape[-2:]) < torch.tensor(self.output_field_shape)):
            output = circular_pad(x, w_padded=self.output_field_shape[-2], h_padded=self.output_field_shape[-1])
        else:
            output = self.circular_pad_or_crop(x)

        if self.del_intermediate_var:
            del x
        return output


def pad_stacked_complex(field, pad_axes, mode="constant", padval=0):
    real = F.pad(field[..., 0], pad_axes, mode=mode, value=padval)
    imag = F.pad(field[..., 1], pad_axes, mode=mode, value=padval)
    return torch.stack((real, imag), dim=-1)


def pad_image(field, target_shape, pytorch=True, stacked_complex=False, padval=0, mode="constant"):
    r"""Pad a 2D field up to target_shape on the last two spatial dimensions."""
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    if not (size_diff > 0).any():
        return field

    pad_total = np.maximum(size_diff, 0)
    pad_front = (pad_total + odd_dim) // 2
    pad_end = (pad_total + 1 - odd_dim) // 2

    if pytorch:
        pad_axes = [int(p) for pair in zip(pad_front[::-1], pad_end[::-1]) for p in pair]
        if stacked_complex:
            return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
        return nn.functional.pad(field, pad_axes, mode=mode, value=padval)

    leading_dims = field.ndim - 2
    if leading_dims > 0:
        pad_front = np.concatenate(([0] * leading_dims, pad_front))
        pad_end = np.concatenate(([0] * leading_dims, pad_end))
    return np.pad(field, tuple(zip(pad_front, pad_end)), mode, constant_values=padval)


def circular_pad(u, w_padded=None, h_padded=None, pad_scale=None):
    """Zero-pad the last two dimensions of a tensor."""
    w, h = u.shape[-2], u.shape[-1]
    if pad_scale is not None:
        w_padded, h_padded = w * pad_scale, h * pad_scale
    ww = int(round((w_padded - w) / 2.0))
    hh = int(math.ceil((h_padded - h) / 2.0))
    return F.pad(u, (hh, hh, ww, ww), mode="constant", value=0)


def shift_list(num_class, shift):
    if num_class == 10:
        return [
            [-shift, -shift],
            [-shift, 0],
            [-shift, shift],
            [0, int(-1.5 * shift)],
            [0, int(-0.5 * shift)],
            [0, int(0.5 * shift)],
            [0, int(1.5 * shift)],
            [shift, -shift],
            [shift, 0],
            [shift, shift],
        ]
    if num_class == 4:
        return [
            [-shift, -shift],
            [shift, shift],
            [-shift, shift],
            [shift, -shift],
        ]
    if num_class == 2:
        return [
            [-shift, -shift],
            [shift, shift],
        ]
    raise NotImplementedError("Unsupported number of classes: {}".format(num_class))


def convert_tensor_to_cpu(input_value):
    if torch.is_tensor(input_value) and input_value.device.type != "cpu":
        return input_value.cpu().numpy()
    return input_value
