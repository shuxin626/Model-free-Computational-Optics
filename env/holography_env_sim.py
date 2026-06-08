"""Simulated coherent diffraction environment for CGH."""

from utils.gpu_device_config import device
from env.optics.propagator import FresnelProp
from env.optics.components import SLM
from utils.tensor_utils import central_crop
import torch
import torch.nn as nn


class HoloEnvSim(object):
    def __init__(
        self,
        input_dx,
        input_field_shape,
        output_dx,
        output_field_shape,
        wave_lengths,
        z,
        response_type,
        pad_scale,
        sensor_effective_shape,
        slm_size,
        partition,
        slm_type,
    ) -> None:
        self.input_field_shape = input_field_shape
        self.slm = SLM()
        self.upsample = nn.Upsample(scale_factor=slm_size / partition, mode='nearest')
        self.prop = FresnelProp(
            input_dx,
            input_field_shape,
            output_dx,
            output_field_shape,
            wave_lengths,
            z,
            response_type,
            pad_scale,
        )
        self.sensor_effective_shape = sensor_effective_shape
        self.slm_type = slm_type
        self.partition = partition

    def step(self, slm_mask):
        """Apply a batch of SLM phase masks and return output intensities."""
        if self.slm_type != 'phase':
            raise NotImplementedError("Only phase-only CGH is supported here.")

        if slm_mask.dim() == 3:
            slm_mask = slm_mask[:, None, :, :]
        if slm_mask.shape[-1] == 1 and slm_mask.dim() == 5:
            slm_mask = slm_mask[..., 0]

        with torch.no_grad():
            source = torch.ones(1, 1, *self.input_field_shape, device=device)
            slm_mask = self.upsample(slm_mask.to(device).float())
            x = self.slm(source, slm_mask)
            x = self.prop(x)
            x = torch.abs(x) ** 2
            x = central_crop(
                x,
                tw=self.sensor_effective_shape[-2],
                th=self.sensor_effective_shape[-1],
            )
        return x
