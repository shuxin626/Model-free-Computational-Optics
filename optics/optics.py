"""optical modules
- SLM: Spatial Light Modulator
- BaseAreaSensor: Area Camera
- PlaneTilt: Plane Tilt Module (used for testing)
"""

from config import *
from utils.general_utils import central_crop, shift_and_crop
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class SLM(nn.Module):
    def __init__(self, misalignment=[0, 0]):
        super(SLM, self).__init__()
        self.misalignment = misalignment

    def forward(self, input, slm_mask):

        slm_mask = F.pad(slm_mask, (abs(self.misalignment[-1]), abs(self.misalignment[-1]), abs(self.misalignment[-2]), abs(self.misalignment[-2])))
        slm_mask = torch.roll(slm_mask, self.misalignment, (-2, -1))
        slm_mask = slm_mask[..., abs(self.misalignment[-2]): -abs(self.misalignment[-2]) if self.misalignment[-2] != 0 else None, abs(self.misalignment[-1]):-abs(self.misalignment[-1]) if self.misalignment[-1] != 0 else None]
        output = torch.exp(1j*slm_mask) * input

        return output

class BaseAreaSensor(nn.Module):
    def __init__(self, input_pixel_size, input_effective_shape, output_pixel_size, output_effective_shape, center_shift):
        ''' Area Camera, modeld acc to the FLIR camera
        '''
        super().__init__()

        assert input_effective_shape[-2] * \
            input_pixel_size >= output_effective_shape[-2] * output_pixel_size
        assert input_effective_shape[-1] * \
            input_pixel_size >= output_effective_shape[-1] * output_pixel_size
        # copy to class
        self.interpolate_field_shape = [int(input_pixel_size/output_pixel_size*width_or_height)
                                        for width_or_height in input_effective_shape[-2:]]

        self.output_effective_shape = output_effective_shape
        self.center_shift = center_shift

    def forward(self, arriving_field):
        # convert incident complex field to discrete real camera measurements
        img_out1 = torch.abs(arriving_field) ** 2
        if abs(self.center_shift[-2]) + abs(self.center_shift[-1])  == 0.:
            img_out2 = central_crop(
                img_out1, tw=self.output_effective_shape[-2], th=self.output_effective_shape[-1])
        else:
            img_out2 = shift_and_crop(
                img_out1, self.center_shift, centersize=self.output_effective_shape[-1])

        return img_out2

class PlaneTilt(nn.Module):
    def __init__(self, angle_deg, dx, shape, wavelength):
        super(PlaneTilt, self).__init__()

        # prepare titl phase map
        x = torch.tensor(range(shape[0]))
        y = torch.tensor(range(shape[1]))
        grid_x, _ = torch.meshgrid(x, y)  # suppose only tilt along x direction
        grid_x = grid_x * dx # turn to actual length
        angle_rad = angle_deg / 180 * math.pi # turn degree to rad
        self.tilt_phase_map = grid_x * math.tan(angle_rad) / wavelength * 2 * math.pi
        self.tilt_phase_map = self.tilt_phase_map.to(device)

    def forward(self, input):
        output = input * torch.exp(1j * self.tilt_phase_map)
        return output
