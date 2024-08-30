"""optical propagators under various approximating conditions
"""

from config import *
import math
from utils.general_utils import circular_pad, InterpolateComplex2d
import torch
import torch.nn as nn

class FresnelProp(nn.Module):
    """ Use Fresnel approximation to model the free-space wave-prop.

    Property of using transfer function (TF) is that the "u_in" and "u_out" planes have the same sampling interval.

    TF are IR  are Modified according to the Chapeter 5 -- TT89-CH5 of the book "Computational Fourier Optics."

    Padding schemes are based on the analysis in Zhang et al. 2020 "Frequency sampling strategy for numerical diffraction calculations"

    if z==0, just pass the propagation step
    """

    def __init__(self,
                 input_dx,
                 input_field_shape,
                 output_dx=None,
                 output_field_shape=None,
                 wave_lengths=None,
                 z=1.,  # prop dist, key param here
                 response_type="transfer_function",
                 pad_scale=2.,
                 pre_compute_H=True,  # set False when H is learnable
                 del_intermediate_var=False,
                 interp_sign=True,
                 ) -> None:
        super().__init__()
        self.pad_scale = pad_scale
        self.pre_compute_H = pre_compute_H
        self.output_dx = output_dx

        self.output_field_shape = output_field_shape
        self.input_field_shape = input_field_shape

        self.interp_sign = interp_sign

        self.z_crictical, response_type = self.cal_z_critical(
            input_field_shape, input_dx, wave_lengths, z, response_type)
        self.z = z
        if self.pre_compute_H:
            # precalculate the transfer kernel to save computation cost
            self.H = self.get_prop_kernel(
                input_field_shape, input_dx, z, wave_lengths, response_type, pad_scale)
        else:
            self.input_dx, self.wave_lengths, self.response_type, self.pad_scale = input_dx, wave_lengths, response_type, pad_scale

        # interpolation at the output plane when dx_ouput != dx_input
        self.interpolate_complex_2d = InterpolateComplex2d(
            input_dx, [input_field_shape[i]*pad_scale for i in range(len(input_field_shape))], output_dx, output_field_shape,
            del_intermediate_var=del_intermediate_var)

        self.del_intermediate_var = del_intermediate_var

    def cal_z_critical(self, input_field_shape, input_dx, wave_lengths, z, response_type):
        z_c = 2 * input_field_shape[0] * input_dx**2 / wave_lengths
        if response_type is None:
            if z > z_c:
                response_type = 'impulse_response'
            else:
                response_type = 'transfer_function'
            print('z_c is {}, z is {}, response type is {}'.format(
                z_c, z, response_type))
        return z_c, response_type

    def get_prop_kernel(self, input_field_shape, input_dx, z, wave_lengths, response_type, pad_scale):
        k = torch.tensor(2*math.pi/wave_lengths)

        if response_type == "transfer_function":
            """directly generate H in frequency domain"""
            M, N = input_field_shape[-2], input_field_shape[-1]
            fx = torch.linspace(-1/(2*input_dx), 1 /
                                (2*input_dx), int(M*pad_scale)).to(device)
            fy = torch.linspace(-1/(2*input_dx), 1 /
                                (2*input_dx), int(N*pad_scale)).to(device)
            FX, FY = torch.meshgrid(fx, fy, indexing='ij')
            H = torch.exp(-1j*math.pi * wave_lengths *
                          z * (FX**2 + FY**2))[None, None]
            print('Ht', H.shape)

        elif response_type == "impulse_response":
            """
            1. generate h in spatial domain
            2. do the Fourier transform to get H
            """
            M, N = input_field_shape[-2], input_field_shape[-1],
            x = torch.linspace(-M*input_dx/(2), M *
                               input_dx/2, int(M)).to(device)
            y = torch.linspace(-N*input_dx/(2), N *
                               input_dx/2, int(N)).to(device)

            meshx, meshy = torch.meshgrid(x, y, indexing='ij')
            # eq 5.4 of TT89_ch5
            h = torch.exp(1j*k*z)/(1j*wave_lengths*z) * \
                torch.exp(1j*k/(2*z)*(meshx**2+meshy**2))

            # pad kernel to avoid error from circular convolution
            h = circular_pad(h, pad_scale=self.pad_scale)
            print('Hi', h.shape)
            H = torch.fft.fftshift(torch.fft.fft2(
                torch.fft.fftshift(h, dim=[-2, -1])), dim=[-2, -1]) * input_dx**2
        return H.to(device)

    def forward(self, u1, z=None):
        if z == 0. or self.z == 0.:
            # will not propagate but just do the interpolation and crop
            u2=u1
        else:
            if not self.pre_compute_H:
                assert z != None, 'pls input value of z'
                self.H = self.get_prop_kernel(
                    self.input_field_shape, self.input_dx, z, self.wave_lengths, self.response_type, self.pad_scale)

            u1 = circular_pad(u1, pad_scale=self.pad_scale)
            U1 = torch.fft.fftshift(torch.fft.fft2(
                torch.fft.fftshift(u1, dim=[-2, -1])), dim=[-2, -1])
            if self.del_intermediate_var: del u1
            U2 = U1 * self.H
            if self.del_intermediate_var: del U1
            u2 = torch.fft.ifftshift(torch.fft.ifft2(
                torch.fft.ifftshift(U2, dim=[-2, -1])), dim=[-2, -1])
            if self.del_intermediate_var: del U2

        # NOTE interpolate in case input_dx is not equal to output_dx
        if self.interp_sign:
            u2 = self.interpolate_complex_2d(u2)
        return u2