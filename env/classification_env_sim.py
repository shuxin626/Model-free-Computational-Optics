''' simulated coherent diffractive single-layer optical computing system
'''
from config import *
from optics.propagator import FresnelProp
from optics.optics import SLM, BaseAreaSensor, PlaneTilt
from utils.general_utils import central_crop
import torch
import torch.nn as nn

class OpticsSim(nn.Module):

    def __init__(self, optics_param, input_type, del_intermediate_var=False):
        super(OpticsSim, self).__init__()

        input_layer_effective_shape = optics_param['input_layer']['effective_shape']
        input_layer_pixel_size = optics_param['input_layer']['pixel_size']
        input_layer_tilt_degree = optics_param['input_layer']['tilt']

        optical_computing_layer_effective_shape = optics_param['optical_computing_layer']['effective_shape']
        optical_computing_layer_pixel_size = optics_param['optical_computing_layer']['pixel_size']
        optical_computing_layer_misalignment = optics_param['optical_computing_layer']['misalignment']

        camera_full_shape = optics_param['camera']['full_size']
        camera_effective_shape = optics_param['camera']['effective_shape']
        camera_pixel_size = optics_param['camera']['pixel_size']
        camera_misalignment = optics_param['camera']['misalignment']

        # extract other parameters
        interp_sign = optics_param['general']['interp_sign']
        response_type = optics_param['general']['response_type']
        wave_length = optics_param['light_source']['wave_length']
        pad_scale = optics_param['general']['pad_scale']

        self.slm = SLM(optical_computing_layer_misalignment)

        self.propogator_IC = FresnelProp(input_layer_pixel_size, input_layer_effective_shape, optical_computing_layer_pixel_size,
                                       optical_computing_layer_effective_shape, wave_length, optics_param[
                                           'propogator.IC']['length'], response_type, pad_scale,
                                       del_intermediate_var=del_intermediate_var,
                                       interp_sign=interp_sign,
                                       )

        self.propogator_CO = FresnelProp(optical_computing_layer_pixel_size, optical_computing_layer_effective_shape, camera_pixel_size,
                                       camera_full_shape, wave_length, optics_param[
                                           'propogator.CO']['length'], response_type, pad_scale,
                                       del_intermediate_var=del_intermediate_var,
                                       interp_sign=interp_sign)

        self.camera = BaseAreaSensor(camera_pixel_size, camera_full_shape,
                                     camera_pixel_size, camera_full_shape, camera_misalignment)

        # copy to class
        self.input_type = input_type
        self.input_layer_tilt_degree = input_layer_tilt_degree
        self.camera_effective_shape = camera_effective_shape
        self.del_intermediate_var = del_intermediate_var
        self.tilt_of_input_layer = PlaneTilt(
            self.input_layer_tilt_degree, input_layer_pixel_size, input_layer_effective_shape, wave_length)


    def forward(self, obj, slm_mask, propogator_CO_length=None, H_12_fun=None, H_23=None, output_crop=True):

        if self.input_type == 'phase_only':

            u_in = torch.exp(1j * obj[:, :1, :, :])

        else:
            raise Exception("Not a valid type")

        if self.input_layer_tilt_degree != 0.0:
            u_in = self.tilt_of_input_layer(u_in)

        u_out_ic = self.propogator_IC(u_in)
        if self.del_intermediate_var:
            del u_in

        u_out_slm = self.slm(u_out_ic, -slm_mask)
        if self.del_intermediate_var:
            del u_out_ic

        u_out_co = self.propogator_CO(u_out_slm, propogator_CO_length)

        if self.del_intermediate_var:
            del u_out_slm

        img_out = self.camera(u_out_co)

        img_out = torch.flip(img_out, dims=[-2, -1])

        if self.del_intermediate_var:
            del u_out_co
        if output_crop:
            img_out = central_crop(img_out, tw=self.camera_effective_shape[-2], th=self.camera_effective_shape[-1])

        return img_out