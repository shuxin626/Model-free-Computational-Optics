''' simulated coherent diffractive optical computing systems
'''
from utils.gpu_device_config import device
from env.optics.propagator import FresnelProp
from env.optics.components import SLM, BaseAreaSensor, PlaneTilt
from utils.tensor_utils import central_crop, pad_image, shift_and_crop, shift_list
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


MASK_QUERY_CHUNK_SIZE = 36


class BaseOpticsSim(nn.Module):
    """Shared input and camera logic for simulated optical classifiers."""

    def __init__(
        self,
        input_type,
        input_layer_tilt_degree,
        input_layer_pixel_size,
        input_layer_effective_shape,
        wave_length,
        camera_effective_shape,
        del_intermediate_var=False,
    ):
        super().__init__()
        self.input_type = input_type
        self.input_layer_tilt_degree = input_layer_tilt_degree
        self.camera_effective_shape = camera_effective_shape
        self.del_intermediate_var = del_intermediate_var
        self.tilt_of_input_layer = PlaneTilt(
            self.input_layer_tilt_degree,
            input_layer_pixel_size,
            input_layer_effective_shape,
            wave_length,
        )

    def prepare_input_field(self, obj):
        if self.input_type != 'phase_only':
            raise ValueError("Unsupported optical input type: {}".format(self.input_type))

        u_in = torch.exp(1j * obj[:, :1, :, :])
        if self.input_layer_tilt_degree != 0.0:
            u_in = self.tilt_of_input_layer(u_in)
        return u_in

    def read_camera(self, arriving_field, output_crop=True):
        img_out = self.camera(arriving_field)
        img_out = torch.flip(img_out, dims=[-2, -1])

        if self.del_intermediate_var:
            del arriving_field
        if output_crop:
            img_out = central_crop(
                img_out,
                tw=self.camera_effective_shape[-2],
                th=self.camera_effective_shape[-1],
            )
        return img_out


class OpticsSim(BaseOpticsSim):

    def __init__(self, optics_param, input_type, del_intermediate_var=False):
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

        super().__init__(
            input_type,
            input_layer_tilt_degree,
            input_layer_pixel_size,
            input_layer_effective_shape,
            wave_length,
            camera_effective_shape,
            del_intermediate_var,
        )

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

    def forward(self, obj, slm_mask, propogator_CO_length=None, H_12_fun=None, H_23=None, output_crop=True):
        u_in = self.prepare_input_field(obj)
        u_out_ic = self.propogator_IC(u_in)
        if self.del_intermediate_var:
            del u_in

        u_out_slm = self.slm(u_out_ic, -slm_mask)
        if self.del_intermediate_var:
            del u_out_ic

        u_out_co = self.propogator_CO(u_out_slm, propogator_CO_length)

        if self.del_intermediate_var:
            del u_out_slm

        return self.read_camera(u_out_co, output_crop)


class OpticsSimTwoLayer(BaseOpticsSim):
    """Simulated two-layer optical computing system.

    The optical path is input layer -> optical computing layer 1 -> optical
    computing layer 2 -> camera. This is the simulator counterpart of the
    two-SLM real setup.
    """

    def __init__(self, optics_param, input_type, del_intermediate_var=False):
        input_layer_effective_shape = optics_param['input_layer']['effective_shape']
        input_layer_pixel_size = optics_param['input_layer']['pixel_size']
        input_layer_tilt_degree = optics_param['input_layer']['tilt']

        layer1_param = optics_param['optical_computing_layer1']
        layer2_param = optics_param['optical_computing_layer2']

        layer1_effective_shape = layer1_param['effective_shape']
        layer1_pixel_size = layer1_param['pixel_size']
        layer1_misalignment = layer1_param['misalignment']

        layer2_effective_shape = layer2_param['effective_shape']
        layer2_pixel_size = layer2_param['pixel_size']
        layer2_misalignment = layer2_param['misalignment']

        camera_full_shape = optics_param['camera']['full_size']
        camera_effective_shape = optics_param['camera']['effective_shape']
        camera_pixel_size = optics_param['camera']['pixel_size']
        camera_misalignment = optics_param['camera']['misalignment']

        interp_sign = optics_param['general']['interp_sign']
        response_type = optics_param['general']['response_type']
        wave_length = optics_param['light_source']['wave_length']
        pad_scale = optics_param['general']['pad_scale']

        super().__init__(
            input_type,
            input_layer_tilt_degree,
            input_layer_pixel_size,
            input_layer_effective_shape,
            wave_length,
            camera_effective_shape,
            del_intermediate_var,
        )

        self.slm1 = SLM(layer1_misalignment)
        self.slm2 = SLM(layer2_misalignment)

        self.propogator_input_layer1 = FresnelProp(
            input_layer_pixel_size,
            input_layer_effective_shape,
            layer1_pixel_size,
            layer1_effective_shape,
            wave_length,
            optics_param['propogator.IC1']['length'],
            response_type,
            pad_scale,
            del_intermediate_var=del_intermediate_var,
            interp_sign=interp_sign,
        )

        self.propogator_layer1_layer2 = FresnelProp(
            layer1_pixel_size,
            layer1_effective_shape,
            layer2_pixel_size,
            layer2_effective_shape,
            wave_length,
            optics_param['propogator.C1C2']['length'],
            response_type,
            pad_scale,
            del_intermediate_var=del_intermediate_var,
            interp_sign=interp_sign,
        )

        self.propogator_layer2_camera = FresnelProp(
            layer2_pixel_size,
            layer2_effective_shape,
            camera_pixel_size,
            camera_full_shape,
            wave_length,
            optics_param['propogator.C2O']['length'],
            response_type,
            pad_scale,
            del_intermediate_var=del_intermediate_var,
            interp_sign=interp_sign,
        )

        self.camera = BaseAreaSensor(camera_pixel_size, camera_full_shape,
                                     camera_pixel_size, camera_full_shape, camera_misalignment)

        self.layer_effective_shapes = [layer1_effective_shape, layer2_effective_shape]

    def forward(self, obj, slm_mask, output_crop=True):
        if slm_mask.dim() != 5 or slm_mask.shape[-1] != 2:
            raise ValueError("Two-layer simulation expects phase mask shape [1, query, height, width, 2].")

        u_in = self.prepare_input_field(obj)
        u_out = self.propogator_input_layer1(u_in)
        if self.del_intermediate_var:
            del u_in

        layer1_mask = F.interpolate(slm_mask[..., 0], size=self.layer_effective_shapes[0])
        u_out = self.slm1(u_out, layer1_mask)
        if self.del_intermediate_var:
            del layer1_mask

        u_out = self.propogator_layer1_layer2(u_out)

        layer2_mask = F.interpolate(slm_mask[..., 1], size=self.layer_effective_shapes[1])
        u_out = self.slm2(u_out, layer2_mask)
        if self.del_intermediate_var:
            del layer2_mask

        u_out = self.propogator_layer2_camera(u_out)
        return self.read_camera(u_out, output_crop)


def get_optical_weight_for_classifier(img, num_classes=4, dim=(-3, -2, -1),
                                      use_pbr_as_optical_weight=False, shift=30, crop_size=40):
    weight_list = [torch.mean(shift_and_crop(img, pos, crop_size), dim=dim)
                   for pos in shift_list(num_classes, shift)]
    x = torch.stack(weight_list, dim=-1)

    if use_pbr_as_optical_weight:
        sum_img = torch.sum(img, dim=dim).unsqueeze(-1)
        x = torch.log(x * crop_size * crop_size / sum_img)
    return x


def get_init_phase_mask(mask_num_partitions, effective_slm_shape, mask_representation, mask_init_type, actor=None,
                        num_optical_computing_layers=1):
    if mask_num_partitions is not None:
        assert mask_num_partitions <= effective_slm_shape[-1]
        assert effective_slm_shape[0] == effective_slm_shape[1]
        spatial_shape = [mask_num_partitions, mask_num_partitions]
    else:
        spatial_shape = [effective_slm_shape[0], effective_slm_shape[1]]

    if num_optical_computing_layers == 1:
        shape = [1, 1, spatial_shape[0], spatial_shape[1]]
    else:
        shape = [1, 1, spatial_shape[0], spatial_shape[1], num_optical_computing_layers]

    if mask_representation == "pixelwise":
        if actor is None:
            if mask_init_type == 'rand_init':
                mask_related_param = torch.rand(shape).to(device)
                mask_related_param = (mask_related_param - 0.5) * 2 * math.pi
            elif mask_init_type == 'zero_init':
                mask_related_param = torch.zeros(shape).to(device)
            else:
                raise NotImplementedError
            phase_mask = nn.Parameter(mask_related_param, requires_grad=True)
            phase_mask_clone = None
        else:
            phase_mask, phase_mask_clone = torch.tensor([]), torch.tensor([])
    else:
        raise NotImplementedError

    return phase_mask, phase_mask_clone


class OpticalClassifier(nn.Module):
    def __init__(self, input_type="intensity_only",
                 optics_param=None, number_of_classes=2, maskquery_batchsize=None, mask_num_partitions=None,
                 use_pbr_as_optical_weight=False, exp_param=None, actor=None, shift=30, crop_size=40,
                 optimizer_type='mfo', pg_type='loo', optics_param_dummy=None,
                 num_optical_computing_layers=1):
        """
        maskquery_batchsize: when use mfo, maskquery_batchsize is not None
        """
        super(OpticalClassifier, self).__init__()

        del_intermediate_var = True if optimizer_type == 'mfo' else False

        self.num_optical_computing_layers = num_optical_computing_layers
        self.optics_sim = self.build_optics_sim(optics_param, input_type, del_intermediate_var)

        if optimizer_type == 'hbt' or optimizer_type == 'sbt':
            self.optics_sim_dummy = self.build_optics_sim(optics_param_dummy, input_type, del_intermediate_var)

        self.phase_mask, self.phase_mask_clone = get_init_phase_mask(
            mask_num_partitions,
            optics_param['optical_computing_layer']['effective_shape'],
            optics_param['optical_computing_layer']['mask_representation'],
            optics_param['optical_computing_layer']['mask_init_type'],
            actor=actor,
            num_optical_computing_layers=num_optical_computing_layers,
        )

        self.number_of_classes = number_of_classes
        self.maskquery_batchsize = maskquery_batchsize
        self.use_pbr_as_optical_weight = use_pbr_as_optical_weight
        self.shift = shift
        self.crop_size = crop_size
        self.effective_shape = optics_param['optical_computing_layer']['effective_shape']
        self.optimizer_type = optimizer_type
        self.mask_query_chunk_size = MASK_QUERY_CHUNK_SIZE

    def build_optics_sim(self, optics_param, input_type, del_intermediate_var):
        if self.num_optical_computing_layers == 1:
            return OpticsSim(optics_param, input_type, del_intermediate_var)
        if self.num_optical_computing_layers == 2:
            return OpticsSimTwoLayer(optics_param, input_type, del_intermediate_var)
        raise NotImplementedError("Only one-layer and two-layer simulators are supported.")

    def run_optics_in_query_chunks(self, obj, phase_mask, optics_sim):
        maskquery_batchsize = phase_mask.shape[1]
        if maskquery_batchsize <= self.mask_query_chunk_size:
            return optics_sim(obj, phase_mask)

        img_chunks = []
        for start in range(0, maskquery_batchsize, self.mask_query_chunk_size):
            stop = start + self.mask_query_chunk_size
            img_chunks.append(optics_sim(obj, phase_mask[:, start:stop, ...]))
        return torch.cat(img_chunks, dim=1)

    def forward_sim(self, obj, phase_mask, if_test):
        if not if_test:
            if self.optimizer_type == 'mfo':
                with torch.no_grad():
                    img = self.run_optics_in_query_chunks(obj, phase_mask, self.optics_sim)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    weights = self.get_weights(img)

            elif self.optimizer_type == 'sbt':
                img = self.optics_sim_dummy(obj, phase_mask)
                weights = self.get_weights(img)

            elif self.optimizer_type == 'hbt':
                with torch.no_grad():
                    img = self.optics_sim(obj, phase_mask)
                    img_weights = self.get_weights(img)
                img_dummy = self.optics_sim_dummy(obj, phase_mask)
                img_dummy_weights = self.get_weights(img_dummy)
                weights = (img_weights - img_dummy_weights).detach() + img_dummy_weights
            else:
                raise NotImplementedError

        else:
            img = self.optics_sim(obj, phase_mask)
            weights = self.get_weights(img, True)

        img_out = img.clone()
        return weights, img_out

    def get_weights(self, img, if_test=False):
        if self.optimizer_type != 'mfo' or if_test:
            dim = (-3, -2, -1)
        else:
            dim = (-2, -1)

        logit_weights = get_optical_weight_for_classifier(
            img,
            num_classes=self.number_of_classes,
            dim=dim,
            use_pbr_as_optical_weight=self.use_pbr_as_optical_weight,
            shift=self.shift,
            crop_size=self.crop_size,
        )
        return logit_weights

    def normalize_phase_mask_shape(self, exogenous_phase_mask):
        if exogenous_phase_mask is None:
            phase_mask = self.phase_mask
        elif exogenous_phase_mask.dim() == 2:
            phase_mask = exogenous_phase_mask[None, None, ...]
        elif exogenous_phase_mask.dim() == 3 and self.num_optical_computing_layers == 1:
            phase_mask = exogenous_phase_mask[None, ...]
        elif exogenous_phase_mask.dim() == 3:
            phase_mask = exogenous_phase_mask[None, None, ...]
        else:
            phase_mask = exogenous_phase_mask

        if self.num_optical_computing_layers == 1:
            phase_mask = F.interpolate(phase_mask, size=[int(self.effective_shape[i]) for i in range(2)])
            phase_mask = pad_image(phase_mask, target_shape=self.effective_shape)
        return phase_mask

    def forward(self, obj, exogenous_phase_mask=None, if_test=False):
        phase_mask = self.normalize_phase_mask_shape(exogenous_phase_mask)

        self.interp_phase_mask = phase_mask

        weights, cam_img = self.forward_sim(obj, phase_mask, if_test)

        del phase_mask
        return weights, cam_img
