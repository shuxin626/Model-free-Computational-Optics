
from config import *
from env.classification_env_sim import OpticsSim
import torch.nn.functional as F
from utils.general_utils import shift_and_crop
import math
from utils.general_utils import shift_list, pad_image
import torch
import torch.nn as nn


def get_optical_weight_for_classfier(img, num_classes=4, dim=(-3, -2, -1),
                                     use_pbr_as_optical_weight=False, shift=30, crop_size=40):


      weight_list = [torch.mean(shift_and_crop(img, pos, crop_size), dim=dim)
                  for pos in shift_list(num_classes, shift)]
      x = torch.stack(weight_list, dim=-1)
      if use_pbr_as_optical_weight:
          # shape of sum_img [batchsize, maskquery_batchsize, 1, 1]
          sum_img = torch.sum(img, dim=dim)
          # shape of sum _img [batchsize, maskquery_batchsize, 1]
          sum_img = sum_img.unsqueeze(-1)

          # pbr: ROI/background_img
          x = torch.log(x * crop_size * crop_size /
                      (sum_img))
      return x


def get_init_phase_mask(mask_num_partitions, effective_slm_shape, mask_representation, mask_init_type, actor=None):

    if mask_num_partitions is not None:
        assert mask_num_partitions <= effective_slm_shape[-1]
        assert effective_slm_shape[0] == effective_slm_shape[1]
        shape = [mask_num_partitions, mask_num_partitions]
    else:
        shape = [effective_slm_shape[0], effective_slm_shape[1]]

    if mask_representation == "pixelwise":
        if actor is None:
            # HBT or SBT
            if mask_init_type == 'rand_init':
                mask_related_param = torch.rand(
                    1, 1, shape[0], shape[1]).to(device)
                mask_related_param = (mask_related_param - 0.5) * 2 * math.pi
            elif mask_init_type == 'zero_init':
                mask_related_param = torch.zeros(
                    1, 1, shape[0], shape[1]).to(device)
            phase_mask = nn.Parameter(mask_related_param, requires_grad=True)
            phase_mask_clone = None
        else:
            # MFO
            phase_mask, phase_mask_clone = torch.tensor([]), torch.tensor([])

    else:
        raise NotImplementedError

    return phase_mask, phase_mask_clone

class OpticalClassifier(nn.Module):
    def __init__(self, input_type="intensity_only",
                 optics_param=None, number_of_classes=2, maskquery_batchsize=None, mask_num_partitions=None, use_pbr_as_optical_weight=False, exp_param=None,
                 actor=None, shift=30, crop_size=40, optimizer_type='mfo',
                 pg_type='loo', optics_param_dummy=None,
                 ):
        """
        maskquery_batchsize: when use mfo, maskquery_batchsize is not None
        """
        super(OpticalClassifier, self).__init__()

        del_intermediate_var = True if optimizer_type == 'mfo' else False

        self.optics_sim = OpticsSim(
            optics_param, input_type, del_intermediate_var)

        if optimizer_type == 'hbt' or optimizer_type == 'sbt':
            self.optics_sim_dummy = OpticsSim(
                optics_param_dummy, input_type, del_intermediate_var)

        # get init mask
        self.phase_mask, self.phase_mask_clone = get_init_phase_mask(mask_num_partitions, optics_param['optical_computing_layer']['effective_shape'], optics_param[
                                                                     'optical_computing_layer']['mask_representation'], optics_param['optical_computing_layer']['mask_init_type'], actor=actor)

        self.number_of_classes = number_of_classes
        self.maskquery_batchsize = maskquery_batchsize
        self.use_pbr_as_optical_weight = use_pbr_as_optical_weight
        self.shift = shift
        self.crop_size = crop_size
        self.effective_shape = optics_param['optical_computing_layer']['effective_shape']
        self.optimizer_type = optimizer_type

    def forward_sim(self, obj, phase_mask, if_test):
        time_batch = {}
        if not if_test:
            if self.optimizer_type == 'mfo':   # does not store grad if not using ideal optimization for masks
                maskquery_batchsize = phase_mask.shape[1]
                with torch.no_grad():
                    # use the simulated optics
                    if maskquery_batchsize <= 36:
                        img = self.optics_sim(obj, phase_mask)
                    else: # if maskquery_batchsize > 48, gpu memeory will be full, so we calculate all mask outputs in two steps
                        for iter_idx in range(int(maskquery_batchsize / 36 + 1)):
                            if iter_idx == 0:
                                img = self.optics_sim(obj, phase_mask[:, :36, ...])
                            elif iter_idx == int(maskquery_batchsize / 36):
                                img_iter = self.optics_sim(obj, phase_mask[:, int(iter_idx*36):, ...])
                                img = torch.cat((img, img_iter), dim=1)
                            else:
                                img_iter = self.optics_sim(obj, phase_mask[:, int(iter_idx*36): int(iter_idx*36+36), ...])
                                img = torch.cat((img, img_iter), dim=1)
                    torch.cuda.empty_cache()
                    weights = self.get_weights(img)


            elif self.optimizer_type == 'sbt':
                img = self.optics_sim_dummy(obj, phase_mask)
                weights = self.get_weights(img)


            elif self.optimizer_type == 'hbt':
                with torch.no_grad():
                    img = self.optics_sim(
                        obj, phase_mask)
                    img_weights = self.get_weights(img)
                img_dummy = self.optics_sim_dummy(obj, phase_mask)

                img_dummy_weights = self.get_weights(img_dummy)
                weights = (img_weights-img_dummy_weights).detach() + img_dummy_weights

        else:
            img = self.optics_sim(obj, phase_mask)
            weights = self.get_weights(img, True)

        img_out = img.clone()
        return weights, img_out

    def get_weights(self, img, if_test=False):
        if self.optimizer_type != 'mfo' or if_test:
            # x is in shape [batch_size, 1, height, width], when get_optical_weight_for_classifier along dim (-3, -2, -1)
            dim = (-3, -2, -1)
        else:
            # x is in shape [batch_size, maskquery_batchsize, height, width], when get_optical_weight_for_classifier along dim (-2, -1)
            dim = (-2, -1)

        logit_weights = get_optical_weight_for_classfier(img, num_classes=self.number_of_classes, dim=dim,
                                                         use_pbr_as_optical_weight=self.use_pbr_as_optical_weight, shift=self.shift, crop_size=self.crop_size,
                                                         )
        return logit_weights


    def forward(self, obj, exogenous_phase_mask=None, if_test=False):
        if exogenous_phase_mask is None:
            phase_mask = self.phase_mask
        else:
            if exogenous_phase_mask.dim() == 2:
                phase_mask = exogenous_phase_mask[None, None, ...]

        phase_mask = F.interpolate(phase_mask, size=[int(
            self.effective_shape[i]) for i in range(2)])
        phase_mask = pad_image(phase_mask, target_shape=self.effective_shape)

        self.interp_phase_mask = phase_mask  # copy to class

        weights, cam_img = self.forward_sim(obj, phase_mask, if_test)

        del phase_mask
        return weights, cam_img