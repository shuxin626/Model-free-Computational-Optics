from torch.optim.optimizer import Optimizer
import torch
import math

class MFOOptimizer(Optimizer):
    def __init__(self, params, actor, actor_param):
        defaults = dict(actor=actor)
        super(MFOOptimizer, self).__init__(params, defaults)
        self.actor = actor
        self.actor_param = actor_param
        for groups in self.param_groups:
            slm_mask = groups['params'][0]
            slm_mask_clone = groups['params'][1]
            sampled_masks, sampled_masks_clone = self.sample()
            slm_mask.data = sampled_masks
            slm_mask_clone.data = sampled_masks_clone

    def __setstate__(self, state):
        super(MFOOptimizer, self).__setstate__(state)

    def output_normalize(self, x):
        x = (torch.sigmoid(x).detach() - 0.5) * 2
        x = x * math.pi
        return x
    
    def sampled_mask_preprocessing(self, sampled_masks):
            if self.actor_param['output_normalize_flag']:
                sampled_masks = self.output_normalize(sampled_masks[:, None, :, :])
            else:
                sampled_masks = sampled_masks[:, None, :, :] * math.pi
            return sampled_masks.clone().permute(1, 0, 2, 3), sampled_masks
    
    def sample(self):
        sampled_masks = self.actor.sample()
        sampled_masks, sampled_masks_clone = self.sampled_mask_preprocessing(sampled_masks)
        return sampled_masks, sampled_masks_clone

    def step(self, rewards):
        for groups in self.param_groups:
            topest_ind = int(torch.argmax(rewards))

            slm_mask = groups['params'][0]
            slm_mask_clone = groups['params'][1]
            # - slm_mask.data [1, qb, h, w]
            # - cam_image [b, qb, h,w]
            # - rewards [qb]
            pg_loss = self.actor.optim(rewards, slm_mask_clone)
            sampled_masks, sampled_masks_clone = self.sample()
            slm_mask.data = sampled_masks
            slm_mask_clone.data = sampled_masks_clone

        return topest_ind, pg_loss