"""Naive model-free optimizer for the CGH task."""

from utils.gpu_device_config import device
from utils.mfo_utils import rank_and_get_the_best
from utils.tensor_utils import normalize
import math
import torch


class MFO(object):
    """Controller for vanilla policy-gradient CGH optimization."""

    def __init__(
        self,
        macro_itr,
        slm_type,
        env,
        actor,
        reward_fn,
        query_batchsize,
        pg_type,
        output_normalize_flag=True,
        show_result=True,
        show_interval=100,
    ):
        if pg_type != 'loo':
            raise NotImplementedError("This repo only includes the naive 'loo' CGH optimizer.")

        self.macro_itr = macro_itr
        self.slm_type = slm_type
        self.env = env
        self.actor = actor
        self.reward_fn = reward_fn
        self.query_batchsize = query_batchsize
        self.pg_type = pg_type
        self.output_normalize_flag = output_normalize_flag
        self.show_result = show_result
        self.show_interval = show_interval

    def phase_from_actor_sample(self, raw_masks):
        if self.slm_type != 'phase':
            raise NotImplementedError("Only phase-only CGH is supported here.")
        if self.output_normalize_flag:
            phase_masks = (torch.sigmoid(raw_masks).detach() - 0.5) * 2 * math.pi
        else:
            phase_masks = raw_masks.detach() * math.pi
        return phase_masks[:, None, :, :]

    def vis(self, itr_buffer, best_reward_buffer, best_img, best_mask):
        from utils.visualize_utils import plot_loss, show

        print("best reward at itr {} is {}".format(itr_buffer[-1], best_reward_buffer[-1]))
        plot_loss(itr_buffer, best_reward_buffer, filename='cgh_reward')
        show(best_img[0].detach().cpu(), 'best CGH image at itr {}'.format(itr_buffer[-1]))
        show(best_mask[0].detach().cpu(), 'best phase mask at itr {}'.format(itr_buffer[-1]))

    def single_itr_update(self):
        raw_masks = self.actor.sample()
        phase_masks = self.phase_from_actor_sample(raw_masks)

        with torch.no_grad():
            batch_images = self.env.step(phase_masks)
            batch_images = normalize(batch_images.float().to(device))
            batch_rewards = self.reward_fn(batch_images).to(device)

        _ = self.actor.optim(batch_rewards, raw_masks)
        return batch_rewards, batch_images, phase_masks

    def optimize(self):
        itr_buffer = []
        best_reward_buffer = []
        best_reward = -math.inf
        best_img = None
        best_mask = None

        for itr in range(self.macro_itr):
            batch_rewards, batch_images, batch_masks = self.single_itr_update()
            best_reward_itr, best_img_itr, best_mask_itr = rank_and_get_the_best(
                batch_rewards, batch_images, batch_masks)

            if best_reward_itr > best_reward:
                best_reward = best_reward_itr
                best_img = best_img_itr
                best_mask = best_mask_itr

            itr_buffer.append(itr)
            best_reward_buffer.append(best_reward.item())

            if self.show_result and (itr + 1) % self.show_interval == 0:
                self.vis(itr_buffer, best_reward_buffer, best_img, best_mask)

        return {
            'itr': itr_buffer,
            'best_reward': best_reward_buffer,
            'best_img': best_img,
            'best_mask': best_mask,
        }
