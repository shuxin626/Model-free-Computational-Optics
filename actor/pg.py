import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import math
from torch.distributions import Normal
from config import *

class PG(nn.Module):
    """Policy gradient method for generating the next batch"""

    def __init__(
        self,
        mask_shape=[32, 32],
        pg_lr=1500.0,  # 0.1 for phase mode
        use_scheduler=False,
        dp_std=0.05,
        query_batchsize=20,
        optimizer_type = 'sgd',
        output_normalize_flag=True,
    ):
        super(PG, self).__init__()
        self.mask_shape = mask_shape
        self.use_scheduler = use_scheduler
        self.query_batchsize = query_batchsize
        self.output_normalize_flag = output_normalize_flag
        self.pg_lr = pg_lr
        self.optimizer_type = optimizer_type

        self.std = dp_std # in the context of determinisic policy, the noise level is not learnable and mainly contributes to exploration
        self.mu = torch.rand(mask_shape, device=device)
        self.mu = nn.Parameter(self.mu)
        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD([self.mu], lr=pg_lr)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam([self.mu], lr=pg_lr)
        else:
            raise Exception('Optimizer not found')


        if use_scheduler:
            self.scheduler = StepLR(self.policy.optimizer, step_size=100, gamma=0.5)

    def get_distri(self,):
        mask_distri = Normal(self.mu, self.std)
        return mask_distri


    def sample(self):
        sample_mask_distri = self.get_distri()
        x_t = sample_mask_distri.rsample([self.query_batchsize])
        x_t = x_t.detach().to(device)

        return x_t

    def update_mu(self, masks, rewards):
        """update the policy w/ or w/o the learned forward scattering model."""
        assert len(rewards.shape) == 1, 'error, rewards should be dim 1'

        mask_distri = self.get_distri()
        sum_reward = torch.sum(rewards, dim=0, keepdim=True)
        baseline = (sum_reward - rewards) / (masks.shape[0] - 1)

        dim = tuple(range(int(-1 * masks.dim() + 1), 0))
        logp_pi = torch.mean(mask_distri.log_prob(masks), dim)

        loss = -torch.mean(
            (rewards - baseline)/torch.std(rewards, unbiased=True)
            * logp_pi
        ) 

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        if self.use_scheduler:
            self.scheduler.step()

        return loss.clone().detach()

    def optim(self, batch_rewards, batch_mask_clone):
        # update policy only with res from current batch
        pg_loss = self.update_mu(batch_mask_clone, batch_rewards)
        return pg_loss