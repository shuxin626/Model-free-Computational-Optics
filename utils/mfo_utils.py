import torch


def _center_function(population_size):
    centers = torch.arange(population_size).float()
    centers = centers / (population_size - 1)
    centers -= 0.5
    return centers


def _compute_ranks(rewards):
    ranks = torch.empty(rewards.shape)
    ranks[rewards.argsort()] = torch.arange(rewards.shape[-1]).float()
    return ranks


def reward_reshaping(rewards):
    """Rank-shape rewards for vanilla policy-gradient updates."""
    ranks = _compute_ranks(rewards.detach().cpu())
    values = _center_function(rewards.shape[-1])
    return values[ranks.type(torch.long)].to(rewards.device)


def rank_rewards(rewards):
    sorted_inds = torch.argsort(rewards, descending=True)
    sorted_batch_rewards = rewards[sorted_inds]
    return sorted_batch_rewards, sorted_inds[0]


def rank_and_get_the_best(batch_rewards, batch_images, batch_masks):
    _, top_ind = rank_rewards(batch_rewards)
    return batch_rewards[top_ind], batch_images[top_ind], batch_masks[top_ind]
