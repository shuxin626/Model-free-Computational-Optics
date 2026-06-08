import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gpu_device_config import device
from utils.tensor_utils import circular_pad, normalize, total_variation


class RewardFn(nn.Module):
    def __init__(self, metric, tv_weight=0.0, inspect_rewards=False, target=None):
        super().__init__()
        self.metric = metric
        self.target = target
        self.mse_loss = nn.MSELoss(reduction="none")
        self.tv_weight = tv_weight
        self.inspect_rewards = inspect_rewards

    def forward(self, x, mean_logits=None):
        if self.metric != "mse":
            raise NotImplementedError

        target = self.target.to(x.device)
        if target.shape[0] == 1 and x.shape[0] != 1:
            target = target.expand_as(x)
        task_rewards = -torch.log(torch.mean(self.mse_loss(normalize(x), target), dim=(1, 2, 3)) + 1e-12)

        if mean_logits is not None:
            mean_logits = mean_logits[None, None, :, :]
            tv_reg = -self.tv_weight * total_variation(mean_logits) / torch.numel(mean_logits)
            rewards = task_rewards + tv_reg[0]
        else:
            rewards = task_rewards

        if self.inspect_rewards:
            print("task_rewards", task_rewards)
        return rewards


def load_target(dataset_name, in_size, sample_idx=1):
    """Load one image target for the CGH task."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "mnist":
        trainset = datasets.MNIST("data/mnist_train", train=True, download=True, transform=transform)
    elif dataset_name == "fmnist":
        trainset = datasets.FashionMNIST("data/fmnist_train", train=True, download=True, transform=transform)
    elif dataset_name == "cifar":
        trainset = datasets.CIFAR10(root="data/cifar_10", train=True, download=True, transform=transform)
    else:
        raise NotImplementedError("Unsupported CGH target dataset: {}".format(dataset_name))

    batch_size = max(sample_idx + 1, 2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    inputs, _ = next(iter(trainloader))

    if inputs.shape[-1] < 32:
        inputs = circular_pad(inputs, pad_scale=32 / inputs.shape[-1])
    inputs = F.interpolate(inputs, size=[in_size, in_size])
    return inputs[sample_idx : sample_idx + 1, :1, :, :].to(device)
