from dataclasses import dataclass

import torch

from utils.gpu_device_config import device


@dataclass
class ClassificationStats:
    loss_sum: float = 0.0
    correct: int = 0
    total: int = 0

    def update(self, outputs, targets, loss=None):
        if loss is not None:
            self.loss_sum += float(loss.item())
        _, predicted = outputs.max(1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets).sum().item()
        return predicted

    @property
    def accuracy(self):
        return 100.0 * self.correct / self.total if self.total else 0.0

    @property
    def mean_loss_per_sample(self):
        return self.loss_sum / self.total if self.total else 0.0


def move_batch_to_device(inputs, targets):
    return inputs.float().to(device), targets.long().to(device)


def select_input_channel(inputs, in_ch):
    if isinstance(in_ch, int):
        return inputs[:, in_ch : in_ch + 1, ...]
    return inputs[:, in_ch, ...]


def prepare_classification_batch(inputs, targets, in_ch):
    inputs, targets = move_batch_to_device(inputs, targets)
    return select_input_channel(inputs, in_ch), targets


def evaluate_classification_loader(dataloader, in_ch, criterion, forward_batch, log_interval=None):
    stats = ClassificationStats()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = prepare_classification_batch(inputs, targets, in_ch)
            outputs = forward_batch(inputs)
            loss = criterion(outputs, targets)
            stats.update(outputs, targets, loss)

            if log_interval is not None and (batch_idx + 1) % log_interval == 0:
                print(
                    "[{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}".format(
                        (batch_idx + 1) * len(inputs),
                        len(dataloader.dataset),
                        100.0 * (batch_idx + 1) / len(dataloader),
                        loss.item(),
                    )
                )

    return stats
