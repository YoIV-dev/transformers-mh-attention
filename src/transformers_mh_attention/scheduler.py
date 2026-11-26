"""Learning rate schedulers for Transformer training."""

from __future__ import annotations

import math

import torch.optim as optim


class CosineWarmupScheduler(optim.lr_scheduler.LRScheduler):
    """Cosine annealing learning rate scheduler with linear warm-up.

    Args:
        optimizer: Wrapped PyTorch optimizer.
        warmup: Number of linear warm-up steps (``>= 0``).
        max_iters: Total number of training steps (``> 0``).
        last_epoch: Index of the last step. Defaults to ``-1``.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup: int,
        max_iters: int,
        last_epoch: int = -1,
    ) -> None:
        if warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {warmup}.")
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {max_iters}.")
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rates for each parameter group at the current step."""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """Compute the multiplicative learning rate factor for a given step."""
        lr_factor = 0.5 * (1.0 + math.cos(math.pi * epoch / self.max_num_iters))
        if self.warmup > 0 and epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor
