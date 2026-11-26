"""Unit tests for CosineWarmupScheduler."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from transformers_mh_attention.scheduler import CosineWarmupScheduler


def test_cosine_warmup_scheduler_profile() -> None:
    param = nn.Parameter(torch.empty(4, 4))
    optimizer = optim.Adam([param], lr=1e-3)
    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=1000)

    # Starts at 0, increases monotonically during warmup, decays to 0 at max_iters
    assert scheduler.get_lr_factor(0) == pytest.approx(0.0)
    assert scheduler.get_lr_factor(50) > scheduler.get_lr_factor(10)
    assert scheduler.get_lr_factor(100) > scheduler.get_lr_factor(50)
    assert scheduler.get_lr_factor(500) < scheduler.get_lr_factor(100)
    assert scheduler.get_lr_factor(1000) == pytest.approx(0.0, abs=1e-7)


def test_cosine_warmup_scheduler_zero_warmup() -> None:
    param = nn.Parameter(torch.empty(2, 2))
    optimizer = optim.Adam([param], lr=1e-3)
    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=0, max_iters=100)

    assert scheduler.get_lr_factor(0) == pytest.approx(1.0)
    assert scheduler.get_lr_factor(50) == pytest.approx(0.5)
    assert scheduler.get_lr_factor(100) == pytest.approx(0.0, abs=1e-7)


def test_cosine_warmup_scheduler_step_updates_optimizer_lr() -> None:
    param = nn.Parameter(torch.zeros(2, 2))
    optimizer = optim.Adam([param], lr=1e-2)
    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=100)

    lrs: list[float] = []
    for _ in range(15):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] < lrs[9]
    assert lrs[14] < lrs[9]


@pytest.mark.parametrize(("warmup", "max_iters"), [(-1, 100), (10, 0)])
def test_cosine_warmup_scheduler_invalid_args(warmup: int, max_iters: int) -> None:
    param = nn.Parameter(torch.zeros(2, 2))
    optimizer = optim.Adam([param], lr=1e-3)
    with pytest.raises(ValueError):
        CosineWarmupScheduler(optimizer=optimizer, warmup=warmup, max_iters=max_iters)
