"""Smoke tests for training pipelines and feature extraction."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers_mh_attention.datasets import SetAnomalyDataset
from transformers_mh_attention.training import (
    AnomalyTrainConfig,
    ReverseTrainConfig,
    extract_features,
    train_anomaly,
    train_reverse,
)


def test_train_reverse_smoke(tmp_path: Path) -> None:
    cfg = ReverseTrainConfig(
        num_categories=6,
        seq_len=6,
        train_size=32,
        val_size=16,
        test_size=16,
        batch_size=16,
        model_dim=16,
        num_heads=2,
        num_layers=1,
        max_epochs=1,
        warmup=1,
        checkpoint_dir=tmp_path,
    )
    model, metrics = train_reverse(config=cfg, accelerator="cpu")

    assert model is not None
    assert 0.0 <= metrics["val_acc"] <= 1.0
    assert 0.0 <= metrics["test_acc"] <= 1.0


def test_train_anomaly_smoke(tmp_path: Path) -> None:
    num_classes = 4
    per_class = 8
    feats = torch.randn(num_classes * per_class, 16)
    labels = torch.arange(num_classes).repeat_interleave(per_class)

    ds_train = SetAnomalyDataset(feats, labels, set_size=4, train=True, seed=1)
    ds_eval = SetAnomalyDataset(feats, labels, set_size=4, train=False, seed=2)

    train_loader = DataLoader(ds_train, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_eval, batch_size=8)
    test_loader = DataLoader(ds_eval, batch_size=8)

    cfg = AnomalyTrainConfig(
        model_dim=16,
        num_heads=2,
        num_layers=1,
        max_epochs=1,
        warmup=1,
        checkpoint_dir=tmp_path,
    )
    model, metrics = train_anomaly(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=16,
        config=cfg,
        accelerator="cpu",
    )

    assert model is not None
    assert 0.0 <= metrics["train_acc"] <= 1.0
    assert 0.0 <= metrics["val_acc"] <= 1.0
    assert 0.0 <= metrics["test_acc"] <= 1.0


def test_extract_features_caches_to_disk(tmp_path: Path) -> None:
    images = torch.randn(10, 3, 8, 8)
    targets = torch.zeros(10, dtype=torch.long)
    dataset = TensorDataset(images, targets)

    extractor = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 12))
    save_file = tmp_path / "feats.pt"

    feats1 = extract_features(
        dataset=dataset,
        feature_extractor=extractor,
        save_file=save_file,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert save_file.is_file()
    assert feats1.shape == (10, 12)

    # Second call should load cached tensor directly
    feats2 = extract_features(
        dataset=dataset,
        feature_extractor=extractor,
        save_file=save_file,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert torch.equal(feats1, feats2)
