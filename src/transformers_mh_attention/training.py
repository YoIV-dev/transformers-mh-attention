"""Training routines, configuration dataclasses, and feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights
from tqdm import tqdm

from transformers_mh_attention.datasets import ReverseDataset
from transformers_mh_attention.models import AnomalyPredictor, ReversePredictor


@dataclass(frozen=True)
class ReverseTrainConfig:
    """Hyperparameters and paths for training :class:`ReversePredictor`."""

    num_categories: int = 10
    seq_len: int = 16
    train_size: int = 50_000
    val_size: int = 1_000
    test_size: int = 10_000
    batch_size: int = 128
    model_dim: int = 32
    num_heads: int = 1
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 5e-4
    warmup: int = 50
    max_epochs: int = 10
    gradient_clip_val: float = 5.0
    seed: int = 42
    checkpoint_dir: Path = Path("saved_models/Transformers")


@dataclass(frozen=True)
class AnomalyTrainConfig:
    """Hyperparameters and paths for training :class:`AnomalyPredictor`."""

    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    input_dropout: float = 0.1
    lr: float = 5e-4
    warmup: int = 100
    max_epochs: int = 100
    gradient_clip_val: float = 2.0
    seed: int = 42
    checkpoint_dir: Path = Path("saved_models/Transformers")


def get_device() -> torch.device:
    """Return CUDA, MPS, or CPU device based on local hardware availability."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet34_feature_extractor(device: torch.device | None = None) -> nn.Module:
    """Load a frozen pretrained ResNet-34 backbone with its classification head removed."""
    target_device = device or get_device()
    model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Identity()
    backbone: nn.Module = model.to(target_device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


@torch.no_grad()
def extract_features(
    dataset: Dataset[Any],
    feature_extractor: nn.Module,
    save_file: Path,
    batch_size: int = 128,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> Tensor:
    """Extract and cache image features using ``feature_extractor``."""
    if save_file.is_file():
        loaded: Tensor = torch.load(save_file, weights_only=True)
        return loaded

    target_device = device or get_device()
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    feature_batches: list[Tensor] = []
    for imgs, _ in tqdm(loader, desc="Extracting features"):
        feats: Tensor = feature_extractor(imgs.to(target_device))
        feature_batches.append(feats.detach().cpu())

    extracted = torch.cat(feature_batches, dim=0)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extracted, save_file)
    return extracted


def train_reverse(
    config: ReverseTrainConfig | None = None,
    accelerator: str = "auto",
) -> tuple[ReversePredictor, dict[str, float]]:
    """Train (or load) and evaluate :class:`ReversePredictor`."""
    cfg = config or ReverseTrainConfig()
    pl.seed_everything(cfg.seed)

    train_dataset = ReverseDataset(cfg.num_categories, cfg.seq_len, cfg.train_size, seed=cfg.seed)
    val_dataset = ReverseDataset(cfg.num_categories, cfg.seq_len, cfg.val_size, seed=cfg.seed + 1)
    test_dataset = ReverseDataset(cfg.num_categories, cfg.seq_len, cfg.test_size, seed=cfg.seed + 2)

    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
    )

    root_dir = cfg.checkpoint_dir / "ReverseTask"
    root_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")
    trainer = pl.Trainer(
        default_root_dir=str(root_dir),
        callbacks=[checkpoint_cb],
        accelerator=accelerator,
        devices=1,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
    )

    pretrained_path = cfg.checkpoint_dir / "ReverseTask.ckpt"
    if pretrained_path.is_file():
        model = ReversePredictor.load_from_checkpoint(pretrained_path)
    else:
        max_iters = max(1, cfg.max_epochs * len(train_loader))
        model = ReversePredictor(
            input_dim=cfg.num_categories,
            model_dim=cfg.model_dim,
            num_classes=cfg.num_categories,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            lr=cfg.lr,
            warmup=cfg.warmup,
            max_iters=max_iters,
            dropout=cfg.dropout,
        )
        trainer.fit(model, train_loader, val_loader)

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    metrics = {
        "val_acc": float(val_result[0]["test_acc"]),
        "test_acc": float(test_result[0]["test_acc"]),
    }
    return model, metrics


def train_anomaly(
    train_loader: DataLoader[tuple[Tensor, Tensor, int]],
    val_loader: DataLoader[tuple[Tensor, Tensor, int]],
    test_loader: DataLoader[tuple[Tensor, Tensor, int]],
    input_dim: int,
    config: AnomalyTrainConfig | None = None,
    accelerator: str = "auto",
) -> tuple[AnomalyPredictor, dict[str, float]]:
    """Train (or load) and evaluate :class:`AnomalyPredictor`."""
    cfg = config or AnomalyTrainConfig()
    pl.seed_everything(cfg.seed)

    root_dir = cfg.checkpoint_dir / "SetAnomalyTask"
    root_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")
    trainer = pl.Trainer(
        default_root_dir=str(root_dir),
        callbacks=[checkpoint_cb],
        accelerator=accelerator,
        devices=1,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
    )

    pretrained_path = cfg.checkpoint_dir / "SetAnomalyTask.ckpt"
    if pretrained_path.is_file():
        model = AnomalyPredictor.load_from_checkpoint(pretrained_path)
    else:
        max_iters = max(1, cfg.max_epochs * len(train_loader))
        model = AnomalyPredictor(
            input_dim=input_dim,
            model_dim=cfg.model_dim,
            num_classes=1,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            lr=cfg.lr,
            warmup=cfg.warmup,
            max_iters=max_iters,
            dropout=cfg.dropout,
            input_dropout=cfg.input_dropout,
        )
        trainer.fit(model, train_loader, val_loader)
        if checkpoint_cb.best_model_path:
            model = AnomalyPredictor.load_from_checkpoint(checkpoint_cb.best_model_path)

    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    metrics = {
        "train_acc": float(train_result[0]["test_acc"]),
        "val_acc": float(val_result[0]["test_acc"]),
        "test_acc": float(test_result[0]["test_acc"]),
    }
    return model, metrics
