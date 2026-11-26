"""PyTorch Lightning modules for Transformer-based sequence and set prediction."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor

from transformers_mh_attention.encoder import PositionalEncoding, TransformerEncoder
from transformers_mh_attention.scheduler import CosineWarmupScheduler


class TransformerPredictor(pl.LightningModule):
    """Base LightningModule wrapping a Transformer encoder for per-element prediction.

    Args:
        input_dim: Hidden dimensionality of the input features.
        model_dim: Hidden dimensionality inside the Transformer.
        num_classes: Number of classes to predict per sequence/set element.
        num_heads: Number of heads in Multi-Head Attention blocks.
        num_layers: Number of stacked encoder blocks.
        lr: Peak learning rate for Adam optimizer.
        warmup: Number of warm-up iterations for :class:`CosineWarmupScheduler`.
        max_iters: Total number of training iterations.
        dropout: Dropout probability inside Transformer and classifier head.
        input_dropout: Dropout probability applied to raw input features.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        lr: float,
        warmup: int,
        max_iters: int,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.lr = lr
        self.warmup = warmup
        self.max_iters = max_iters
        self.dropout = dropout
        self.input_dropout = input_dropout

        self.save_hyperparameters()
        self._create_model()

    def _create_model(self) -> None:
        """Build input projection, positional encoding, Transformer encoder, and classifier."""
        self.input_net = nn.Sequential(
            nn.Dropout(self.input_dropout),
            nn.Linear(self.input_dim, self.model_dim),
        )
        self.positional_encoding = PositionalEncoding(d_model=self.model_dim)
        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=self.model_dim,
            dim_feedforward=2 * self.model_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        self.output_net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.num_classes),
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        add_positional_encoding: bool = True,
    ) -> Tensor:
        """Run forward pass through the Transformer predictor.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, input_dim)``.
            mask: Optional attention mask.
            add_positional_encoding: Whether to add sinusoidal positional encodings.

        Returns:
            Logits tensor of shape ``(batch_size, seq_len, num_classes)``.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        output: Tensor = self.output_net(x)
        return output

    @torch.no_grad()
    def get_attention_maps(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        add_positional_encoding: bool = True,
    ) -> list[Tensor]:
        """Extract attention matrices for all Transformer layers on a single batch."""
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        return self.transformer.get_attention_maps(x, mask=mask)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure Adam optimizer with per-step :class:`CosineWarmupScheduler`."""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.warmup,
            max_iters=self.max_iters,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }


class ReversePredictor(TransformerPredictor):
    """Transformer predictor for the sequence-reversal task."""

    def _calculate_loss(
        self,
        batch: tuple[Tensor, Tensor],
        mode: str = "train",
    ) -> tuple[Tensor, Tensor]:
        inp_data, labels = batch
        inp_one_hot = F.one_hot(inp_data, num_classes=self.num_classes).float()

        preds = self.forward(inp_one_hot, add_positional_encoding=True)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._calculate_loss(batch, mode="test")


class AnomalyPredictor(TransformerPredictor):
    """Permutation-equivariant Transformer predictor for set anomaly detection."""

    def _calculate_loss(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        mode: str = "train",
    ) -> tuple[Tensor, Tensor]:
        img_sets, _, labels = batch
        # No positional encoding because the input is an unordered set
        preds = self.forward(img_sets, add_positional_encoding=False)
        preds = preds.squeeze(dim=-1)  # [batch_size, set_size]
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True)
        return loss, acc

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        self._calculate_loss(batch, mode="test")
