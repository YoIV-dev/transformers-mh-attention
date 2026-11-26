"""Transformer encoder blocks and sinusoidal positional encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from transformers_mh_attention.attention import MultiheadAttention


class EncoderBlock(nn.Module):
    """Single Transformer encoder block with post-LayerNorm residual connections.

    Args:
        input_dim: Dimensionality of the input features.
        num_heads: Number of heads to use in the attention block.
        dim_feedforward: Dimensionality of the hidden layer in the MLP.
        dropout: Dropout probability to use in the dropout layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Normalization and dropout layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply self-attention and feed-forward sublayers."""
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out: Tensor = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """Stack of :class:`EncoderBlock` layers.

    Args:
        num_layers: Number of stacked encoder blocks.
        input_dim: Dimensionality of the input features.
        num_heads: Number of attention heads per block.
        dim_feedforward: Hidden dimensionality of the feed-forward network.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Pass input sequentially through all encoder layers."""
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x: Tensor, mask: Tensor | None = None) -> list[Tensor]:
        """Extract attention weight matrices from each encoder layer."""
        attention_maps: list[Tensor] = []
        for layer in self.layers:
            assert isinstance(layer, EncoderBlock)
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x, mask=mask)
        return attention_maps


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as introduced in *Attention Is All You Need*.

    Args:
        d_model: Hidden dimensionality of the input.
        max_len: Maximum supported sequence length.
    """

    pe: Tensor

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model and max_len must be positive integers.")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # persistent=False avoids saving deterministic sinusoidal encodings in checkpoints
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings to input tensor ``x`` of shape ``(batch, seq_len, d_model)``."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_len ({self.pe.size(1)})."
            )
        return x + self.pe[:, :seq_len]
