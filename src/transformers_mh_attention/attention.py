"""Scaled dot-product and Multi-Head Attention modules."""

from __future__ import annotations

import math
from typing import Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape ``(..., seq_len_q, d_k)``.
        k: Key tensor of shape ``(..., seq_len_k, d_k)``.
        v: Value tensor of shape ``(..., seq_len_k, d_v)``.
        mask: Optional mask tensor broadcastable to ``(..., seq_len_q, seq_len_k)`` where
            ``0`` (or ``False``) indicates masked-out positions.

    Returns:
        A tuple ``(values, attention)`` where ``values`` has shape ``(..., seq_len_q, d_v)``
        and ``attention`` has shape ``(..., seq_len_q, seq_len_k)``.
    """
    d_k = q.size(-1)
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask_value = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits.masked_fill(mask == 0, mask_value)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    """Multi-Head Self-Attention layer.

    Args:
        input_dim: Dimensionality of the input features.
        embed_dim: Total embedding dimensionality across all heads.
        num_heads: Number of parallel attention heads.

    Raises:
        ValueError: If ``embed_dim`` is not divisible by ``num_heads`` or arguments are
            non-positive.
    """

    def __init__(self, input_dim: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if input_dim <= 0 or embed_dim <= 0 or num_heads <= 0:
            raise ValueError("input_dim, embed_dim, and num_heads must be positive integers.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack Q, K, V projections together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weights with Xavier uniform and biases with zeros."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    @overload
    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        return_attention: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        *,
        return_attention: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        mask: Tensor | None,
        return_attention: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply multi-head self-attention to ``x``.

        Args:
            x: Input tensor of shape ``(batch_size, seq_length, input_dim)``.
            mask: Optional attention mask broadcastable to
                ``(batch_size, num_heads, seq_length, seq_length)``.
            return_attention: If ``True``, also return the attention weight matrix.

        Returns:
            Output tensor of shape ``(batch_size, seq_length, embed_dim)``, or a tuple of
            ``(output, attention)`` when ``return_attention=True``.
        """
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, 3 * HeadDim]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, HeadDim]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        output: Tensor = self.o_proj(values)

        if return_attention:
            return output, attention
        return output
