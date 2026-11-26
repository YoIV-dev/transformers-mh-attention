"""Unit tests for scaled dot-product attention and MultiheadAttention."""

from __future__ import annotations

import pytest
import torch

from transformers_mh_attention.attention import MultiheadAttention, scaled_dot_product


def test_scaled_dot_product_shapes_and_probabilities() -> None:
    torch.manual_seed(42)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 10)

    values, attention = scaled_dot_product(q, k, v)

    assert values.shape == (2, 4, 5, 10)
    assert attention.shape == (2, 4, 5, 6)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6)
    assert (attention >= 0).all()


def test_scaled_dot_product_respects_mask() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    # Causal mask: position i can only attend to <= i
    mask = torch.tril(torch.ones(1, 3, 3))

    _, attention = scaled_dot_product(q, k, v, mask=mask)

    upper_tri = torch.triu(attention[0], diagonal=1)
    assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6)


def test_scaled_dot_product_mask_float16_safe() -> None:
    q = torch.randn(2, 4, 8, dtype=torch.float16)
    k = torch.randn(2, 4, 8, dtype=torch.float16)
    v = torch.randn(2, 4, 8, dtype=torch.float16)
    mask = torch.tril(torch.ones(1, 4, 4))

    values, attention = scaled_dot_product(q, k, v, mask=mask)

    assert not torch.isnan(values).any()
    assert not torch.isnan(attention).any()


def test_multihead_attention_same_dim() -> None:
    torch.manual_seed(42)
    mha = MultiheadAttention(input_dim=16, embed_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16)

    out = mha(x)
    assert out.shape == (2, 5, 16)

    out_with_attn, attn = mha(x, return_attention=True)
    assert torch.allclose(out, out_with_attn)
    assert attn.shape == (2, 4, 5, 5)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6)


def test_multihead_attention_different_input_and_embed_dim() -> None:
    """Regression test: input_dim != embed_dim failed in the original notebook."""
    mha = MultiheadAttention(input_dim=12, embed_dim=24, num_heads=4)
    x = torch.randn(3, 7, 12)

    out, attn = mha(x, return_attention=True)
    assert out.shape == (3, 7, 24)
    assert attn.shape == (3, 4, 7, 7)


@pytest.mark.parametrize(
    ("input_dim", "embed_dim", "num_heads"),
    [
        (16, 15, 4),
        (0, 16, 4),
        (16, 0, 4),
        (16, 16, 0),
    ],
)
def test_multihead_attention_invalid_args(input_dim: int, embed_dim: int, num_heads: int) -> None:
    with pytest.raises(ValueError):
        MultiheadAttention(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads)
