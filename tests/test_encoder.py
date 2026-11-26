"""Unit tests for EncoderBlock, TransformerEncoder, and PositionalEncoding."""

from __future__ import annotations

import pytest
import torch

from transformers_mh_attention.encoder import (
    EncoderBlock,
    PositionalEncoding,
    TransformerEncoder,
)


def test_encoder_block_output_shape() -> None:
    block = EncoderBlock(input_dim=32, num_heads=4, dim_feedforward=64, dropout=0.1)
    x = torch.randn(2, 10, 32)
    out = block(x)
    assert out.shape == (2, 10, 32)


def test_transformer_encoder_and_attention_maps_propagate_mask() -> None:
    torch.manual_seed(0)
    encoder = TransformerEncoder(
        num_layers=2,
        input_dim=16,
        num_heads=2,
        dim_feedforward=32,
        dropout=0.0,
    )
    encoder.eval()
    x = torch.randn(2, 6, 16)
    mask = torch.tril(torch.ones(1, 1, 6, 6))

    out = encoder(x, mask=mask)
    assert out.shape == (2, 6, 16)

    attn_maps = encoder.get_attention_maps(x, mask=mask)
    assert len(attn_maps) == 2
    for attn in attn_maps:
        assert attn.shape == (2, 2, 6, 6)
        upper = torch.triu(attn[0, 0], diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)

    # Second layer attention must differ when mask is applied vs unmasked (regression check)
    attn_maps_unmasked = encoder.get_attention_maps(x, mask=None)
    assert not torch.allclose(attn_maps[1], attn_maps_unmasked[1])


def test_transformer_encoder_invalid_num_layers() -> None:
    with pytest.raises(ValueError):
        TransformerEncoder(num_layers=0, input_dim=16, num_heads=2, dim_feedforward=32)


def test_positional_encoding_even_and_odd_d_model() -> None:
    for d_model in (16, 17):
        pe = PositionalEncoding(d_model=d_model, max_len=50)
        x = torch.zeros(2, 12, d_model)
        out = pe(x)
        assert out.shape == (2, 12, d_model)
        # Position 0 has sin(0)=0 at even indices and cos(0)=1 at odd indices
        assert torch.allclose(out[0, 0, 0::2], torch.zeros(len(out[0, 0, 0::2])))
        assert torch.allclose(out[0, 0, 1::2], torch.ones(len(out[0, 0, 1::2])))


def test_positional_encoding_raises_on_sequence_too_long() -> None:
    pe = PositionalEncoding(d_model=8, max_len=10)
    x = torch.zeros(1, 11, 8)
    with pytest.raises(ValueError, match="exceeds max_len"):
        pe(x)
