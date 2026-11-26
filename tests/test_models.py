"""Unit tests for ReversePredictor and AnomalyPredictor LightningModules."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from transformers_mh_attention.models import AnomalyPredictor, ReversePredictor


def test_reverse_predictor_forward_and_attention_maps() -> None:
    model = ReversePredictor(
        input_dim=10,
        model_dim=16,
        num_classes=10,
        num_heads=2,
        num_layers=2,
        lr=1e-3,
        warmup=5,
        max_iters=50,
    )
    x = F.one_hot(torch.randint(0, 10, (4, 8)), num_classes=10).float()

    logits = model(x, add_positional_encoding=True)
    assert logits.shape == (4, 8, 10)

    attn_maps = model.get_attention_maps(x, add_positional_encoding=True)
    assert len(attn_maps) == 2
    assert attn_maps[0].shape == (4, 2, 8, 8)


def test_anomaly_predictor_is_permutation_equivariant() -> None:
    """Verify key property from UvA-DL Tutorial 6: without PE, predictions are equivariant."""
    torch.manual_seed(42)
    model = AnomalyPredictor(
        input_dim=16,
        model_dim=32,
        num_classes=1,
        num_heads=4,
        num_layers=2,
        lr=5e-4,
        warmup=10,
        max_iters=100,
        dropout=0.0,
        input_dropout=0.0,
    )
    model.eval()

    x = torch.randn(3, 10, 16)
    perm = torch.randperm(10)

    with torch.no_grad():
        preds = F.softmax(model(x, add_positional_encoding=False).squeeze(-1), dim=-1)
        perm_preds = F.softmax(
            model(x[:, perm], add_positional_encoding=False).squeeze(-1), dim=-1
        )

    assert torch.allclose(preds[:, perm], perm_preds, atol=1e-5)
