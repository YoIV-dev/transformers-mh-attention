"""Unit tests for ReverseDataset, SetAnomalyDataset, and feature splitting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from transformers_mh_attention.datasets import (
    ReverseDataset,
    SetAnomalyDataset,
    split_train_val_features,
)


def test_reverse_dataset_reverses_sequences_deterministically() -> None:
    ds1 = ReverseDataset(num_categories=10, seq_len=8, size=20, seed=123)
    ds2 = ReverseDataset(num_categories=10, seq_len=8, size=20, seed=123)

    assert len(ds1) == 20
    inp1, lbl1 = ds1[0]
    inp2, lbl2 = ds2[0]
    assert torch.equal(inp1, inp2)
    assert torch.equal(lbl1, torch.flip(inp1, dims=(0,)))
    assert inp1.shape == (8,)


def test_set_anomaly_dataset_invariants_and_rng_isolation() -> None:
    num_classes = 5
    per_class = 12
    feats = torch.randn(num_classes * per_class, 16)
    labels = torch.arange(num_classes).repeat_interleave(per_class)

    # Verify global numpy RNG is not mutated by SetAnomalyDataset
    np.random.seed(999)
    expected_global = np.random.random()
    np.random.seed(999)

    eval_ds = SetAnomalyDataset(feats, labels, set_size=6, train=False, seed=42)
    actual_global = np.random.random()
    assert actual_global == expected_global

    assert len(eval_ds) == num_classes * per_class
    img_set, indices, label = eval_ds[0]
    assert img_set.shape == (6, 16)
    assert indices.shape == (6,)
    assert label == 5  # Anomaly placed at last index

    normal_labels = labels[indices[:-1]]
    anomaly_label = labels[indices[-1]]
    # All normal items share one class, distinct from the anomaly's class
    assert (normal_labels == normal_labels[0]).all()
    assert normal_labels[0] != anomaly_label

    # Evaluation mode returns identical set across repeated calls
    img_set_2, indices_2, _ = eval_ds[0]
    assert torch.equal(indices, indices_2)
    assert torch.equal(img_set, img_set_2)


def test_split_train_val_features_preserves_class_balance() -> None:
    num_classes = 4
    per_class = 20
    feats = torch.randn(num_classes * per_class, 8)
    labels = torch.arange(num_classes).repeat_interleave(per_class)

    tr_f, tr_l, val_f, val_l = split_train_val_features(feats, labels, val_fraction=0.2)

    assert tr_f.shape == (64, 8)
    assert val_f.shape == (16, 8)
    for c in range(num_classes):
        assert int((tr_l == c).sum().item()) == 16
        assert int((val_l == c).sum().item()) == 4


def test_split_train_val_features_invalid_fraction() -> None:
    feats = torch.randn(10, 4)
    labels = torch.zeros(10, dtype=torch.long)
    with pytest.raises(ValueError):
        split_train_val_features(feats, labels, val_fraction=1.5)
