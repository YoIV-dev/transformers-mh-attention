"""Datasets and data utilities for sequence reversal and set anomaly detection tasks."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ReverseDataset(Dataset[tuple[Tensor, Tensor]]):
    """Synthetic dataset where the target is the reversed input integer sequence.

    Args:
        num_categories: Number of distinct token categories.
        seq_len: Length of each sequence.
        size: Number of sequences in the dataset.
        seed: Optional random seed for deterministic data generation.
    """

    def __init__(
        self,
        num_categories: int,
        seq_len: int,
        size: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if num_categories <= 0 or seq_len <= 0 or size <= 0:
            raise ValueError("num_categories, seq_len, and size must be positive integers.")
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
            self.data = torch.randint(
                self.num_categories, size=(self.size, self.seq_len), generator=generator
            )
        else:
            self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(0,))
        return inp_data, labels


class SetAnomalyDataset(Dataset[tuple[Tensor, Tensor, int]]):
    """Dataset of image feature sets containing ``set_size - 1`` normal items and 1 anomaly.

    Args:
        img_feats: Tensor of shape ``(num_imgs, img_dim)`` with extracted image features.
        labels: Tensor of shape ``(num_imgs,)`` containing class labels (balanced per class).
        set_size: Total number of elements per set (must be ``>= 2``).
        train: If ``True``, sample a new set on every ``__getitem__`` call; otherwise
            pre-generate deterministic test sets.
        seed: Random seed used by the internal NumPy RNG.
    """

    def __init__(
        self,
        img_feats: Tensor,
        labels: Tensor,
        set_size: int = 10,
        train: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if set_size < 2:
            raise ValueError(f"set_size must be at least 2, got {set_size}.")

        self.img_feats = img_feats
        self.labels = labels.long()
        self.set_size = set_size - 1  # Number of normal (non-anomaly) images
        self.train = train
        self._rng = np.random.default_rng(seed)

        self.num_labels = int(self.labels.max().item()) + 1
        if self.num_labels < 2:
            raise ValueError("SetAnomalyDataset requires at least 2 distinct classes.")

        self.img_idx_by_label = torch.argsort(self.labels).reshape(self.num_labels, -1)
        if self.img_idx_by_label.shape[1] < self.set_size:
            raise ValueError(
                f"Each class has only {self.img_idx_by_label.shape[1]} samples, "
                f"which is fewer than required normal set size ({self.set_size})."
            )

        self.test_sets: Tensor | None = None
        if not train:
            self.test_sets = self._create_test_sets()

    def _create_test_sets(self) -> Tensor:
        """Pre-generate deterministic sets for every image in evaluation mode."""
        num_imgs = self.img_feats.shape[0]
        test_sets = [self.sample_img_set(int(self.labels[idx].item())) for idx in range(num_imgs)]
        return torch.stack(test_sets, dim=0)

    def sample_img_set(self, anomaly_label: int) -> Tensor:
        """Sample ``set_size - 1`` image indices from a class other than ``anomaly_label``."""
        set_label = int(self._rng.integers(0, self.num_labels - 1))
        if set_label >= anomaly_label:
            set_label += 1

        chosen = self._rng.choice(
            self.img_idx_by_label.shape[1],
            size=self.set_size,
            replace=False,
        )
        return self.img_idx_by_label[set_label, chosen]

    def __len__(self) -> int:
        return int(self.img_feats.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        anomaly = self.img_feats[idx]
        if self.train or self.test_sets is None:
            img_indices = self.sample_img_set(int(self.labels[idx].item()))
        else:
            img_indices = self.test_sets[idx]

        # Concatenate images; anomaly is placed at the last index for simplicity
        img_set = torch.cat([self.img_feats[img_indices], anomaly.unsqueeze(0)], dim=0)
        indices = torch.cat([img_indices, torch.tensor([idx], dtype=torch.long)], dim=0)
        label = int(img_set.shape[0]) - 1

        return img_set, indices, label


def split_train_val_features(
    features: Tensor,
    labels: Tensor,
    val_fraction: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split balanced per-class features and labels into train and validation partitions.

    Args:
        features: Feature tensor of shape ``(num_samples, feat_dim)``.
        labels: Integer label tensor of shape ``(num_samples,)``.
        val_fraction: Fraction of samples per class reserved for validation (in ``(0, 1)``).

    Returns:
        Tuple ``(train_feats, train_labels, val_feats, val_labels)``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}.")

    labels_long = labels.long()
    num_labels = int(labels_long.max().item()) + 1
    sorted_indices = torch.argsort(labels_long).reshape(num_labels, -1)

    num_val_per_class = max(1, int(sorted_indices.shape[1] * val_fraction))
    val_indices = sorted_indices[:, :num_val_per_class].reshape(-1)
    train_indices = sorted_indices[:, num_val_per_class:].reshape(-1)

    return (
        features[train_indices],
        labels_long[train_indices],
        features[val_indices],
        labels_long[val_indices],
    )
