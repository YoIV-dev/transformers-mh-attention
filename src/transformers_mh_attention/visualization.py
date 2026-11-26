"""Visualization utilities for attention maps, positional encodings, and LR schedules."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torch import Tensor

from transformers_mh_attention.encoder import PositionalEncoding
from transformers_mh_attention.scheduler import CosineWarmupScheduler


def plot_attention_maps(
    attn_maps: Sequence[Tensor],
    input_data: Tensor | None = None,
    idx: int = 0,
    show: bool = True,
) -> Figure:
    """Plot attention weight heatmaps for each layer and head of sample ``idx``."""
    if input_data is not None:
        labels = input_data[idx].detach().cpu().numpy()
    else:
        labels = np.arange(attn_maps[0][idx].shape[-1])

    maps_np = [m[idx].detach().cpu().numpy() for m in attn_maps]
    num_heads = int(maps_np[0].shape[0])
    num_layers = len(maps_np)
    seq_len = int(labels.shape[0])
    fig_size = 4 if num_heads == 1 else 3

    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=(num_heads * fig_size, num_layers * fig_size),
        squeeze=False,
    )
    for row in range(num_layers):
        for col in range(num_heads):
            ax = axes[row][col]
            ax.imshow(maps_np[row][col], origin="lower", vmin=0.0)
            ax.set_xticks(list(range(seq_len)))
            ax.set_xticklabels(labels.tolist())
            ax.set_yticks(list(range(seq_len)))
            ax.set_yticklabels(labels.tolist())
            ax.set_title(f"Layer {row + 1}, Head {col + 1}")

    fig.subplots_adjust(hspace=0.5)
    if show:
        plt.show()
    return fig


def plot_positional_encoding(
    d_model: int = 48,
    max_len: int = 96,
    show: bool = True,
) -> Figure:
    """Visualize sinusoidal positional encodings across positions and hidden dimensions."""
    encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
    pe = encoding.pe.squeeze(0).T.cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Hidden dimension")
    ax.set_title("Positional encoding over hidden dimensions")
    ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
    ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])

    if show:
        plt.show()
    return fig


def plot_lr_schedule(
    scheduler: CosineWarmupScheduler,
    num_steps: int,
    show: bool = True,
) -> Figure:
    """Plot the multiplicative learning rate factor over ``num_steps`` iterations."""
    steps = list(range(num_steps))
    factors = [scheduler.get_lr_factor(step) for step in steps]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, factors)
    ax.set_ylabel("Learning rate factor")
    ax.set_xlabel("Iterations (in batches)")
    ax.set_title("Cosine Warm-up Learning Rate Scheduler")

    if show:
        plt.show()
    return fig
