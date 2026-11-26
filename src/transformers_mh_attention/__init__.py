"""Refactored Multi-Head Attention and Transformer implementation."""

from transformers_mh_attention.attention import MultiheadAttention, scaled_dot_product
from transformers_mh_attention.datasets import (
    ReverseDataset,
    SetAnomalyDataset,
    split_train_val_features,
)
from transformers_mh_attention.encoder import (
    EncoderBlock,
    PositionalEncoding,
    TransformerEncoder,
)
from transformers_mh_attention.models import (
    AnomalyPredictor,
    ReversePredictor,
    TransformerPredictor,
)
from transformers_mh_attention.scheduler import CosineWarmupScheduler
from transformers_mh_attention.training import (
    AnomalyTrainConfig,
    ReverseTrainConfig,
    extract_features,
    get_device,
    train_anomaly,
    train_reverse,
)

__all__ = [
    "AnomalyPredictor",
    "AnomalyTrainConfig",
    "CosineWarmupScheduler",
    "EncoderBlock",
    "MultiheadAttention",
    "PositionalEncoding",
    "ReverseDataset",
    "ReversePredictor",
    "ReverseTrainConfig",
    "SetAnomalyDataset",
    "TransformerEncoder",
    "TransformerPredictor",
    "extract_features",
    "get_device",
    "scaled_dot_product",
    "split_train_val_features",
    "train_anomaly",
    "train_reverse",
]
