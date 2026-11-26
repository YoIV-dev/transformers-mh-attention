# transformers-mh-attention

UvA Deep Learning Tutorial 6 (*Transformers and Multi-Head Attention*, Lightning-AI/tutorials) の Jupyter Notebook をベースに、`uv` によるパッケージ管理、厳密な型ヒント、`pytest` による自動テスト、モジュール分割、および既存バグの修正を施したリファクタリング版リポジトリです。

## 元コードの情報

- リポジトリ: `Lightning-AI/tutorials`
- 対象ファイル: `.notebooks/course_UvA-DL/05-transformers-and-MH-attention.ipynb`
- URL: https://github.com/Lightning-AI/tutorials/blob/69112e6fe73b50d159c4b7add6ddea412b458691/.notebooks/course_UvA-DL/05-transformers-and-MH-attention.ipynb

## ディレクトリ構成

```text
.
├── pyproject.toml
├── uv.lock
├── src/
│   └── transformers_mh_attention/
│       ├── __init__.py
│       ├── attention.py       # scaled_dot_product, MultiheadAttention
│       ├── encoder.py         # EncoderBlock, TransformerEncoder, PositionalEncoding
│       ├── scheduler.py       # CosineWarmupScheduler
│       ├── datasets.py        # ReverseDataset, SetAnomalyDataset, split_train_val_features
│       ├── models.py          # TransformerPredictor, ReversePredictor, AnomalyPredictor
│       ├── training.py        # 学習設定 dataclass、特徴抽出、学習関数
│       ├── visualization.py   # Attention Map・Positional Encoding・LR 推移の可視化
│       └── cli.py             # CLI エントリポイント
└── tests/
    ├── test_attention.py
    ├── test_encoder.py
    ├── test_scheduler.py
    ├── test_datasets.py
    ├── test_models.py
    └── test_training.py
```

## セットアップと実行

```bash
# 依存パッケージのインストール（開発ツール含む）
uv sync

# テスト実行
uv run pytest

# 静的型チェック
uv run mypy src tests

# Linter / フォーマッタ確認
uv run ruff check src tests

# CLI から Reverse Task を実行（例: 1エポック）
uv run transformers-mh-attention reverse --max-epochs 1
```
