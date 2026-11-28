# transformers-mh-attention

UvA Deep Learning Tutorials（*Tutorial 5: Transformers and Multi-Head Attention*, `Lightning-AI/tutorials`）の Jupyter Notebook をベースに、`uv` によるパッケージ管理、厳密な型ヒント、`pytest` による自動テスト、モジュール分割、および既存バグの修正を施したリファクタリング版リポジトリです。

## 元コードの情報・クレジット（Attribution）

本リポジトリは、以下の原作（[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)）を改変して作成しています。

- **原作タイトル**: Tutorial 5: Transformers and Multi-Head Attention（UvA Deep Learning Tutorials）
- **原著者**: Phillip Lippe
- **原作リポジトリ**: `Lightning-AI/tutorials`（https://uvadlc-notebooks.rtfd.io）
- **参照ファイル**: `.notebooks/course_UvA-DL/05-transformers-and-MH-attention.ipynb`
- **参照 URL**: https://github.com/Lightning-AI/tutorials/blob/69112e6fe73b50d159c4b7add6ddea412b458691/.notebooks/course_UvA-DL/05-transformers-and-MH-attention.ipynb
- **原作ライセンス**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) — https://creativecommons.org/licenses/by-sa/4.0/

### 原作からの主な変更点（Indication of Changes）

- 単一の Jupyter Notebook を `src/` レイアウトの複数モジュールへ分割し、`uv`（`pyproject.toml` / `uv.lock`）でパッケージ管理
- 全モジュールへの型ヒント付与と `mypy --strict` / `ruff` による静的解析対応
- `MultiheadAttention.forward` の `input_dim != embed_dim` 時の形状バグ、`TransformerEncoder.get_attention_maps` のマスク未伝播、`scaled_dot_product` の `float16` オーバーフロー、非推奨 API（`_LRScheduler`, `resnet34(pretrained=True)`）、グローバル NumPy RNG 汚染の修正
- `pytest` による単体テスト・回帰テスト・スモークテストの追加

## ライセンス（License）

本リポジトリのコードおよびドキュメントは、原作の ShareAlike 条件に従い、**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** のもとで公開されています。詳細は [LICENSE](./LICENSE) および https://creativecommons.org/licenses/by-sa/4.0/ を参照してください。

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
