"""Command-line interface for running Transformer tutorial experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from transformers_mh_attention.training import ReverseTrainConfig, train_reverse


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="transformers-mh-attention",
        description="Run refactored UvA-DL Tutorial 6 Transformer experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reverse_parser = subparsers.add_parser(
        "reverse",
        help="Train and evaluate the sequence-reversal Transformer.",
    )
    reverse_parser.add_argument("--max-epochs", type=int, default=10)
    reverse_parser.add_argument("--batch-size", type=int, default=128)
    reverse_parser.add_argument("--train-size", type=int, default=50_000)
    reverse_parser.add_argument("--val-size", type=int, default=1_000)
    reverse_parser.add_argument("--test-size", type=int, default=10_000)
    reverse_parser.add_argument("--model-dim", type=int, default=32)
    reverse_parser.add_argument("--num-heads", type=int, default=1)
    reverse_parser.add_argument("--num-layers", type=int, default=1)
    reverse_parser.add_argument("--lr", type=float, default=5e-4)
    reverse_parser.add_argument("--warmup", type=int, default=50)
    reverse_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("saved_models/Transformers"),
    )
    reverse_parser.add_argument("--accelerator", type=str, default="auto")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "reverse":
        config = ReverseTrainConfig(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            lr=args.lr,
            warmup=args.warmup,
            max_epochs=args.max_epochs,
            checkpoint_dir=args.checkpoint_dir,
        )
        _, metrics = train_reverse(config=config, accelerator=args.accelerator)
        print(f"Val accuracy:  {100.0 * metrics['val_acc']:5.2f}%")
        print(f"Test accuracy: {100.0 * metrics['test_acc']:5.2f}%")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
