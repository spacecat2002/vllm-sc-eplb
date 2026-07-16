from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_gate_lora.collect import CollectionConfig, collect
from moe_gate_lora.plot import write_overlap_plot


def run_pipeline(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = work_dir / "lora"
    common = {
        "model": args.model,
        "ep_size": args.ep_size,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "collect_batch_size": args.batch_size,
        "timeout": args.timeout,
        "load_format": args.load_format,
        "moe_backend": args.moe_backend,
        "rank_dim": args.rank_dim,
        "alpha": args.alpha,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": args.device,
        "lora_dir": lora_dir,
    }
    train = collect(
        CollectionConfig(
            **common,
            prompts=args.prompts,
            output_dir=work_dir / "train",
            mode="train",
            epochs=args.epochs,
        )
    )
    evaluate = collect(
        CollectionConfig(
            **common,
            prompts=args.eval_prompts or args.prompts,
            output_dir=work_dir / "eval",
            mode="eval",
            epochs=1,
        )
    )
    summary = {"train": train, "eval": evaluate}
    (work_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved streaming training and evaluation under {work_dir}")


def refresh_plot(args: argparse.Namespace) -> None:
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    output = args.output or args.metrics.with_name("overlap.png")
    write_overlap_plot(payload["rows"], output)
    print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming next-gate LoRA training for vLLM MoE models"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pipeline = subparsers.add_parser("pipeline")
    pipeline.add_argument("--model", required=True)
    pipeline.add_argument("--prompts", type=Path)
    pipeline.add_argument("--eval-prompts", type=Path)
    pipeline.add_argument("--work-dir", type=Path, required=True)
    pipeline.add_argument("--ep-size", type=int, default=1)
    pipeline.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Prompts per DP-rank inference batch; one optimizer step consumes "
            "the active ranks."
        ),
    )
    pipeline.add_argument("--max-model-len", type=int, default=4096)
    pipeline.add_argument("--max-new-tokens", type=int, default=16)
    pipeline.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs; evaluation always runs once.",
    )
    pipeline.add_argument("--rank-dim", type=int, default=8)
    pipeline.add_argument("--alpha", type=float, default=16.0)
    pipeline.add_argument("--lr", type=float, default=1e-3)
    pipeline.add_argument("--weight-decay", type=float, default=0.0)
    pipeline.add_argument("--seed", type=int, default=0)
    pipeline.add_argument("--device")
    pipeline.add_argument("--timeout", type=int, default=1800)
    pipeline.add_argument("--load-format", default="auto")
    pipeline.add_argument("--moe-backend", default="auto")
    pipeline.set_defaults(func=run_pipeline)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--metrics", type=Path, required=True)
    plot.add_argument("--output", type=Path)
    plot.set_defaults(func=refresh_plot)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
