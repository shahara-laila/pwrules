"""CLI for Phase 5 QLoRA fine-tuning.

Usage
-----
    python -m pwrules.train \\
        --data   /kaggle/input/pwrules-rules \\
        --out    /kaggle/working/adapter \\
        --config configs/train.yaml

    # Resume after a 12-hour Kaggle session:
    python -m pwrules.train \\
        --data    /kaggle/input/pwrules-rules \\
        --out     /kaggle/working/adapter \\
        --resume  /kaggle/input/pwrules-adapter/checkpoints

    # Use targeted (Phase 4) instruction dataset:
    python -m pwrules.train \\
        --data     /kaggle/input/pwrules-targeted \\
        --out      /kaggle/working/adapter_targeted \\
        --targeted
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pwrules.train",
        description=(
            "Phase 5 — QLoRA fine-tuning with Unsloth + TRL SFTTrainer. "
            "Requires a GPU. Run on Kaggle with GPU enabled."
        ),
    )
    p.add_argument(
        "--data", default=None,
        help=(
            "Directory containing instruction JSONL files "
            "(instructions_train.jsonl, instructions_val.jsonl, "
            "and optionally targeted_dataset.jsonl). "
            "Auto-discovered if omitted (targeted dir when --targeted)."
        ),
    )
    p.add_argument(
        "--out", default=None,
        help="Output directory (default: <working>/adapter).",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to train.yaml (default: configs/train.yaml).",
    )
    p.add_argument(
        "--targeted", action="store_true", default=False,
        help="Use the targeted instruction dataset (Phase 4 output).",
    )
    p.add_argument(
        "--resume", default=None,
        help="Checkpoint directory to resume from (overrides train.yaml setting).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from pwrules.train import train
    from pwrules import paths

    if args.data:
        data_dir = args.data
    elif args.targeted:
        data_dir = str(paths.targeted_dir())
    else:
        data_dir = str(paths.rules_dir())
    out_dir = args.out or str(paths.out("adapter"))
    logging.info("data  : %s", data_dir)
    logging.info("output: %s", out_dir)
    if args.resume:
        logging.info("resume: %s", args.resume)

    try:
        adapter_dir = train(
            data_dir=data_dir,
            output_dir=out_dir,
            config_path=args.config,
            use_targeted=args.targeted,
            resume_from_checkpoint=args.resume,
        )
        print(f"\nPhase 5 done. Adapter saved to: {adapter_dir}")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
