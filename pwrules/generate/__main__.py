"""CLI for Phase 6 rule generation.

Usage
-----
    python -m pwrules.generate \\
        --adapter  /kaggle/input/pwrules-adapter/adapter \\
        --out      /kaggle/working/rules

    # With targeted generation:
    python -m pwrules.generate \\
        --adapter      /kaggle/input/pwrules-adapter/adapter \\
        --out          /kaggle/working/rules \\
        --target-users /kaggle/input/pwrules-targeted/target_users_test.jsonl

    # With a custom probe wordlist:
    python -m pwrules.generate \\
        --adapter  /kaggle/input/pwrules-adapter/adapter \\
        --wordlist /kaggle/input/rockyou/rockyou.txt \\
        --budget   5000 \\
        --out      /kaggle/working/rules
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pwrules.generate",
        description=(
            "Phase 6 — generate Hashcat rules using the fine-tuned adapter. "
            "Requires a GPU. Run on Kaggle with GPU enabled."
        ),
    )
    p.add_argument(
        "--adapter", default=None,
        help="LoRA adapter directory (Phase 5). Auto-discovered if omitted.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output directory (default: <working>/rules).",
    )
    p.add_argument(
        "--target-users", default=None,
        help=(
            "Path to target_users_test.jsonl (Phase 4); enables targeted mode. "
            "Auto-discovered if present, unless --no-targeted is given."
        ),
    )
    p.add_argument(
        "--no-targeted", action="store_true", default=False,
        help="Skip targeted generation even if target users are discoverable.",
    )
    p.add_argument(
        "--wordlist", default=None,
        help="Probe wordlist for generation prompts (default: built-in list).",
    )
    p.add_argument(
        "--budget", type=int, default=None,
        help="Override generation.budget from configs/train.yaml.",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to train.yaml.",
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

    from pwrules.generate import generate_rules
    from pwrules import paths

    adapter_dir = args.adapter or str(paths.adapter_dir())
    out_dir = args.out or str(paths.out("rules"))
    if args.no_targeted:
        target_users = None
    else:
        target_users = args.target_users or (
            str(paths.target_users()) if paths.target_users(required=False) else None
        )
    logging.info("adapter: %s", adapter_dir)
    logging.info("output : %s", out_dir)
    logging.info("targeted users: %s", target_users or "(none — untargeted only)")

    try:
        result = generate_rules(
            adapter_dir=adapter_dir,
            out_dir=out_dir,
            target_users_path=target_users,
            probe_words_path=args.wordlist,
            config_path=args.config,
            budget=args.budget,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    unt = result["untargeted_rules"]
    tgt = result["targeted_rules"]
    print(
        f"\nPhase 6 done.\n"
        f"  Untargeted rules : {len(unt):,} unique\n"
        f"  Targeted users   : {len(tgt):,}\n"
        f"Outputs in: {result['out_dir']}"
    )


if __name__ == "__main__":
    main()
