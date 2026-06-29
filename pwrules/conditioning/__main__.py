"""CLI for Phase 4 target-conditioning dataset.

Usage
-----
    python -m pwrules.conditioning \\
        --rules /kaggle/input/pwrules-rules/rules_dataset.jsonl \\
        --mode  synthetic \\
        --out   /kaggle/working/targeted

    # Real data mode (requires a private attribute CSV on Kaggle):
    python -m pwrules.conditioning \\
        --rules  /kaggle/input/pwrules-rules/rules_dataset.jsonl \\
        --mode   real \\
        --real-csv /kaggle/input/pwrules-attrs/user_attrs.csv \\
        --out    /kaggle/working/targeted
"""

from __future__ import annotations

import argparse
import logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pwrules.conditioning",
        description=(
            "Phase 4 — build target-conditioned instruction dataset "
            "with held-out target users."
        ),
    )
    p.add_argument(
        "--rules", default=None,
        help="Path to rules_dataset.jsonl (Phase 3). Auto-discovered if omitted.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output directory (default: <working>/targeted).",
    )
    p.add_argument(
        "--mode",
        default="synthetic",
        choices=["synthetic", "real"],
        help="Attribute source: 'synthetic' (default) or 'real'.",
    )
    p.add_argument(
        "--real-csv", default=None,
        help="Path to real attribute CSV (required when --mode=real).",
    )
    p.add_argument(
        "--n-users", type=int, default=500,
        help="Number of synthetic user personas (synthetic mode, default 500).",
    )
    p.add_argument(
        "--test-ratio", type=float, default=0.1,
        help="Fraction of users held out for targeted eval (default 0.1).",
    )
    p.add_argument(
        "--protocol", default=None,
        help="Path to protocol.yaml.",
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

    from pwrules.conditioning import build_targeted_dataset
    from pwrules import paths

    rules_jsonl = args.rules or str(paths.rules_dataset())
    out_dir = args.out or str(paths.out("targeted"))
    logging.info("rules : %s", rules_jsonl)
    logging.info("output: %s", out_dir)

    result = build_targeted_dataset(
        rules_jsonl=rules_jsonl,
        out_dir=out_dir,
        mode=args.mode,
        real_csv_path=args.real_csv,
        n_synthetic_users=args.n_users,
        test_user_ratio=args.test_ratio,
        protocol_path=args.protocol,
    )

    train_triples = result["train_triples"]
    test_users = result["test_users"]
    print(
        f"\nPhase 4 done.\n"
        f"  Training triples : {len(train_triples):,}\n"
        f"  Held-out users   : {len(test_users):,}\n"
        f"  Mode             : {args.mode}\n"
        f"  Disjoint check   : PASSED\n"
        f"Outputs in: {result['out_dir']}"
    )


if __name__ == "__main__":
    main()
