"""CLI for Phase 2 corpus cleaning.

Usage
-----
    python -m pwrules.clean \\
        --input /kaggle/input/rockyou/rockyou.txt \\
        --out   /kaggle/working/clean

Optional flags
--------------
    --filter            Enable length/charset filter.
    --min-len N         Minimum password length (default 4, requires --filter).
    --max-len N         Maximum password length (default 64, requires --filter).
    --protocol PATH     Override configs/protocol.yaml.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pwrules.clean",
        description="Phase 2 — clean corpus and produce train/val/test splits.",
    )
    p.add_argument(
        "--input", default=None,
        help="Raw corpus file path (read-only). Auto-discovered (rockyou.txt) if omitted.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output directory (default: <working>/clean).",
    )
    p.add_argument(
        "--filter",
        dest="filter_enabled",
        action="store_true",
        default=False,
        help="Enable optional length/charset filter (off by default).",
    )
    p.add_argument("--min-len", type=int, default=0, help="Min password length.")
    p.add_argument("--max-len", type=int, default=0, help="Max password length.")
    p.add_argument(
        "--protocol",
        default=None,
        help="Path to protocol.yaml (default: configs/protocol.yaml).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
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

    from pwrules.clean import clean_corpus
    from pwrules import paths

    input_path = args.input or str(paths.corpus())
    out_dir = args.out or str(paths.out("clean"))
    logging.info("input : %s", input_path)
    logging.info("output: %s", out_dir)

    result = clean_corpus(
        input_path=input_path,
        out_dir=out_dir,
        protocol_path=args.protocol,
        filter_enabled=args.filter_enabled,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    splits = result["splits"]
    total = result["stats"]["total"]
    print(
        f"\nDone. {total:,} unique passwords cleaned.\n"
        f"  train : {len(splits['train']):,}\n"
        f"  val   : {len(splits['val']):,}\n"
        f"  test  : {len(splits['test']):,}\n"
        f"Outputs in: {result['out_dir']}"
    )


if __name__ == "__main__":
    main()
