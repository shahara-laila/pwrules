"""CLI entrypoint: python -m pwrules.filter"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pwrules.config import load_protocol, set_seed
from pwrules.filter import filter_rules


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m pwrules.filter",
        description="Phase 7 — 3-stage rule filtering funnel.",
    )
    p.add_argument(
        "--rules", nargs="+", required=True,
        help="One or more .rule files to filter.",
    )
    p.add_argument(
        "--out", required=True,
        help="Output directory.",
    )
    p.add_argument(
        "--val", default=None,
        help="Validation password file (required for --effectiveness-ranking).",
    )
    p.add_argument(
        "--wordlist", default=None,
        help="Base wordlist (required for --effectiveness-ranking).",
    )
    p.add_argument(
        "--hashcat-bin", default="hashcat",
        help="Path to the hashcat binary.",
    )
    p.add_argument(
        "--hashcat-sample", type=int, default=200,
        help="Number of rules to cross-check with real hashcat (0 = skip).",
    )
    p.add_argument(
        "--effectiveness-ranking", action="store_true",
        help="Enable Stage 3 effectiveness ranking (needs --val and --wordlist).",
    )
    p.add_argument(
        "--top-k", type=int, default=None,
        help="Keep only the top-k most effective rules (Stage 3 only).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    proto = load_protocol()
    seed = args.seed if args.seed is not None else proto.get("seed", 1337)
    set_seed(seed)

    if args.effectiveness_ranking and (not args.val or not args.wordlist):
        print(
            "ERROR: --effectiveness-ranking requires --val and --wordlist.",
            file=sys.stderr,
        )
        sys.exit(1)

    rule_files = [Path(r) for r in args.rules]
    for rf in rule_files:
        if not rf.exists():
            print(f"ERROR: rule file not found: {rf}", file=sys.stderr)
            sys.exit(1)

    result = filter_rules(
        rule_files=rule_files,
        out_dir=Path(args.out),
        val_path=Path(args.val) if args.val else None,
        base_wordlist_path=Path(args.wordlist) if args.wordlist else None,
        hashcat_bin=args.hashcat_bin,
        hashcat_sample=args.hashcat_sample,
        effectiveness_ranking=args.effectiveness_ranking,
        top_k=args.top_k,
        seed=seed,
    )

    print("\nFilter funnel summary:")
    for row in result["funnel"]:
        print(
            f"  {row['file']}: "
            f"{row['generated']} generated → {row['valid']} valid → "
            f"{row['unique']} unique → {row['effective']} effective"
        )
    print(f"\nOutput: {result['out_dir']}")


if __name__ == "__main__":
    main()
