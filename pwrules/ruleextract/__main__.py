"""CLI for Phase 3 rule extraction.

Usage
-----
    python -m pwrules.ruleextract \\
        --clean  /kaggle/input/pwrules-clean/clean \\
        --out    /kaggle/working/rules

Optional flags
--------------
    --wordlist PATH   Base wordlist for base-word selection
                      (default: built from the train split).
    --protocol PATH   Override configs/protocol.yaml.
    --parity N        After extraction, run a Hashcat parity check on N random
                      triples (requires hashcat on PATH; default 100).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pwrules.ruleextract",
        description="Phase 3 — extract validated (base, rule, password) triples.",
    )
    p.add_argument("--clean", required=True, help="Phase 2 output dir (train/val/test splits).")
    p.add_argument("--out", required=True, help="Output directory for rule extraction results.")
    p.add_argument("--wordlist", default=None, help="Reference base wordlist path.")
    p.add_argument("--protocol", default=None, help="Path to protocol.yaml.")
    p.add_argument(
        "--parity",
        type=int,
        default=100,
        metavar="N",
        help="Number of triples to cross-check against real hashcat (0 = skip).",
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

    from pwrules.ruleextract import extract_rules
    from pwrules.ruleextract.extractor import parity_check

    result = extract_rules(
        clean_dir=args.clean,
        out_dir=args.out,
        base_wordlist_path=args.wordlist,
        protocol_path=args.protocol,
    )

    triples = result["triples"]
    report = result["coverage_report"]

    print(
        f"\nPhase 3 done.\n"
        f"  Train: {report['train']['triples']:,} triples  "
        f"({report['train']['coverage_pct']}% coverage)\n"
        f"  Val  : {report['val']['triples']:,} triples  "
        f"({report['val']['coverage_pct']}% coverage)\n"
        f"Outputs in: {result['out_dir']}"
    )

    # Optional Hashcat parity check.
    if args.parity > 0 and triples:
        from pwrules.config import load_protocol
        seed = int(load_protocol().get("seed", 1337))
        rng = random.Random(seed)
        sample = rng.sample(triples, min(args.parity, len(triples)))
        sample_tuples = [(t["base"], t["rule"], t["password"]) for t in sample]
        passed, total = parity_check(sample_tuples)
        if total == 0:
            print("\nParity check SKIPPED (hashcat not found on PATH).")
        else:
            pct = passed / total * 100
            print(f"\nHashcat parity check: {passed}/{total} passed ({pct:.1f}%)")
            if passed < total:
                failures = total - passed
                print(
                    f"WARNING: {failures} triples failed parity. "
                    "Check rule applier correctness.",
                    file=sys.stderr,
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
