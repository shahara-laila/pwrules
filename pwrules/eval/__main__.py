"""CLI entrypoint: python -m pwrules.eval

Subcommands
-----------
run      Phase 8: evaluate rule sets and compute Hit@k.
ablate   Phase 9: ablation table + significance vs baseline.
report   Phase 10: export paper-ready artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _run_cmd(args):
    from pwrules.config import load_protocol, set_seed
    from pwrules.eval import run_eval
    from pwrules.eval.baselines import get_best64_rule, run_ruleforge

    proto = load_protocol()
    seed = args.seed if args.seed is not None else proto.get("seed", 1337)
    set_seed(seed)

    # Auto-detect best64 if not supplied.
    best64_rule = None
    if args.best64:
        best64_rule = Path(args.best64)
    elif not args.no_best64:
        best64_rule = get_best64_rule()

    # Optionally generate RuleForge baseline.
    ruleforge_rule = None
    if args.ruleforge:
        ruleforge_rule = Path(args.ruleforge)
    elif args.run_ruleforge:
        ruleforge_rule = run_ruleforge(
            Path(args.wordlist),
            out_dir=Path(args.out) / "baselines",
            variant=args.ruleforge_variant,
        )

    result = run_eval(
        wordlist_path=Path(args.wordlist),
        test_path=Path(args.test),
        out_dir=Path(args.out),
        llm_untargeted_rule=Path(args.llm_untargeted) if args.llm_untargeted else None,
        llm_targeted_rule=Path(args.llm_targeted) if args.llm_targeted else None,
        llm_filtered_rule=Path(args.llm_filtered) if args.llm_filtered else None,
        best64_rule=best64_rule,
        ruleforge_rule=ruleforge_rule,
        targeted_rules_dir=Path(args.targeted_rules_dir) if args.targeted_rules_dir else None,
        target_users_path=Path(args.target_users) if args.target_users else None,
        hashcat_bin=args.hashcat_bin,
        seed=seed,
        dataset_name=args.dataset_name,
    )

    print(f"\nEvaluation complete.")
    print(f"  Methods evaluated : {result['methods']}")
    print(f"  Budget schedule   : {result['k_values']}")
    print(f"  Results CSV       : {result['results_csv']}")
    print(f"  Output directory  : {result['out_dir']}")


def _ablate_cmd(args):
    from pwrules.eval.ablations import run_ablations

    result = run_ablations(
        results_dir=Path(args.results_dir),
        out_dir=Path(args.out),
        baseline_method=args.baseline,
        k_pivot=int(args.k_pivot),
        n_bootstrap=int(args.n_bootstrap),
        min_seeds=int(args.min_seeds),
    )

    print(f"\nAblations complete.")
    print(f"  Ablation conditions : {result.get('n_ablation_conditions', 0)}")
    print(f"  Significance tests  : {result.get('n_significance_tests', 0)}")
    print(f"  Output directory    : {result.get('out_dir')}")


def _report_cmd(args):
    from pwrules.eval.reporting import export_paper_artifacts

    k_values = [int(k) for k in args.k_values.split(",")] if args.k_values else None

    result = export_paper_artifacts(
        results_dir=Path(args.results_dir),
        out_dir=Path(args.out),
        k_values=k_values,
        ablations_dir=Path(args.ablations_dir) if args.ablations_dir else None,
        filter_dir=Path(args.filter_dir) if args.filter_dir else None,
        dataset=args.dataset,
    )

    if result["missing"]:
        print(f"\nWARNING: {result['n_missing']} source file(s) missing:")
        for m in result["missing"]:
            print(f"  MISSING: {m}")
    else:
        print("\nAll source files present.")
    print(f"Paper artifacts → {result['out_dir']}")


def _build_parser():
    p = argparse.ArgumentParser(
        prog="python -m pwrules.eval",
        description="Phases 8–10: evaluate, ablate, and export results.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    r = sub.add_parser("run", help="Phase 8: evaluate rule sets.")
    r.add_argument("--wordlist", required=True, help="Base wordlist path.")
    r.add_argument("--test",     required=True, help="Test plaintext path.")
    r.add_argument("--out",      required=True, help="Output directory.")
    r.add_argument("--llm-untargeted", default=None)
    r.add_argument("--llm-targeted",   default=None)
    r.add_argument("--llm-filtered",   default=None)
    r.add_argument("--best64",         default=None, help="Path to best64.rule.")
    r.add_argument("--no-best64",      action="store_true", help="Skip best64 baseline.")
    r.add_argument("--ruleforge",      default=None, help="Path to pre-built RuleForge rule file.")
    r.add_argument("--run-ruleforge",  action="store_true", help="Generate RuleForge baseline.")
    r.add_argument("--ruleforge-variant", default="auto",
                   choices=["auto", "mdbscan", "dbscan", "hac"])
    r.add_argument("--targeted-rules-dir", default=None,
                   help="Directory of per-user rule files for targeted eval.")
    r.add_argument("--target-users", default=None,
                   help="JSONL file with test user {user_id, password} records.")
    r.add_argument("--hashcat-bin", default="hashcat")
    r.add_argument("--dataset-name", default="default")
    r.add_argument("--seed", type=int, default=None)

    # ------------------------------------------------------------------
    # ablate
    # ------------------------------------------------------------------
    a = sub.add_parser("ablate", help="Phase 9: ablation + significance.")
    a.add_argument("--results-dir", required=True)
    a.add_argument("--out",         required=True)
    a.add_argument("--baseline",    default="best64")
    a.add_argument("--k-pivot",     default=1_000_000)
    a.add_argument("--n-bootstrap", default=10_000)
    a.add_argument("--min-seeds",   default=3)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    rp = sub.add_parser("report", help="Phase 10: export paper artifacts.")
    rp.add_argument("--results-dir",   required=True)
    rp.add_argument("--out",           required=True)
    rp.add_argument("--ablations-dir", default=None)
    rp.add_argument("--filter-dir",    default=None)
    rp.add_argument("--dataset",       default=None)
    rp.add_argument("--k-values",      default=None,
                    help="Comma-separated k values, e.g. '10,100,1000'.")

    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "run":
        _run_cmd(args)
    elif args.command == "ablate":
        _ablate_cmd(args)
    elif args.command == "report":
        _report_cmd(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
