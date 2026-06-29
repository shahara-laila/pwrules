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
    from pwrules import paths

    proto = load_protocol()
    seed = args.seed if args.seed is not None else proto.get("seed", 1337)
    set_seed(seed)

    # Resolve protocol inputs/outputs (env override -> discovery -> explicit).
    wordlist = Path(args.wordlist) if args.wordlist else paths.train_txt()
    test = Path(args.test) if args.test else paths.test_txt()
    out_dir = Path(args.out) if args.out else paths.out("results")

    # LLM rule sets: prefer explicit, else the filtered untargeted rules.
    llm_untargeted = Path(args.llm_untargeted) if args.llm_untargeted else \
        paths.filtered_untargeted(required=False)
    llm_filtered = Path(args.llm_filtered) if args.llm_filtered else llm_untargeted
    target_users = Path(args.target_users) if args.target_users else \
        paths.target_users(required=False)
    targeted_rules_dir = Path(args.targeted_rules_dir) if args.targeted_rules_dir else \
        paths.filtered_dir(required=False)

    logging.info("wordlist: %s", wordlist)
    logging.info("test    : %s", test)
    logging.info("output  : %s", out_dir)

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
            wordlist,
            out_dir=out_dir / "baselines",
            variant=args.ruleforge_variant,
        )

    result = run_eval(
        wordlist_path=wordlist,
        test_path=test,
        out_dir=out_dir,
        llm_untargeted_rule=llm_untargeted,
        llm_targeted_rule=Path(args.llm_targeted) if args.llm_targeted else None,
        llm_filtered_rule=llm_filtered,
        best64_rule=best64_rule,
        ruleforge_rule=ruleforge_rule,
        targeted_rules_dir=targeted_rules_dir,
        target_users_path=target_users,
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
    from pwrules import paths

    results_dir = Path(args.results_dir) if args.results_dir else paths.results_dir()
    out_dir = Path(args.out) if args.out else paths.out("ablations")
    logging.info("results-dir: %s", results_dir)
    logging.info("output     : %s", out_dir)

    result = run_ablations(
        results_dir=results_dir,
        out_dir=out_dir,
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
    from pwrules import paths

    k_values = [int(k) for k in args.k_values.split(",")] if args.k_values else None

    results_dir = Path(args.results_dir) if args.results_dir else paths.results_dir()
    out_dir = Path(args.out) if args.out else paths.out("paper")
    ablations_dir = Path(args.ablations_dir) if args.ablations_dir else \
        paths.ablations_dir(required=False)
    filter_dir = Path(args.filter_dir) if args.filter_dir else \
        paths.filtered_dir(required=False)

    result = export_paper_artifacts(
        results_dir=results_dir,
        out_dir=out_dir,
        k_values=k_values,
        ablations_dir=ablations_dir,
        filter_dir=filter_dir,
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
    r.add_argument("--wordlist", default=None, help="Base wordlist (default: discovered train.txt).")
    r.add_argument("--test",     default=None, help="Test plaintext (default: discovered test.txt).")
    r.add_argument("--out",      default=None, help="Output directory (default: <working>/results).")
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
    a.add_argument("--results-dir", default=None, help="Dir of result CSVs (default: discovered).")
    a.add_argument("--out",         default=None, help="Output directory (default: <working>/ablations).")
    a.add_argument("--baseline",    default="best64")
    a.add_argument("--k-pivot",     default=1_000_000)
    a.add_argument("--n-bootstrap", default=10_000)
    a.add_argument("--min-seeds",   default=3)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    rp = sub.add_parser("report", help="Phase 10: export paper artifacts.")
    rp.add_argument("--results-dir",   default=None, help="Phase 8 results dir (default: discovered).")
    rp.add_argument("--out",           default=None, help="Output directory (default: <working>/paper).")
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
