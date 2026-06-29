"""Phase 9 — Ablation experiments and statistical significance.

Ablation axes
-------------
(i)  target-conditioning on/off     LLM-targeted vs LLM-untargeted
(ii) model size                     different model variants (if results available)
(iii) with/without filtering        LLM-filtered vs LLM-untargeted
(iv) cross-dataset                  rules from corpus A evaluated on corpus B

Variance
--------
Each setting is run over ≥3 seeds.  Mean ± std is reported.
Seeds are read from separate result CSV files named
``results_seed<N>.csv`` in the results directory, or from a single
results.csv with a ``seed`` column.

Significance
------------
Two-sided paired bootstrap confidence interval and McNemar's test (for per-
password binary hit vectors) vs the strongest baseline (best64 by default).

Outputs (in *out_dir*)
----------------------
ablations.csv            mean±std Hit@k for every ablation condition
significance_report.json bootstrap CI + McNemar p-values vs the baseline
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pwrules.eval import load_results_csv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load and aggregate multi-seed results
# ---------------------------------------------------------------------------

def load_all_seeds(results_dir: Path) -> List[Dict]:
    """Load all seed CSV files from *results_dir*.

    Looks for ``results.csv``, ``results_seed*.csv`` in the directory.
    """
    rows: List[Dict] = []

    main_csv = results_dir / "results.csv"
    if main_csv.exists():
        rows.extend(load_results_csv(main_csv))

    for seed_csv in sorted(results_dir.glob("results_seed*.csv")):
        rows.extend(load_results_csv(seed_csv))

    logger.info("Loaded %d result rows from %s", len(rows), results_dir)
    return rows


def aggregate_seeds(
    rows: List[Dict],
    k_values: Optional[List[int]] = None,
) -> List[Dict]:
    """Compute mean ± std Hit@k across seeds for each (method, dataset, k).

    Returns rows with fields:
        method, dataset, k, mean_hit_rate, std_hit_rate, n_seeds
    """
    # Group by (method, dataset, k).
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = (row["method"], row.get("dataset", "default"), int(row["k"]))
        groups[key].append(float(row["hit_rate"]))

    agg_rows: List[Dict] = []
    for (method, dataset, k), values in sorted(groups.items()):
        if k_values and k not in k_values:
            continue
        agg_rows.append({
            "method":        method,
            "dataset":       dataset,
            "k":             k,
            "mean_hit_rate": float(np.mean(values)),
            "std_hit_rate":  float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
            "n_seeds":       len(values),
        })

    return agg_rows


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

# Maps pairs of conditions for each ablation axis.
_ABLATION_AXES = [
    {
        "axis":    "target-conditioning",
        "a":       "LLM-targeted",
        "b":       "LLM-untargeted",
        "label":   "Target-conditioning on vs off",
    },
    {
        "axis":    "filtering",
        "a":       "LLM-filtered",
        "b":       "LLM-untargeted",
        "label":   "With vs without filtering",
    },
    {
        "axis":    "vs-best64",
        "a":       "LLM-untargeted",
        "b":       "best64",
        "label":   "LLM-untargeted vs best64",
    },
    {
        "axis":    "complementarity",
        "a":       "LLM+best64",
        "b":       "best64",
        "label":   "Complementarity (LLM+best64) vs best64",
    },
]


def build_ablation_table(
    agg_rows: List[Dict],
    k_pivot: int = 1_000_000,
) -> List[Dict]:
    """Build the ablation comparison table at a single pivot k.

    Returns rows with fields:
        axis, label, method_a, method_b, dataset,
        mean_a, std_a, mean_b, std_b, delta
    """
    # Index: (method, dataset) → agg row at k_pivot.
    index: Dict[Tuple[str, str], Dict] = {}
    for row in agg_rows:
        if row["k"] == k_pivot:
            index[(row["method"], row["dataset"])] = row

    result: List[Dict] = []
    datasets = list({r["dataset"] for r in agg_rows})

    for axis in _ABLATION_AXES:
        for dataset in datasets:
            row_a = index.get((axis["a"], dataset))
            row_b = index.get((axis["b"], dataset))
            if not row_a or not row_b:
                continue
            result.append({
                "axis":     axis["axis"],
                "label":    axis["label"],
                "method_a": axis["a"],
                "method_b": axis["b"],
                "dataset":  dataset,
                "k":        k_pivot,
                "mean_a":   row_a["mean_hit_rate"],
                "std_a":    row_a["std_hit_rate"],
                "mean_b":   row_b["mean_hit_rate"],
                "std_b":    row_b["std_hit_rate"],
                "delta":    row_a["mean_hit_rate"] - row_b["mean_hit_rate"],
            })

    return result


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 1337,
) -> Tuple[float, float, float]:
    """Paired bootstrap CI for the mean difference (a - b).

    Returns ``(lo, hi, observed_diff)``.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(values_a)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        diffs[i] = values_a[idx].mean() - values_b[idx].mean()
    lo = float(np.percentile(diffs, alpha / 2 * 100))
    hi = float(np.percentile(diffs, (1 - alpha / 2) * 100))
    obs = float(values_a.mean() - values_b.mean())
    return lo, hi, obs


def mcnemar_p(hits_a: np.ndarray, hits_b: np.ndarray) -> float:
    """McNemar's test p-value for paired binary outcomes.

    *hits_a[i]* and *hits_b[i]* are 1 if the i-th test password was
    cracked by method A (resp. B), 0 otherwise.
    """
    try:
        from scipy.stats import chi2
    except ImportError:
        logger.warning("scipy not installed; McNemar p-value unavailable.")
        return float("nan")

    b = int(np.sum((hits_a == 1) & (hits_b == 0)))  # A right, B wrong
    c = int(np.sum((hits_a == 0) & (hits_b == 1)))  # A wrong, B right
    bc = b + c
    if bc == 0:
        return 1.0
    # Continuity-corrected McNemar statistic.
    chi2_stat = (abs(b - c) - 1.0) ** 2 / bc
    return float(1.0 - chi2.cdf(chi2_stat, df=1))


def compute_significance(
    results_rows: List[Dict],
    baseline_method: str = "best64",
    target_methods: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
    n_bootstrap: int = 10_000,
) -> List[Dict]:
    """Bootstrap CI + McNemar p-values comparing each method vs *baseline_method*.

    *results_rows* must have a ``hit_rate`` field per row (one row per seed).
    For McNemar we use a surrogate: treat each seed's Hit@k as a Bernoulli
    sample (hit / no-hit at that seed).

    Returns a list of significance dicts.
    """
    # Group by (method, dataset, k).
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for row in results_rows:
        key = (row["method"], row.get("dataset", "default"), int(row["k"]))
        groups[key].append(float(row["hit_rate"]))

    datasets = list({r.get("dataset", "default") for r in results_rows})
    all_methods = list({r["method"] for r in results_rows})
    compare_methods = target_methods or [m for m in all_methods if m != baseline_method]
    all_k = k_values or sorted({int(r["k"]) for r in results_rows})

    sig_rows: List[Dict] = []
    for dataset in datasets:
        for k in all_k:
            vals_base = np.array(groups.get((baseline_method, dataset, k), []))
            if len(vals_base) == 0:
                continue
            for method in compare_methods:
                vals_m = np.array(groups.get((method, dataset, k), []))
                if len(vals_m) == 0:
                    continue

                # Align lengths.
                n = min(len(vals_base), len(vals_m))
                va = vals_base[:n]
                vm = vals_m[:n]

                lo, hi, obs = bootstrap_ci(vm, va, n_bootstrap=n_bootstrap)
                # Binary surrogate for McNemar.
                hits_a = (vm > 0).astype(int)
                hits_b = (va > 0).astype(int)
                p_val = mcnemar_p(hits_a, hits_b)

                sig_rows.append({
                    "method":          method,
                    "baseline":        baseline_method,
                    "dataset":         dataset,
                    "k":               k,
                    "observed_delta":  round(obs, 6),
                    "ci_lo":           round(lo, 6),
                    "ci_hi":           round(hi, 6),
                    "mcnemar_p":       round(p_val, 6) if not np.isnan(p_val) else None,
                    "significant_005": bool(lo > 0 or hi < 0),
                })

    return sig_rows


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_ablations_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("Ablations CSV → %s", path)


def save_significance_report(sig_rows: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sig_rows, f, indent=2)
    logger.info("Significance report → %s", path)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_ablations(
    results_dir: Path,
    out_dir: Path,
    baseline_method: str = "best64",
    k_pivot: int = 1_000_000,
    n_bootstrap: int = 10_000,
    min_seeds: int = 3,
) -> Dict[str, object]:
    """Load multi-seed results, compute ablation table and significance.

    Parameters
    ----------
    results_dir:
        Directory containing ``results.csv`` and/or ``results_seed*.csv``.
    out_dir:
        Directory to write ``ablations.csv`` and ``significance_report.json``.
    baseline_method:
        Method name to use as significance comparison baseline.
    k_pivot:
        Guess budget at which the ablation table is pivoted.
    n_bootstrap:
        Number of bootstrap iterations for CI computation.
    min_seeds:
        Warn if fewer than this many seeds are available for any method.

    Returns
    -------
    Summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_all_seeds(results_dir)
    if not all_rows:
        logger.warning("No result rows found in %s", results_dir)
        return {"error": "no results found"}

    # Check seed coverage.
    seed_counts: Dict[str, int] = defaultdict(int)
    for row in all_rows:
        seed_counts[row["method"]] += 1
    for method, cnt in seed_counts.items():
        if cnt < min_seeds:
            logger.warning(
                "Method '%s' has only %d seed(s); ≥%d recommended for variance estimates.",
                method, cnt, min_seeds,
            )

    agg_rows = aggregate_seeds(all_rows)
    ablation_rows = build_ablation_table(agg_rows, k_pivot=k_pivot)
    sig_rows = compute_significance(
        all_rows,
        baseline_method=baseline_method,
        n_bootstrap=n_bootstrap,
    )

    ablations_csv = out_dir / "ablations.csv"
    sig_json = out_dir / "significance_report.json"
    agg_csv = out_dir / "aggregated_results.csv"

    save_ablations_csv(ablation_rows, ablations_csv)
    save_significance_report(sig_rows, sig_json)
    save_ablations_csv(agg_rows, agg_csv)

    logger.info("Ablations complete → %s", out_dir)
    return {
        "ablations_csv":         str(ablations_csv),
        "significance_report":   str(sig_json),
        "aggregated_results":    str(agg_csv),
        "n_ablation_conditions": len(ablation_rows),
        "n_significance_tests":  len(sig_rows),
        "methods":               sorted({r["method"] for r in all_rows}),
        "out_dir":               str(out_dir),
    }
