"""Phase 8 — Hit@k evaluation harness.

Follows the frozen protocol in configs/protocol.yaml exactly:

    Hit@k = |dedup(candidates)[:k] ∩ test_plaintexts| / |test_plaintexts|

Candidate generation
--------------------
    hashcat --stdout -r <ruleset> <base_wordlist> --quiet

Outputs (in *out_dir*)
-----------------------
results.csv                 method, dataset, k, hit_rate, seed (one row per (method,k))
guessing_curve.png          Hit@k vs k (log-x) for all methods
targeted_results.csv        per-user and aggregate targeted evaluation
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwrules.config import load_config, load_protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def stream_candidates(
    ruleset_path: Path,
    wordlist_path: Path,
    hashcat_bin: str = "hashcat",
) -> Iterator[str]:
    """Yield deduplicated candidates from ``hashcat --stdout``."""
    cmd = [
        hashcat_bin, "--stdout",
        "-r", str(ruleset_path),
        str(wordlist_path),
        "--quiet",
    ]
    seen: Set[str] = set()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding="utf-8",
        errors="replace",
    )
    try:
        for raw_line in proc.stdout:
            word = raw_line.rstrip("\n")
            if word and word not in seen:
                seen.add(word)
                yield word
    finally:
        proc.stdout.close()
        proc.wait()


def generate_candidates(
    ruleset_path: Path,
    wordlist_path: Path,
    hashcat_bin: str = "hashcat",
    max_candidates: int = 10_000_000,
) -> List[str]:
    """Return up to *max_candidates* deduplicated candidates (order preserved)."""
    candidates: List[str] = []
    for cand in stream_candidates(ruleset_path, wordlist_path, hashcat_bin):
        candidates.append(cand)
        if len(candidates) >= max_candidates:
            break
    return candidates


def merge_rulesets_to_tmp(paths: List[Path]) -> str:
    """Concatenate multiple .rule files into one temp file; return its path."""
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".rule", delete=False, encoding="utf-8"
    )
    for p in paths:
        with open(p, encoding="utf-8") as f:
            tf.write(f.read())
        tf.write("\n")
    tf.close()
    return tf.name


# ---------------------------------------------------------------------------
# Hit@k computation
# ---------------------------------------------------------------------------

def hit_at_k(
    candidates: List[str],
    test_set: Set[str],
    k_values: List[int],
) -> Dict[int, float]:
    """Compute Hit@k for each k.

    Hit@k = |set(candidates[:k]) ∩ test_set| / |test_set|

    *candidates* are expected to be in generation order; this function tracks the
    set of matched test passwords so duplicate candidates never double-count (it
    is correct even if *candidates* contains duplicates).
    """
    if not test_set:
        return {k: 0.0 for k in k_values}

    n = len(test_set)
    sorted_ks = sorted(k_values)
    results: Dict[int, float] = {}
    matched: Set[str] = set()
    prev_k = 0
    cand_iter = iter(candidates)

    for k in sorted_ks:
        for _ in range(k - prev_k):
            try:
                c = next(cand_iter)
            except StopIteration:
                break
            if c in test_set:
                matched.add(c)
        results[k] = len(matched) / n
        prev_k = k

    return results


def load_test_set(test_path: Path) -> Set[str]:
    """Load a plaintext file (one password per line) as a set."""
    passwords: Set[str] = set()
    with open(test_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            pw = line.rstrip("\n")
            if pw:
                passwords.add(pw)
    return passwords


# ---------------------------------------------------------------------------
# Evaluate a single method
# ---------------------------------------------------------------------------

def evaluate_method(
    method_name: str,
    ruleset_path: Path,
    wordlist_path: Path,
    test_path: Path,
    k_values: List[int],
    seed: int = 1337,
    dataset_name: str = "default",
    hashcat_bin: str = "hashcat",
) -> List[Dict]:
    """Evaluate one (method, ruleset) pair against the frozen test set.

    Returns a list of dicts: method, dataset, k, hit_rate, seed.
    """
    if not shutil.which(hashcat_bin):
        logger.warning("Hashcat not found at '%s'; skipping %s.", hashcat_bin, method_name)
        return []

    logger.info("Evaluating %s …", method_name)
    max_k = max(k_values)
    candidates = generate_candidates(ruleset_path, wordlist_path, hashcat_bin, max_k)
    test_set = load_test_set(test_path)

    logger.info(
        "%s: %d candidates  |test|=%d", method_name, len(candidates), len(test_set)
    )

    scores = hit_at_k(candidates, test_set, k_values)
    return [
        {
            "method":   method_name,
            "dataset":  dataset_name,
            "k":        k,
            "hit_rate": scores[k],
            "seed":     seed,
        }
        for k in k_values
    ]


def evaluate_method_from_lists(
    method_name: str,
    candidates: List[str],
    test_set: Set[str],
    k_values: List[int],
    seed: int = 1337,
    dataset_name: str = "default",
) -> List[Dict]:
    """Evaluate when candidates are already in memory (no hashcat call)."""
    scores = hit_at_k(candidates, test_set, k_values)
    return [
        {
            "method":   method_name,
            "dataset":  dataset_name,
            "k":        k,
            "hit_rate": scores[k],
            "seed":     seed,
        }
        for k in k_values
    ]


# ---------------------------------------------------------------------------
# Complementarity
# ---------------------------------------------------------------------------

def evaluate_complementarity(
    ruleset_a: Path,
    ruleset_b: Path,
    wordlist_path: Path,
    test_path: Path,
    k_values: List[int],
    label: str = "LLM+best64",
    seed: int = 1337,
    dataset_name: str = "default",
    hashcat_bin: str = "hashcat",
) -> List[Dict]:
    """Evaluate the union of two rulesets (merged into a temp file)."""
    merged_tmp = merge_rulesets_to_tmp([ruleset_a, ruleset_b])
    try:
        return evaluate_method(
            method_name=label,
            ruleset_path=Path(merged_tmp),
            wordlist_path=wordlist_path,
            test_path=test_path,
            k_values=k_values,
            seed=seed,
            dataset_name=dataset_name,
            hashcat_bin=hashcat_bin,
        )
    finally:
        os.unlink(merged_tmp)


# ---------------------------------------------------------------------------
# Targeted evaluation
# ---------------------------------------------------------------------------

def evaluate_targeted(
    targeted_rules_dir: Path,
    target_users_path: Path,
    wordlist_path: Path,
    k_values: List[int],
    hashcat_bin: str = "hashcat",
    seed: int = 1337,
    dataset_name: str = "default",
) -> Tuple[List[Dict], List[Dict]]:
    """Targeted evaluation: per-user Hit@k using each user's dedicated rule file.

    Each line of *target_users_path* is a JSON object with:
        user_id, password (the held-out password for that user)

    The rule file for user_id is ``<targeted_rules_dir>/<user_id>.rule``.

    Returns ``(per_user_rows, aggregate_rows)``.
    """
    if not target_users_path.exists():
        logger.warning("target_users_path not found: %s", target_users_path)
        return [], []

    test_users = []
    with open(target_users_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    test_users.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not test_users:
        return [], []

    per_user_rows: List[Dict] = []
    max_k = max(k_values)

    for user in test_users:
        user_id = str(user.get("user_id", ""))
        password = user.get("password", "")
        rule_file = targeted_rules_dir / f"{user_id}.rule"

        if not rule_file.exists() or not password:
            continue

        candidates = generate_candidates(rule_file, wordlist_path, hashcat_bin, max_k)
        scores = hit_at_k(candidates, {password}, k_values)

        for k in k_values:
            per_user_rows.append({
                "user_id":  user_id,
                "k":        k,
                "hit":      int(scores[k] > 0),
                "dataset":  dataset_name,
                "seed":     seed,
            })

    # Aggregate across users.
    aggregate_rows: List[Dict] = []
    for k in k_values:
        rows_k = [r for r in per_user_rows if r["k"] == k]
        if not rows_k:
            continue
        hit_rate = sum(r["hit"] for r in rows_k) / len(rows_k)
        aggregate_rows.append({
            "method":   "LLM-targeted",
            "dataset":  dataset_name,
            "k":        k,
            "hit_rate": hit_rate,
            "seed":     seed,
        })

    return per_user_rows, aggregate_rows


# ---------------------------------------------------------------------------
# I/O — result CSV
# ---------------------------------------------------------------------------

_RESULT_FIELDS = ["method", "dataset", "k", "hit_rate", "seed"]


def append_results(rows: List[Dict], csv_path: Path) -> None:
    """Append rows to the results CSV (creates header if file is new)."""
    if not rows:
        return
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def load_results_csv(csv_path: Path) -> List[Dict]:
    """Load a results CSV as a list of dicts with typed fields."""
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["k"] = int(row["k"])
            row["hit_rate"] = float(row["hit_rate"])
            row["seed"] = int(row.get("seed") or 1337)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Guessing-number curve
# ---------------------------------------------------------------------------

def save_guessing_curve(rows: List[Dict], out_path: Path) -> None:
    """Plot Hit@k vs k (log-x) for every method."""
    methods: Dict[str, Dict[int, float]] = {}
    for row in rows:
        m = row["method"]
        methods.setdefault(m, {})[int(row["k"])] = float(row["hit_rate"])

    fig, ax = plt.subplots(figsize=(9, 6))
    for method, kv in sorted(methods.items()):
        ks = sorted(kv.keys())
        rates = [kv[k] for k in ks]
        ax.plot(ks, rates, marker="o", label=method, linewidth=1.8, markersize=4)

    ax.set_xscale("log")
    ax.set_xlabel("Guess budget k (log scale)")
    ax.set_ylabel("Hit@k")
    ax.set_title("Guessing-number curve")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Guessing curve → %s", out_path)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_eval(
    wordlist_path: Path,
    test_path: Path,
    out_dir: Path,
    llm_untargeted_rule: Optional[Path] = None,
    llm_targeted_rule: Optional[Path] = None,
    llm_filtered_rule: Optional[Path] = None,
    best64_rule: Optional[Path] = None,
    ruleforge_rule: Optional[Path] = None,
    targeted_rules_dir: Optional[Path] = None,
    target_users_path: Optional[Path] = None,
    hashcat_bin: str = "hashcat",
    seed: int = 1337,
    dataset_name: str = "default",
    config_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Run all evaluation methods and write results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    targeted_csv = out_dir / "targeted_results.csv"

    proto = load_config(config_path) if config_path else load_protocol()
    k_values: List[int] = [
        int(k) for k in proto.get(
            "guess_budget", [10, 100, 1000, 10000, 100000, 1000000, 10000000]
        )
    ]

    all_rows: List[Dict] = []

    def _eval(name: str, rule_path: Optional[Path]) -> None:
        if rule_path and rule_path.exists():
            rows = evaluate_method(
                name, rule_path, wordlist_path, test_path,
                k_values, seed, dataset_name, hashcat_bin,
            )
            all_rows.extend(rows)
        elif rule_path:
            logger.warning("Rule file not found: %s — skipping %s.", rule_path, name)

    _eval("LLM-untargeted", llm_untargeted_rule)
    _eval("LLM-targeted", llm_targeted_rule)
    _eval("LLM-filtered", llm_filtered_rule)
    _eval("best64", best64_rule)
    _eval("RuleForge", ruleforge_rule)

    # Complementarity.
    if (
        llm_untargeted_rule and best64_rule
        and llm_untargeted_rule.exists() and best64_rule.exists()
    ):
        comp_rows = evaluate_complementarity(
            llm_untargeted_rule, best64_rule,
            wordlist_path, test_path, k_values,
            label="LLM+best64", seed=seed, dataset_name=dataset_name,
            hashcat_bin=hashcat_bin,
        )
        all_rows.extend(comp_rows)

    append_results(all_rows, results_csv)

    # Targeted evaluation.
    if targeted_rules_dir and target_users_path:
        per_user, agg = evaluate_targeted(
            targeted_rules_dir, target_users_path, wordlist_path,
            k_values, hashcat_bin, seed, dataset_name,
        )
        # Persist the targeted aggregate to results.csv (it is computed after the
        # first append_results above, so it needs its own write or it is lost).
        append_results(agg, results_csv)
        all_rows.extend(agg)
        if per_user:
            fields = ["user_id", "k", "hit", "dataset", "seed"]
            with open(targeted_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(per_user)

    if all_rows:
        save_guessing_curve(all_rows, out_dir / "guessing_curve.png")

    logger.info("Evaluation complete → %s", results_csv)
    return {
        "results_csv": str(results_csv),
        "n_rows":      len(all_rows),
        "methods":     sorted({r["method"] for r in all_rows}),
        "k_values":    k_values,
        "out_dir":     str(out_dir),
    }
