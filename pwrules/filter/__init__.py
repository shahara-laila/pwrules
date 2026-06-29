"""Phase 7 — 3-stage rule filtering funnel.

Stage 1 — Syntax check
    Each rule is validated with the Python rule applier (fast, no subprocess).
    A rule is invalid if it cannot be parsed or produces empty output.
    A rule is a no-op if it leaves EVERY probe word unchanged.
    When hashcat is available, a configurable random sample is also cross-checked
    against real ``hashcat --stdout`` and any divergences are flagged.

Stage 2 — Deduplication
    Exact duplicates are removed first (set lookup, O(1)).
    Semantically equivalent rules — those that produce identical output on every
    probe word — are collapsed to the first-seen representative via fingerprinting.

Stage 3 — Effectiveness ranking (optional, validation set only)
    Each rule is applied to the base wordlist via ``hashcat --stdout``;
    candidates that appear in the VALIDATION set (never the test set) are counted.
    Rules are ranked by hit count descending; a configurable ``top_k`` cutoff is
    applied to produce the final filtered rule file.

Outputs (all in *out_dir*)
--------------------------
<stem>_filtered.rule    Filtered version of each input rule file.
filter_funnel.csv       generated → valid → unique → effective counts per file.
filter_funnel.png       Stacked bar visualisation of the funnel.
"""

from __future__ import annotations

import csv
import logging
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwrules.ruleextract.applier import apply_rule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Diverse probe words used for no-op detection and semantic fingerprinting.
_PROBE_WORDS: List[str] = [
    "password", "dragon", "letmein", "hello", "shadow",
    "monkey", "sunshine", "football", "master", "iloveyou",
    "princess", "michael", "superman", "batman", "welcome",
    "charlie", "donald", "jessica", "thomas", "ranger",
]

# Characters that can appear as Hashcat rule function tokens.
_VALID_FN_CHARS: Set[str] = set(":lucCtrdpf{}[]kKq$^TDzZyY@+-LRpsio*x")

# Parameter-count expectations per function token.
_PARAM_COUNT: Dict[str, int] = {
    **{c: 0 for c in ":lucCtrdpf{}[]kKq"},
    **{c: 1 for c in "$^TDzZyY@+-LRp"},
    **{c: 2 for c in "sio*x"},
}


# ---------------------------------------------------------------------------
# Stage 1 — Syntax validation
# ---------------------------------------------------------------------------

def _is_parseable(rule: str) -> bool:
    """Return True if every token in *rule* is a recognised Hashcat function."""
    if not rule or rule.isspace():
        return False
    for tok in rule.split():
        if not tok:
            continue
        fn = tok[0]
        if fn not in _VALID_FN_CHARS:
            return False
        params = tok[1:]
        expected = _PARAM_COUNT.get(fn, 0)
        if len(params) != expected:
            return False
    return True


def _is_noop(rule: str, probe_words: List[str] = _PROBE_WORDS) -> bool:
    """Return True if the rule leaves ALL probe words unchanged."""
    for word in probe_words:
        try:
            if apply_rule(word, rule) != word:
                return False
        except Exception:
            return True  # erroring rule is treated as invalid
    return True


def syntax_check(
    rules: List[str],
    probe_words: List[str] = _PROBE_WORDS,
    hashcat_bin: str = "hashcat",
    hashcat_sample: int = 200,
    seed: int = 1337,
) -> Tuple[List[str], int, int]:
    """Stage 1: remove syntactically invalid and no-op rules.

    Parameters
    ----------
    rules:
        Input rule strings (one per element).
    probe_words:
        Words used for no-op detection.
    hashcat_bin:
        Path to the hashcat binary.
    hashcat_sample:
        Number of rules to cross-check against real hashcat (0 = skip).
    seed:
        RNG seed for sampling.

    Returns
    -------
    valid_rules, n_invalid, n_noop
    """
    valid: List[str] = []
    n_invalid = 0
    n_noop = 0

    for rule in rules:
        if not _is_parseable(rule):
            n_invalid += 1
            continue
        if _is_noop(rule, probe_words):
            n_noop += 1
            continue
        valid.append(rule)

    logger.info(
        "Syntax check: %d → %d valid  (%d invalid, %d no-op)",
        len(rules), len(valid), n_invalid, n_noop,
    )

    # Optional hashcat cross-check on a random sample.
    if hashcat_sample > 0 and shutil.which(hashcat_bin) and valid:
        rng = random.Random(seed)
        sample = rng.sample(valid, min(hashcat_sample, len(valid)))
        divergences = _hashcat_crosscheck(sample, hashcat_bin, probe_words[0])
        if divergences:
            logger.warning(
                "Hashcat cross-check: %d/%d sampled rules diverge from Python applier. "
                "Review applier.py for edge cases.",
                divergences, len(sample),
            )

    return valid, n_invalid, n_noop


def _hashcat_crosscheck(
    rules: List[str],
    hashcat_bin: str,
    probe_word: str,
) -> int:
    """Return the number of rules where Python output ≠ hashcat output."""
    divergences = 0
    for rule in rules:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rule", delete=False) as rf:
            rf.write(rule + "\n")
            rf_path = rf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as wf:
            wf.write(probe_word + "\n")
            wf_path = wf.name
        try:
            res = subprocess.run(
                [hashcat_bin, "--stdout", "-r", rf_path, wf_path, "--quiet"],
                capture_output=True, text=True, timeout=10,
            )
            hc_out = res.stdout.strip()
            py_out = apply_rule(probe_word, rule)
            if hc_out != py_out:
                divergences += 1
        except Exception:
            pass
        finally:
            os.unlink(rf_path)
            os.unlink(wf_path)
    return divergences


# ---------------------------------------------------------------------------
# Stage 2 — Semantic deduplication
# ---------------------------------------------------------------------------

def _fingerprint(rule: str, probe_words: List[str]) -> Tuple[str, ...]:
    """Tuple of outputs when the rule is applied to each probe word."""
    outs = []
    for w in probe_words:
        try:
            outs.append(apply_rule(w, rule))
        except Exception:
            outs.append("")
    return tuple(outs)


def semantic_dedup(
    rules: List[str],
    probe_words: List[str] = _PROBE_WORDS,
) -> Tuple[List[str], int]:
    """Stage 2: remove exact duplicates then semantically equivalent rules.

    Returns ``(unique_rules, n_removed)``.
    """
    # Pass 1: exact dedup (preserving first occurrence).
    seen_exact: Set[str] = set()
    exact_deduped: List[str] = []
    for rule in rules:
        if rule not in seen_exact:
            seen_exact.add(rule)
            exact_deduped.append(rule)

    # Pass 2: semantic dedup via fingerprinting.
    seen_fp: Set[Tuple[str, ...]] = set()
    unique: List[str] = []
    for rule in exact_deduped:
        fp = _fingerprint(rule, probe_words)
        if fp not in seen_fp:
            seen_fp.add(fp)
            unique.append(rule)

    n_removed = len(rules) - len(unique)
    logger.info(
        "Semantic dedup: %d → %d unique  (%d removed)",
        len(rules), len(unique), n_removed,
    )
    return unique, n_removed


# ---------------------------------------------------------------------------
# Stage 3 — Effectiveness ranking (validation set only)
# ---------------------------------------------------------------------------

def rank_by_effectiveness(
    rules: List[str],
    val_path: Path,
    base_wordlist_path: Path,
    hashcat_bin: str = "hashcat",
    top_k: Optional[int] = None,
    timeout: int = 120,
) -> Tuple[List[str], List[int]]:
    """Stage 3: rank rules by hit count on the VALIDATION set.

    Writes all rules to a temp file, runs ``hashcat --stdout`` against the
    base wordlist, and counts how many outputs appear in the validation set.
    Only the validation set is used here — the test set is never touched.

    Returns ``(ranked_rules, hit_counts)``.
    """
    val_set: Set[str] = set()
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            pw = line.rstrip("\n")
            if pw:
                val_set.add(pw)

    if not shutil.which(hashcat_bin):
        logger.warning(
            "Hashcat not found — skipping effectiveness ranking; "
            "returning rules in original order."
        )
        # Still honour top_k so the funnel/output counts are consistent with the
        # normal path (which truncates to top_k).
        if top_k is not None:
            return rules[:top_k], [0] * min(top_k, len(rules))
        return rules, [0] * len(rules)

    # Write base wordlist and run hashcat per rule.
    # To keep it tractable, evaluate each rule independently.
    hit_counts: List[int] = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as wf, \
            open(base_wordlist_path, encoding="utf-8") as bw:
        shutil.copyfileobj(bw, wf)
        wf_path = wf.name

    try:
        for rule in rules:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".rule", delete=False) as rf:
                rf.write(rule + "\n")
                rf_path = rf.name
            try:
                res = subprocess.run(
                    [hashcat_bin, "--stdout", "-r", rf_path, wf_path, "--quiet"],
                    capture_output=True, text=True, timeout=timeout,
                )
                # Count DISTINCT validation passwords this rule recovers (not raw
                # candidate occurrences) so duplicate candidates don't inflate rank.
                produced = {line.rstrip("\n") for line in res.stdout.splitlines()}
                hit_counts.append(len(produced & val_set))
            except Exception:
                hit_counts.append(0)
            finally:
                os.unlink(rf_path)
    finally:
        os.unlink(wf_path)

    # Sort by hits descending.
    paired = sorted(zip(hit_counts, rules), reverse=True)
    sorted_counts = [c for c, _ in paired]
    sorted_rules = [r for _, r in paired]

    if top_k is not None:
        sorted_rules = sorted_rules[:top_k]
        sorted_counts = sorted_counts[:top_k]

    logger.info(
        "Effectiveness ranking: top rule hits=%d  bottom=%d",
        sorted_counts[0] if sorted_counts else 0,
        sorted_counts[-1] if sorted_counts else 0,
    )
    return sorted_rules, sorted_counts


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_rules(path: Path) -> List[str]:
    """Load a .rule file, stripping blank lines and comments (lines starting #)."""
    rules: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rule = line.rstrip("\n")
            if rule and not rule.startswith("#"):
                rules.append(rule)
    return rules


def write_rules(rules: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rule in rules:
            f.write(rule + "\n")
    logger.info("Written %d rules → %s", len(rules), path)


# ---------------------------------------------------------------------------
# Funnel visualisation
# ---------------------------------------------------------------------------

def save_funnel(funnel_rows: List[Dict], out_dir: Path) -> None:
    """Write filter_funnel.csv and filter_funnel.png."""
    csv_path = out_dir / "filter_funnel.csv"
    fieldnames = ["file", "generated", "valid", "unique", "effective"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(funnel_rows)

    # Stacked bar chart.
    labels = [r["file"] for r in funnel_rows]
    generated = [r["generated"] for r in funnel_rows]
    valid     = [r["valid"]     for r in funnel_rows]
    unique    = [r["unique"]    for r in funnel_rows]
    effective = [r["effective"] for r in funnel_rows]

    # Nested funnel: each stage is a subset of the previous, so overlaying wide→
    # narrow bars reads as a funnel. Draw widest first and annotate counts.
    import numpy as np
    x = np.arange(len(labels))
    stages = [
        ("generated", generated, "#aec6e8"),
        ("valid",     valid,     "#4c72b0"),
        ("unique",    unique,    "#55a868"),
        ("effective", effective, "#c44e52"),
    ]
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(labels)), 5))
    width = 0.6
    for name, vals, color in stages:
        bars = ax.bar(x, vals, width=width, label=name, color=color, zorder=2)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:,}",
                    ha="center", va="bottom", fontsize=8, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Rule count")
    ax.set_title("Rule-filtering funnel (generated → valid → unique → effective)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    fig.savefig(out_dir / "filter_funnel.png", dpi=150)
    plt.close(fig)
    logger.info("Filter funnel saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def filter_rules(
    rule_files: List[Path],
    out_dir: Path,
    val_path: Optional[Path] = None,
    base_wordlist_path: Optional[Path] = None,
    hashcat_bin: str = "hashcat",
    hashcat_sample: int = 200,
    effectiveness_ranking: bool = False,
    top_k: Optional[int] = None,
    probe_words: List[str] = _PROBE_WORDS,
    seed: int = 1337,
) -> Dict[str, object]:
    """Full Phase 7 pipeline: syntax → dedup → (optional) effectiveness ranking.

    Parameters
    ----------
    rule_files:
        List of .rule files to process (each gets its own output file).
    out_dir:
        Output directory.
    val_path:
        Validation password file (required for effectiveness ranking).
    base_wordlist_path:
        Base wordlist (required for effectiveness ranking).
    hashcat_bin:
        Hashcat binary path.
    hashcat_sample:
        How many rules to cross-check against real hashcat in Stage 1 (0 = skip).
    effectiveness_ranking:
        Whether to run Stage 3 (requires val_path + base_wordlist_path).
    top_k:
        Keep only the top-k most effective rules (Stage 3 only).
    probe_words:
        Words used for no-op detection and semantic fingerprinting.
    seed:
        RNG seed.

    Returns
    -------
    dict with keys: ``funnel``, ``filtered_files``, ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    funnel_rows: List[Dict] = []
    filtered_files: Dict[str, Path] = {}

    for rule_file in rule_files:
        stem = rule_file.stem
        logger.info("Processing %s …", rule_file)

        raw_rules = load_rules(rule_file)
        n_generated = len(raw_rules)

        # Stage 1.
        valid, n_invalid, n_noop = syntax_check(
            raw_rules, probe_words, hashcat_bin, hashcat_sample, seed
        )

        # Stage 2.
        unique, _ = semantic_dedup(valid, probe_words)

        # Stage 3 (optional).
        if effectiveness_ranking and val_path and base_wordlist_path:
            effective, hit_counts = rank_by_effectiveness(
                unique, val_path, base_wordlist_path, hashcat_bin, top_k
            )
        else:
            effective = unique
            hit_counts = []

        n_effective = len(effective)

        # Write filtered rule file.
        out_path = out_dir / f"{stem}_filtered.rule"
        write_rules(effective, out_path)
        filtered_files[stem] = out_path

        funnel_rows.append({
            "file":      stem,
            "generated": n_generated,
            "valid":     len(valid),
            "unique":    len(unique),
            "effective": n_effective,
        })

        logger.info(
            "%s: %d → %d (valid) → %d (unique) → %d (effective)",
            stem, n_generated, len(valid), len(unique), n_effective,
        )

    save_funnel(funnel_rows, out_dir)

    return {
        "funnel":         funnel_rows,
        "filtered_files": {k: str(v) for k, v in filtered_files.items()},
        "out_dir":        str(out_dir),
    }
