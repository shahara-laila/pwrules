"""Baseline rule-set runners for Phase 8.

best64
------
Bundled with hashcat at /usr/share/hashcat/rules/best64.rule (Linux/Kaggle).
Falls back to searching common install locations.

RuleForge
---------
Clustering-based rule generator (public GitHub repo). Supports three variants:
  - MDBSCAN  (requires .NET SDK 7.0 — available on Kaggle)
  - DBSCAN   (Python-based fallback)
  - HAC      (Python-based fallback)
The runner auto-detects which variant is feasible and logs what ran.

Usage
-----
    from pwrules.eval.baselines import get_best64_rule, run_ruleforge
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# best64
# ---------------------------------------------------------------------------

_BEST64_SEARCH_PATHS: List[str] = [
    "/usr/share/hashcat/rules/best64.rule",
    "/usr/local/share/hashcat/rules/best64.rule",
    "/opt/hashcat/rules/best64.rule",
]


def get_best64_rule() -> Optional[Path]:
    """Return the path to best64.rule, or None if not found."""
    for p in _BEST64_SEARCH_PATHS:
        if Path(p).exists():
            logger.info("Found best64 at %s", p)
            return Path(p)

    # Also try finding it next to the hashcat binary.
    hc = shutil.which("hashcat")
    if hc:
        candidate = Path(hc).parent / "rules" / "best64.rule"
        if candidate.exists():
            return candidate

    logger.warning(
        "best64.rule not found in standard locations. "
        "Install hashcat or pass the path manually."
    )
    return None


# ---------------------------------------------------------------------------
# RuleForge
# ---------------------------------------------------------------------------

_RULEFORGE_REPO = "https://github.com/TheWorkingDev/RuleForge.git"

# Variant names accepted by the run_ruleforge function.
_VARIANT_AUTO   = "auto"
_VARIANT_MDBSCAN = "mdbscan"
_VARIANT_DBSCAN  = "dbscan"
_VARIANT_HAC     = "hac"


def _check_dotnet() -> bool:
    """Return True if .NET SDK ≥ 7.0 is available."""
    dotnet = shutil.which("dotnet")
    if not dotnet:
        return False
    try:
        res = subprocess.run(
            [dotnet, "--version"], capture_output=True, text=True, timeout=10
        )
        version_str = res.stdout.strip()
        major = int(version_str.split(".")[0])
        return major >= 7
    except Exception:
        return False


def clone_ruleforge(dest: Path) -> bool:
    """Clone the RuleForge repo to *dest*. Return True on success."""
    if dest.exists():
        logger.info("RuleForge already cloned at %s", dest)
        return True
    git = shutil.which("git")
    if not git:
        logger.warning("git not found; cannot clone RuleForge.")
        return False
    try:
        res = subprocess.run(
            [git, "clone", "--depth", "1", _RULEFORGE_REPO, str(dest)],
            capture_output=True, text=True, timeout=120,
        )
        if res.returncode != 0:
            logger.warning("RuleForge clone failed:\n%s", res.stderr)
            return False
        logger.info("RuleForge cloned to %s", dest)
        return True
    except Exception as exc:
        logger.warning("RuleForge clone error: %s", exc)
        return False


def _run_mdbscan(
    ruleforge_dir: Path,
    wordlist_path: Path,
    out_rule: Path,
    n_rules: int = 1000,
) -> bool:
    """Run MDBSCAN variant via `dotnet run` inside the RuleForge project.

    Returns True on success.
    """
    dotnet = shutil.which("dotnet")
    if not dotnet or not _check_dotnet():
        return False

    # Locate the .csproj file.
    csproj_files = list(ruleforge_dir.rglob("*.csproj"))
    if not csproj_files:
        logger.warning("No .csproj found in %s", ruleforge_dir)
        return False

    project = csproj_files[0].parent

    cmd = [
        dotnet, "run",
        "--project", str(project),
        "--",
        "--wordlist", str(wordlist_path),
        "--output", str(out_rule),
        "--method", "mdbscan",
        "--rules", str(n_rules),
    ]
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(project),
        )
        if res.returncode != 0:
            logger.warning("RuleForge MDBSCAN failed:\n%s", res.stderr[-2000:])
            return False
        logger.info("RuleForge MDBSCAN → %s", out_rule)
        return out_rule.exists()
    except Exception as exc:
        logger.warning("RuleForge MDBSCAN error: %s", exc)
        return False


def _run_python_clustering(
    wordlist_path: Path,
    out_rule: Path,
    variant: str = "dbscan",
    n_rules: int = 1000,
) -> bool:
    """Fallback Python clustering baseline (DBSCAN or HAC via scikit-learn).

    Extracts n-gram character features from the wordlist, clusters passwords,
    and writes representative mangling rules for each cluster.  This is a
    simplified stand-in — it does not replicate RuleForge's full algorithm but
    provides a non-trivial clustering-based baseline.
    """
    try:
        import numpy as np
        from sklearn.cluster import DBSCAN, AgglomerativeClustering
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import normalize
    except ImportError as exc:
        logger.warning("scikit-learn not installed; cannot run %s baseline: %s", variant, exc)
        return False

    # Load wordlist sample (cap at 50k for memory).
    words: List[str] = []
    with open(wordlist_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.rstrip("\n")
            if 4 <= len(w) <= 20:
                words.append(w)
            if len(words) >= 50_000:
                break

    if len(words) < 50:
        logger.warning("Wordlist too small for clustering baseline.")
        return False

    # Char-level n-gram features.
    vec = CountVectorizer(analyzer="char", ngram_range=(2, 3), max_features=500)
    X = normalize(vec.fit_transform(words).toarray().astype(float))

    if variant == "dbscan":
        labels = DBSCAN(eps=0.5, min_samples=5, metric="cosine", n_jobs=-1).fit_predict(X)
    else:  # HAC
        n_clusters = min(n_rules, len(words) // 10)
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)

    # Derive a REAL representative mangling rule per cluster (RuleForge-style):
    # for each cluster, infer the validated (base → password) rule of its members
    # and keep the most common one. Emitting a single trivial "c" for every
    # cluster would artificially weaken the baseline (forbidden).
    from collections import Counter

    from pwrules.ruleextract.applier import apply_rule
    from pwrules.ruleextract.extractor import (
        _reverse_leet, _strip_outer_nonalpha, infer_rule,
    )

    def _derive_rule(pw: str) -> Optional[str]:
        _, core, _ = _strip_outer_nonalpha(pw)
        if not core:
            return None
        base = _reverse_leet(core).lower()
        if not base:
            return None
        rule = infer_rule(base, pw)
        if rule is None:
            return None
        rule = rule.strip() or ":"
        # Keep only rules that provably regenerate the password.
        return rule if apply_rule(base, rule) == pw else None

    cluster_ids = sorted(set(labels) - {-1})
    rules: List[str] = []
    for cid in cluster_ids:
        members = [words[i] for i, l in enumerate(labels) if l == cid]
        cluster_rules = Counter()
        for pw in members:
            r = _derive_rule(pw)
            if r:
                cluster_rules[r] += 1
        if cluster_rules:
            # Representative = most frequent valid rule in the cluster.
            rules.append(cluster_rules.most_common(1)[0][0])
        if len(rules) >= n_rules:
            break

    # Deduplicate while preserving order.
    rules = list(dict.fromkeys(rules))
    if not rules:
        logger.warning("%s baseline produced no valid rules.", variant)
        return False

    out_rule.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rule, "w", encoding="utf-8") as f:
        for r in rules:
            f.write(r + "\n")

    logger.info("Python %s baseline → %d rules → %s", variant, len(rules), out_rule)
    return True


def run_ruleforge(
    wordlist_path: Path,
    out_dir: Path,
    ruleforge_dir: Optional[Path] = None,
    variant: str = _VARIANT_AUTO,
    n_rules: int = 1000,
    clone: bool = True,
) -> Optional[Path]:
    """Generate a RuleForge baseline rule file.

    Parameters
    ----------
    wordlist_path:
        Base wordlist used to fit the clustering model.
    out_dir:
        Directory to write the output rule file.
    ruleforge_dir:
        Where to clone RuleForge (default: ``<out_dir>/ruleforge``).
    variant:
        ``"auto"`` (try MDBSCAN → DBSCAN → HAC), ``"mdbscan"``,
        ``"dbscan"``, or ``"hac"``.
    n_rules:
        Target number of output rules.
    clone:
        Whether to clone the repo if not already present.

    Returns
    -------
    Path to the generated rule file, or ``None`` if generation failed.
    """
    if ruleforge_dir is None:
        ruleforge_dir = out_dir / "ruleforge"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_rule = out_dir / "ruleforge.rule"

    if variant in (_VARIANT_AUTO, _VARIANT_MDBSCAN):
        # Try MDBSCAN first.
        if clone:
            clone_ruleforge(ruleforge_dir)

        if _check_dotnet() and ruleforge_dir.exists():
            if _run_mdbscan(ruleforge_dir, wordlist_path, out_rule, n_rules):
                logger.info("RuleForge variant used: MDBSCAN")
                return out_rule
            logger.warning("MDBSCAN failed; falling back to Python clustering.")
        else:
            if variant == _VARIANT_MDBSCAN:
                logger.warning(
                    "MDBSCAN requested but .NET SDK 7.0 is not available. "
                    "Install .NET SDK 7.0 or use variant='auto'."
                )
                return None
            logger.warning(
                ".NET SDK 7.0 not found; skipping MDBSCAN and falling back to DBSCAN."
            )

    if variant in (_VARIANT_AUTO, _VARIANT_DBSCAN):
        if _run_python_clustering(wordlist_path, out_rule, "dbscan", n_rules):
            logger.info("RuleForge variant used: DBSCAN (Python fallback)")
            return out_rule

    if variant in (_VARIANT_AUTO, _VARIANT_HAC):
        if _run_python_clustering(wordlist_path, out_rule, "hac", n_rules):
            logger.info("RuleForge variant used: HAC (Python fallback)")
            return out_rule

    logger.error(
        "All RuleForge variants failed. "
        "Ensure .NET SDK 7.0 or scikit-learn is installed."
    )
    return None
