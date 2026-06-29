"""Paper-ready research figures (matplotlib).

A single place for the charts a conference submission needs, beyond the inline
figures each phase already emits (length histogram, training curves, guessing
curve, filter funnel). Every function reads an artifact produced by the pipeline
and writes a high-DPI PNG; all are tolerant of missing inputs (they log and skip)
so a partial run still produces what it can.

Figures
-------
rule_op_distribution      which Hashcat operations the generated rules use
memorisation_breakdown    novel vs in-training rule fraction (generalisation)
top_rules                  most frequent generated rules
targeted_vs_untargeted    diversity of targeted vs untargeted generation
per_user_rule_counts      distribution of #rules generated per held-out user
hit_at_k_bars             Hit@k bar comparison across methods at one budget
complementarity           standalone vs union(LLM, best64) Hit@k
ablation_bars             ablation deltas with std error bars
pipeline_diagram          the Figure-1 schematic of the whole pipeline
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_DPI = 150
# Stable colour per logical method (others fall back to the cycle).
_METHOD_COLORS = {
    "best64": "#7f7f7f",
    "MDBSCAN": "#9467bd",
    "RuleForge": "#8c564b",
    "LLM-untargeted": "#1f77b4",
    "LLM-filtered": "#2ca02c",
    "LLM-targeted": "#17becf",
    "LLM+best64": "#d62728",
}


# ---------------------------------------------------------------------------
# Small loaders
# ---------------------------------------------------------------------------

def _load_rules(rule_file: Path) -> List[str]:
    rules: List[str] = []
    with open(rule_file, encoding="utf-8") as f:
        for line in f:
            r = line.rstrip("\n")
            if r and not r.startswith("#"):
                rules.append(r)
    return rules


def _load_csv(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _op_label(fn: str) -> str:
    return {
        "$": "$ append", "^": "^ prepend", "s": "s substitute",
        "c": "c capitalise", "u": "u uppercase", "l": "l lowercase",
        "t": "t toggle", "r": "r reverse", "d": "d duplicate",
        "{": "{ rotate", "}": "} rotate", "[": "[ del-first", "]": "] del-last",
    }.get(fn, fn)


# ---------------------------------------------------------------------------
# Generation / training figures
# ---------------------------------------------------------------------------

def rule_op_distribution(rule_file: Path, out_path: Path,
                         title: str = "Rule operation distribution") -> Optional[Path]:
    """Bar chart of how often each Hashcat operation appears in a rule set."""
    rule_file, out_path = Path(rule_file), Path(out_path)
    if not rule_file.exists():
        logger.warning("rule_op_distribution: %s missing — skipping.", rule_file)
        return None
    from pwrules.ruleextract.applier import tokenize_rule

    counts: Counter = Counter()
    for rule in _load_rules(rule_file):
        for fn, _ in tokenize_rule(rule):
            counts[fn] += 1
    if not counts:
        logger.warning("rule_op_distribution: no ops found in %s.", rule_file)
        return None

    items = counts.most_common()
    labels = [_op_label(fn) for fn, _ in items]
    vals = [c for _, c in items]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(items))))
    ax.barh(labels[::-1], vals[::-1], color="#1f77b4")
    ax.set_xlabel("Occurrences across rule set")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("rule_op_distribution → %s", out_path)
    return out_path


def memorisation_breakdown(report_json: Path, out_path: Path) -> Optional[Path]:
    """Novel vs in-training rule fraction from memorisation_report.json."""
    report_json, out_path = Path(report_json), Path(out_path)
    if not report_json.exists():
        logger.warning("memorisation_breakdown: %s missing — skipping.", report_json)
        return None
    rep = json.loads(report_json.read_text(encoding="utf-8"))
    novel = int(rep.get("n_novel", 0))
    in_train = int(rep.get("n_in_train", 0))
    if novel + in_train == 0:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [novel, in_train],
        labels=[f"Novel\n{novel}", f"In training\n{in_train}"],
        colors=["#2ca02c", "#d62728"], autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    ax.set_title("Generated rules: novel vs memorised")
    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("memorisation_breakdown → %s", out_path)
    return out_path


def top_rules(generation_stats_json: Path, out_path: Path, top_n: int = 20) -> Optional[Path]:
    """Horizontal bar of the most frequent generated rules."""
    p, out_path = Path(generation_stats_json), Path(out_path)
    if not p.exists():
        logger.warning("top_rules: %s missing — skipping.", p)
        return None
    stats = json.loads(p.read_text(encoding="utf-8"))
    top = stats.get("combined", {}).get("top_20_rules", [])[:top_n]
    if not top:
        return None
    labels = [str(r) for r, _ in top]
    vals = [int(c) for _, c in top]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(top))))
    ax.barh(labels[::-1], vals[::-1], color="#ff7f0e")
    ax.set_xlabel("Count")
    ax.set_title(f"Top {len(top)} generated rules")
    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("top_rules → %s", out_path)
    return out_path


def targeted_vs_untargeted(generation_stats_json: Path, out_path: Path) -> Optional[Path]:
    """Compare unique-rule counts for untargeted vs targeted generation."""
    p, out_path = Path(generation_stats_json), Path(out_path)
    if not p.exists():
        logger.warning("targeted_vs_untargeted: %s missing — skipping.", p)
        return None
    stats = json.loads(p.read_text(encoding="utf-8"))
    unt = stats.get("untargeted", {}).get("n_unique", 0)
    tgt = stats.get("targeted", {})
    tgt_total = tgt.get("n_total", 0)
    per_user = tgt.get("n_unique_per_user", {}) or {}
    tgt_unique = len(set().union(*[set() for _ in per_user])) if per_user else tgt_total
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Untargeted\n(unique)", "Targeted\n(total)"],
        [unt, tgt_total], color=["#1f77b4", "#17becf"],
    )
    for rect, v in zip(bars, [unt, tgt_total]):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
    ax.set_ylabel("Rule count")
    ax.set_title("Targeted vs untargeted generation")
    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("targeted_vs_untargeted → %s", out_path)
    return out_path


def per_user_rule_counts(generation_stats_json: Path, out_path: Path) -> Optional[Path]:
    """Histogram of unique rules generated per held-out target user."""
    p, out_path = Path(generation_stats_json), Path(out_path)
    if not p.exists():
        return None
    stats = json.loads(p.read_text(encoding="utf-8"))
    per_user = stats.get("targeted", {}).get("n_unique_per_user", {}) or {}
    if not per_user:
        logger.warning("per_user_rule_counts: no targeted per-user data — skipping.")
        return None
    counts = list(per_user.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=min(30, max(5, len(set(counts)))), color="#17becf", edgecolor="white")
    ax.set_xlabel("Unique rules per user")
    ax.set_ylabel("Number of users")
    ax.set_title("Per-user generated-rule counts (targeted)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("per_user_rule_counts → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Evaluation figures
# ---------------------------------------------------------------------------

def _results_by_method(results_csv: Path) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for row in _load_csv(results_csv):
        out.setdefault(row["method"], {})[int(row["k"])] = float(row["hit_rate"])
    return out


def hit_at_k_bars(results_csv: Path, out_path: Path, k: Optional[int] = None) -> Optional[Path]:
    """Bar chart comparing Hit@k across methods at a single budget *k*."""
    results_csv, out_path = Path(results_csv), Path(out_path)
    if not results_csv.exists():
        logger.warning("hit_at_k_bars: %s missing — skipping.", results_csv)
        return None
    by_method = _results_by_method(results_csv)
    if not by_method:
        return None
    all_ks = sorted({kk for kv in by_method.values() for kk in kv})
    if k is None:
        k = all_ks[-1]
    methods = sorted(by_method.keys(), key=lambda m: by_method[m].get(k, 0))
    vals = [by_method[m].get(k, 0.0) for m in methods]
    colors = [_METHOD_COLORS.get(m, "#1f77b4") for m in methods]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(methods))))
    bars = ax.barh(methods, vals, color=colors)
    for rect, v in zip(bars, vals):
        ax.text(v, rect.get_y() + rect.get_height() / 2, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel(f"Hit@{k:,}")
    ax.set_title(f"Hit@{k:,} by method")
    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("hit_at_k_bars → %s", out_path)
    return out_path


def complementarity(results_csv: Path, out_path: Path) -> Optional[Path]:
    """Guessing-curve comparison of LLM, best64, and their union."""
    results_csv, out_path = Path(results_csv), Path(out_path)
    if not results_csv.exists():
        return None
    by_method = _results_by_method(results_csv)
    wanted = [m for m in ("LLM-filtered", "LLM-untargeted", "best64", "LLM+best64") if m in by_method]
    if "LLM+best64" not in by_method:
        logger.warning("complementarity: no union row — skipping.")
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in wanted:
        kv = by_method[m]
        ks = sorted(kv)
        ax.plot(ks, [kv[k] for k in ks], marker="o", label=m,
                color=_METHOD_COLORS.get(m), linewidth=1.8, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Guess budget k (log scale)")
    ax.set_ylabel("Hit@k")
    ax.set_title("Complementarity: LLM ∪ best64")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("complementarity → %s", out_path)
    return out_path


def ablation_bars(ablations_csv: Path, out_path: Path) -> Optional[Path]:
    """Grouped bars of mean Hit@k for each ablation pair (with std error bars)."""
    ablations_csv, out_path = Path(ablations_csv), Path(out_path)
    if not ablations_csv.exists():
        logger.warning("ablation_bars: %s missing — skipping.", ablations_csv)
        return None
    rows = _load_csv(ablations_csv)
    if not rows:
        return None
    import numpy as np
    labels = [r.get("label", r.get("axis", "")) for r in rows]
    mean_a = [float(r.get("mean_a", 0) or 0) for r in rows]
    std_a = [float(r.get("std_a", 0) or 0) for r in rows]
    mean_b = [float(r.get("mean_b", 0) or 0) for r in rows]
    std_b = [float(r.get("std_b", 0) or 0) for r in rows]
    y = np.arange(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, max(3, 0.9 * len(labels))))
    ax.barh(y + h / 2, mean_a, h, xerr=std_a, label="A (condition)", color="#2ca02c", capsize=3)
    ax.barh(y - h / 2, mean_b, h, xerr=std_b, label="B (baseline)", color="#7f7f7f", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Hit@k (±1 std over seeds)")
    ax.set_title("Ablation comparison")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("ablation_bars → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Pipeline schematic (Figure 1)
# ---------------------------------------------------------------------------

def pipeline_diagram(out_path: Path) -> Path:
    """Render the end-to-end pipeline schematic (paper Figure 1)."""
    out_path = Path(out_path)
    stages = [
        ("Leaked\ncorpus", "#aec7e8"),
        ("Clean &\nsplit", "#aec7e8"),
        ("Rule\nextraction", "#98df8a"),
        ("Target\nconditioning", "#98df8a"),
        ("QLoRA\nfine-tune", "#ffbb78"),
        ("Rule\ngeneration", "#ffbb78"),
        ("Filter", "#ff9896"),
        ("Hashcat\nHit@k eval", "#c5b0d5"),
    ]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set_xlim(0, len(stages)); ax.set_ylim(0, 1); ax.axis("off")
    for i, (label, color) in enumerate(stages):
        ax.add_patch(plt.Rectangle((i + 0.08, 0.28), 0.84, 0.44,
                                   facecolor=color, edgecolor="#333", linewidth=1.2))
        ax.text(i + 0.5, 0.5, label, ha="center", va="center", fontsize=9)
        if i < len(stages) - 1:
            ax.annotate("", xy=(i + 1.06, 0.5), xytext=(i + 0.92, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))
    ax.set_title("pwrules pipeline: corpus → rules → fine-tuned LLM → evaluation", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("pipeline_diagram → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Convenience: generate everything available
# ---------------------------------------------------------------------------

def generate_all_figures(
    out_dir: Path,
    results_csv: Optional[Path] = None,
    ablations_csv: Optional[Path] = None,
    generation_stats_json: Optional[Path] = None,
    memorisation_json: Optional[Path] = None,
    rule_file: Optional[Path] = None,
) -> List[str]:
    """Render every figure for which the source artifact is available.

    Returns the list of produced file paths. Missing inputs are skipped (logged).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Optional[Path]] = [pipeline_diagram(out_dir / "fig_pipeline.png")]

    if rule_file:
        produced.append(rule_op_distribution(Path(rule_file), out_dir / "fig_rule_ops.png"))
    if memorisation_json:
        produced.append(memorisation_breakdown(Path(memorisation_json), out_dir / "fig_memorisation.png"))
    if generation_stats_json:
        gs = Path(generation_stats_json)
        produced.append(top_rules(gs, out_dir / "fig_top_rules.png"))
        produced.append(targeted_vs_untargeted(gs, out_dir / "fig_targeted_vs_untargeted.png"))
        produced.append(per_user_rule_counts(gs, out_dir / "fig_per_user_rules.png"))
    if results_csv:
        rc = Path(results_csv)
        produced.append(hit_at_k_bars(rc, out_dir / "fig_hit_at_k_bars.png"))
        produced.append(complementarity(rc, out_dir / "fig_complementarity.png"))
    if ablations_csv:
        produced.append(ablation_bars(Path(ablations_csv), out_dir / "fig_ablation_bars.png"))

    paths = [str(p) for p in produced if p is not None]
    logger.info("generate_all_figures: %d figures → %s", len(paths), out_dir)
    return paths
