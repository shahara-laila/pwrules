"""Phase 10 — Paper-ready artifact export.

Reads result CSVs produced by Phases 8–9 and emits:

    paper/
      table_hit_at_k.csv          Main Hit@k comparison (rows=methods, cols=k)
      table_hit_at_k.tex          LaTeX version of the same table
      guessing_curve.png          Rebuilt from results.csv (same as Phase 8 output)
      table_targeted.csv          Targeted evaluation aggregate
      table_targeted.tex
      table_filter_funnel.csv     Copy / symlink to filter_funnel.csv
      table_filter_funnel.tex
      table_ablations.csv         Copy / symlink to ablations.csv
      table_ablations.tex
      MISSING.txt                 Lists any missing source files

Rules
-----
* NEVER invent numbers. If a source file is absent, write a MISSING marker.
* Numeric format: 4 decimal places for Hit@k values.
* LaTeX tables use ``\\toprule`` / ``\\midrule`` / ``\\bottomrule`` (booktabs).
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pwrules.eval import load_results_csv, save_guessing_curve

logger = logging.getLogger(__name__)

_MISSING = "MISSING"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(v: object, decimals: int = 4) -> str:
    """Format a value; return MISSING if None or empty."""
    if v is None or v == "":
        return _MISSING
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _pivot_hit_at_k(
    rows: List[Dict],
    k_values: List[int],
    dataset: Optional[str] = None,
) -> Tuple[List[str], List[str], List[List[str]]]:
    """Pivot result rows into a (method × k) table.

    Returns ``(row_labels, col_labels, cell_values)``.
    """
    # Filter by dataset if specified.
    if dataset:
        rows = [r for r in rows if r.get("dataset") == dataset]

    # Group by (method, k).
    index: Dict[Tuple[str, int], float] = {}
    for row in rows:
        index[(row["method"], int(row["k"]))] = float(row["hit_rate"])

    methods = sorted({r["method"] for r in rows})
    col_labels = [str(k) for k in k_values]
    row_labels = methods

    cells: List[List[str]] = []
    for method in methods:
        row_cells: List[str] = []
        for k in k_values:
            val = index.get((method, k))
            row_cells.append(_fmt(val) if val is not None else _MISSING)
        cells.append(row_cells)

    return row_labels, col_labels, cells


def _to_latex(
    row_labels: List[str],
    col_labels: List[str],
    cells: List[List[str]],
    caption: str,
    label: str,
) -> str:
    """Build a booktabs LaTeX table string."""
    n_cols = len(col_labels) + 1  # +1 for the method column
    col_spec = "l" + "r" * len(col_labels)

    def _esc(s: str) -> str:
        return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        "    Method & " + " & ".join(_esc(c) for c in col_labels) + r" \\",
        r"    \midrule",
    ]
    for label_row, row_cells in zip(row_labels, cells):
        lines.append(
            f"    {_esc(label_row)} & "
            + " & ".join(_esc(c) for c in row_cells)
            + r" \\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _write_csv_table(
    row_labels: List[str],
    col_labels: List[str],
    cells: List[List[str]],
    path: Path,
    row_header: str = "Method",
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row_header] + col_labels)
        for label, row_cells in zip(row_labels, cells):
            w.writerow([label] + row_cells)
    logger.info("Table CSV → %s", path)


def _missing(name: str, missing_set: Set[str]) -> None:
    missing_set.add(name)
    logger.warning("MISSING: %s", name)


# ---------------------------------------------------------------------------
# Individual table builders
# ---------------------------------------------------------------------------

def make_hit_at_k_table(
    results_csv: Path,
    out_dir: Path,
    k_values: List[int],
    dataset: Optional[str] = None,
    missing: Optional[Set[str]] = None,
) -> None:
    """Main Hit@k comparison table — CSV + LaTeX."""
    if missing is None:
        missing = set()

    if not results_csv.exists():
        _missing(str(results_csv), missing)
        return

    rows = load_results_csv(results_csv)
    row_labels, col_labels, cells = _pivot_hit_at_k(rows, k_values, dataset)

    _write_csv_table(
        row_labels, col_labels, cells,
        out_dir / "table_hit_at_k.csv",
    )

    latex = _to_latex(
        row_labels,
        [f"k={c}" for c in col_labels],
        cells,
        caption="Hit@k comparison of all methods.",
        label="tab:hit_at_k",
    )
    (out_dir / "table_hit_at_k.tex").write_text(latex, encoding="utf-8")
    logger.info("Hit@k table written → %s", out_dir)


def make_targeted_table(
    targeted_csv: Path,
    out_dir: Path,
    k_values: List[int],
    missing: Optional[Set[str]] = None,
) -> None:
    """Targeted evaluation table — CSV + LaTeX."""
    if missing is None:
        missing = set()

    if not targeted_csv.exists():
        _missing(str(targeted_csv), missing)
        return

    # Load per-user rows.
    rows: List[Dict] = []
    with open(targeted_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["k"] = int(row["k"])
            row["hit"] = int(row["hit"])
            rows.append(row)

    # Aggregate: fraction of users cracked per k.
    agg: Dict[int, float] = {}
    for k in k_values:
        rows_k = [r for r in rows if r["k"] == k]
        if rows_k:
            agg[k] = sum(r["hit"] for r in rows_k) / len(rows_k)

    row_labels = ["LLM-targeted"]
    col_labels = [str(k) for k in k_values]
    cells = [[_fmt(agg.get(k)) for k in k_values]]

    _write_csv_table(
        row_labels, col_labels, cells,
        out_dir / "table_targeted.csv",
        row_header="Method",
    )

    latex = _to_latex(
        row_labels,
        [f"k={c}" for c in col_labels],
        cells,
        caption="Targeted evaluation: fraction of held-out user passwords cracked.",
        label="tab:targeted",
    )
    (out_dir / "table_targeted.tex").write_text(latex, encoding="utf-8")
    logger.info("Targeted table → %s", out_dir)


def make_filter_funnel_table(
    funnel_csv: Path,
    out_dir: Path,
    missing: Optional[Set[str]] = None,
) -> None:
    """Filter funnel table — CSV copy + LaTeX."""
    if missing is None:
        missing = set()

    if not funnel_csv.exists():
        _missing(str(funnel_csv), missing)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(funnel_csv, out_dir / "table_filter_funnel.csv")

    # Build LaTeX from funnel CSV.
    rows: List[Dict] = []
    with open(funnel_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    col_labels = ["Generated", "Valid", "Unique", "Effective"]
    row_labels = [r["file"] for r in rows]
    cells = [
        [r.get("generated", _MISSING), r.get("valid", _MISSING),
         r.get("unique", _MISSING), r.get("effective", _MISSING)]
        for r in rows
    ]

    latex = _to_latex(
        row_labels, col_labels, cells,
        caption="Rule filtering funnel: counts at each stage.",
        label="tab:filter_funnel",
    )
    (out_dir / "table_filter_funnel.tex").write_text(latex, encoding="utf-8")
    logger.info("Filter funnel table → %s", out_dir)


def make_ablation_table(
    ablations_csv: Path,
    out_dir: Path,
    missing: Optional[Set[str]] = None,
) -> None:
    """Ablation table — CSV copy + LaTeX."""
    if missing is None:
        missing = set()

    if not ablations_csv.exists():
        _missing(str(ablations_csv), missing)
        return

    rows: List[Dict] = []
    with open(ablations_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    # "Delta" (plain text) renders cleanly in both CSV and LaTeX; a literal
    # Unicode Δ breaks pdflatex without fontspec.
    col_labels = ["Method A", "Method B", "Dataset", "k", "Mean A", "Std A", "Mean B", "Std B", "Delta"]
    row_labels = [r.get("label", r.get("axis", "")) for r in rows]
    cells = [
        [
            r.get("method_a", _MISSING),
            r.get("method_b", _MISSING),
            r.get("dataset", _MISSING),
            str(r.get("k", _MISSING)),
            _fmt(r.get("mean_a")),
            _fmt(r.get("std_a")),
            _fmt(r.get("mean_b")),
            _fmt(r.get("std_b")),
            _fmt(r.get("delta")),
        ]
        for r in rows
    ]

    _write_csv_table(
        row_labels, col_labels, cells,
        out_dir / "table_ablations.csv",
        row_header="Ablation",
    )

    latex = _to_latex(
        row_labels, col_labels, cells,
        caption="Ablation study: pairwise comparison at the chosen guess budget.",
        label="tab:ablations",
    )
    (out_dir / "table_ablations.tex").write_text(latex, encoding="utf-8")
    logger.info("Ablation table → %s", out_dir)


# ---------------------------------------------------------------------------
# Guessing-number curve
# ---------------------------------------------------------------------------

def make_guessing_curve(
    results_csv: Path,
    out_dir: Path,
    missing: Optional[Set[str]] = None,
) -> None:
    if missing is None:
        missing = set()

    if not results_csv.exists():
        _missing(str(results_csv), missing)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_results_csv(results_csv)
    save_guessing_curve(rows, out_dir / "guessing_curve.png")


# ---------------------------------------------------------------------------
# MISSING.txt
# ---------------------------------------------------------------------------

def write_missing_file(missing: Set[str], out_dir: Path) -> None:
    missing_path = out_dir / "MISSING.txt"
    if missing:
        with open(missing_path, "w", encoding="utf-8") as f:
            for m in sorted(missing):
                f.write(f"MISSING: {m}\n")
        logger.warning(
            "%d source file(s) missing — see %s", len(missing), missing_path
        )
    else:
        missing_path.write_text("All source files present.\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------

def export_paper_artifacts(
    results_dir: Path,
    out_dir: Path,
    k_values: Optional[List[int]] = None,
    ablations_dir: Optional[Path] = None,
    filter_dir: Optional[Path] = None,
    dataset: Optional[str] = None,
) -> Dict[str, object]:
    """Read all result files and write paper-ready artifacts.

    Parameters
    ----------
    results_dir:
        Directory with ``results.csv`` and ``targeted_results.csv``
        (output of Phase 8).
    out_dir:
        Directory for paper artifacts.
    k_values:
        Budget schedule; defaults to the full protocol schedule.
    ablations_dir:
        Directory with ``ablations.csv`` (output of Phase 9).
        Defaults to *results_dir*.
    filter_dir:
        Directory with ``filter_funnel.csv`` (output of Phase 7).
    dataset:
        Dataset name to use when pivoting the Hit@k table
        (None = use all datasets).

    Returns
    -------
    Summary dict.
    """
    if k_values is None:
        k_values = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

    if ablations_dir is None:
        ablations_dir = results_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    missing: Set[str] = set()

    results_csv   = results_dir / "results.csv"
    targeted_csv  = results_dir / "targeted_results.csv"
    ablations_csv = ablations_dir / "ablations.csv"
    funnel_csv    = (filter_dir / "filter_funnel.csv") if filter_dir else None

    make_hit_at_k_table(results_csv, out_dir, k_values, dataset, missing)
    make_guessing_curve(results_csv, out_dir, missing)
    make_targeted_table(targeted_csv, out_dir, k_values, missing)
    make_ablation_table(ablations_csv, out_dir, missing)

    if funnel_csv:
        make_filter_funnel_table(funnel_csv, out_dir, missing)
    else:
        _missing("filter_funnel.csv (no --filter-dir provided)", missing)

    # Paper-ready research figures (best-effort: each is skipped if its source
    # artifact is absent). Inputs beyond results/ablations are auto-discovered.
    figures_made: List[str] = []
    try:
        from pwrules import paths
        from pwrules.eval import figures

        gen_stats = paths.find_file("generation_stats.json", required=False)
        memo = paths.find_file("memorisation_report.json", required=False)
        rule_file = (paths.filtered_untargeted(required=False)
                     or paths.generated_untargeted(required=False))
        figures_made = figures.generate_all_figures(
            out_dir=out_dir / "figures",
            results_csv=results_csv if results_csv.exists() else None,
            ablations_csv=ablations_csv if ablations_csv.exists() else None,
            generation_stats_json=gen_stats,
            memorisation_json=memo,
            rule_file=rule_file,
        )
    except Exception as exc:  # pragma: no cover - figures are non-critical
        logger.warning("Figure generation skipped: %s", exc)

    write_missing_file(missing, out_dir)

    logger.info("Paper artifacts → %s", out_dir)
    return {
        "out_dir":  str(out_dir),
        "missing":  sorted(missing),
        "n_missing": len(missing),
        "figures":  figures_made,
    }
