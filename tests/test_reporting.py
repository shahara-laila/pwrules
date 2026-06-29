"""Tests for Phase 10 — paper-ready artifact export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pwrules.eval.reporting import (
    _fmt,
    _pivot_hit_at_k,
    _to_latex,
    export_paper_artifacts,
    make_ablation_table,
    make_filter_funnel_table,
    make_guessing_curve,
    make_hit_at_k_table,
    make_targeted_table,
    write_missing_file,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_results_csv(path: Path, rows):
    fields = ["method", "dataset", "k", "hit_rate", "seed"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_targeted_csv(path: Path, rows):
    fields = ["user_id", "k", "hit", "dataset", "seed"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _sample_results():
    rows = []
    for method in ["LLM-untargeted", "best64", "LLM+best64"]:
        for k, rate in [(10, 0.01), (100, 0.05), (1000, 0.12), (10000, 0.22)]:
            rows.append({"method": method, "dataset": "d1", "k": k,
                          "hit_rate": rate, "seed": 1})
    return rows


# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------

class TestFmt:
    def test_float(self):
        assert _fmt(0.12345) == "0.1235"

    def test_none_returns_missing(self):
        assert _fmt(None) == "MISSING"

    def test_empty_returns_missing(self):
        assert _fmt("") == "MISSING"


# ---------------------------------------------------------------------------
# _pivot_hit_at_k
# ---------------------------------------------------------------------------

class TestPivotHitAtK:
    def test_shape(self):
        rows = [
            {"method": "A", "dataset": "d1", "k": 10,  "hit_rate": 0.1, "seed": 1},
            {"method": "A", "dataset": "d1", "k": 100, "hit_rate": 0.2, "seed": 1},
            {"method": "B", "dataset": "d1", "k": 10,  "hit_rate": 0.3, "seed": 1},
            {"method": "B", "dataset": "d1", "k": 100, "hit_rate": 0.4, "seed": 1},
        ]
        row_labels, col_labels, cells = _pivot_hit_at_k(rows, [10, 100])
        assert len(row_labels) == 2
        assert col_labels == ["10", "100"]
        assert len(cells) == 2
        assert len(cells[0]) == 2

    def test_missing_cell_is_marked(self):
        rows = [{"method": "A", "dataset": "d1", "k": 10, "hit_rate": 0.1, "seed": 1}]
        _, _, cells = _pivot_hit_at_k(rows, [10, 100])
        # k=100 absent → MISSING
        assert cells[0][1] == "MISSING"


# ---------------------------------------------------------------------------
# _to_latex
# ---------------------------------------------------------------------------

class TestToLatex:
    def test_contains_booktabs_commands(self):
        latex = _to_latex(
            ["A", "B"],
            ["k=10", "k=100"],
            [["0.1000", "0.2000"], ["0.3000", "0.4000"]],
            caption="Test", label="tab:test",
        )
        assert r"\toprule" in latex
        assert r"\midrule" in latex
        assert r"\bottomrule" in latex
        assert r"\begin{tabular}" in latex
        assert r"\end{tabular}" in latex

    def test_contains_data(self):
        latex = _to_latex(
            ["LLM"], ["k=10"], [["0.1234"]],
            caption="C", label="L",
        )
        assert "LLM" in latex
        assert "0.1234" in latex


# ---------------------------------------------------------------------------
# make_hit_at_k_table
# ---------------------------------------------------------------------------

class TestMakeHitAtKTable:
    def test_creates_csv_and_tex(self, tmp_path):
        results_csv = tmp_path / "results.csv"
        _write_results_csv(results_csv, _sample_results())
        out_dir = tmp_path / "paper"
        make_hit_at_k_table(results_csv, out_dir, [10, 100, 1000, 10000])
        assert (out_dir / "table_hit_at_k.csv").exists()
        assert (out_dir / "table_hit_at_k.tex").exists()

    def test_missing_file_adds_to_set(self, tmp_path):
        missing = set()
        make_hit_at_k_table(
            tmp_path / "nonexistent.csv", tmp_path / "out", [10], missing=missing
        )
        assert len(missing) == 1


# ---------------------------------------------------------------------------
# make_targeted_table
# ---------------------------------------------------------------------------

class TestMakeTargetedTable:
    def test_creates_csv_and_tex(self, tmp_path):
        tgt_csv = tmp_path / "targeted_results.csv"
        _write_targeted_csv(tgt_csv, [
            {"user_id": "u1", "k": 100, "hit": 1, "dataset": "d1", "seed": 1},
            {"user_id": "u2", "k": 100, "hit": 0, "dataset": "d1", "seed": 1},
        ])
        out_dir = tmp_path / "paper"
        make_targeted_table(tgt_csv, out_dir, [100])
        assert (out_dir / "table_targeted.csv").exists()
        assert (out_dir / "table_targeted.tex").exists()

    def test_missing_file(self, tmp_path):
        missing = set()
        make_targeted_table(tmp_path / "no.csv", tmp_path, [100], missing=missing)
        assert len(missing) == 1


# ---------------------------------------------------------------------------
# make_filter_funnel_table
# ---------------------------------------------------------------------------

class TestMakeFilterFunnelTable:
    def test_creates_files(self, tmp_path):
        funnel_csv = tmp_path / "filter_funnel.csv"
        funnel_csv.write_text(
            "file,generated,valid,unique,effective\nrules,100,90,80,70\n"
        )
        out_dir = tmp_path / "paper"
        make_filter_funnel_table(funnel_csv, out_dir)
        assert (out_dir / "table_filter_funnel.csv").exists()
        assert (out_dir / "table_filter_funnel.tex").exists()

    def test_missing_file(self, tmp_path):
        missing = set()
        make_filter_funnel_table(tmp_path / "no.csv", tmp_path, missing=missing)
        assert len(missing) == 1


# ---------------------------------------------------------------------------
# make_ablation_table
# ---------------------------------------------------------------------------

class TestMakeAblationTable:
    def test_creates_files(self, tmp_path):
        abl_csv = tmp_path / "ablations.csv"
        abl_csv.write_text(
            "axis,label,method_a,method_b,dataset,k,mean_a,std_a,mean_b,std_b,delta\n"
            "target-conditioning,test,LLM-targeted,LLM-untargeted,d1,1000000,0.35,0.02,0.30,0.02,0.05\n"
        )
        out_dir = tmp_path / "paper"
        make_ablation_table(abl_csv, out_dir)
        assert (out_dir / "table_ablations.csv").exists()
        assert (out_dir / "table_ablations.tex").exists()


# ---------------------------------------------------------------------------
# write_missing_file
# ---------------------------------------------------------------------------

class TestWriteMissingFile:
    def test_writes_missing_entries(self, tmp_path):
        write_missing_file({"file_a.csv", "file_b.csv"}, tmp_path)
        content = (tmp_path / "MISSING.txt").read_text()
        assert "MISSING" in content

    def test_all_present_message(self, tmp_path):
        write_missing_file(set(), tmp_path)
        content = (tmp_path / "MISSING.txt").read_text()
        assert "present" in content.lower()


# ---------------------------------------------------------------------------
# export_paper_artifacts (integration)
# ---------------------------------------------------------------------------

class TestExportPaperArtifacts:
    def test_integration_partial(self, tmp_path):
        """Run with results.csv only; expect MISSING markers for others."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_csv = results_dir / "results.csv"
        _write_results_csv(results_csv, _sample_results())

        out_dir = tmp_path / "paper"
        result = export_paper_artifacts(
            results_dir=results_dir,
            out_dir=out_dir,
            k_values=[10, 100, 1000, 10000],
        )

        assert (out_dir / "table_hit_at_k.csv").exists()
        assert (out_dir / "table_hit_at_k.tex").exists()
        assert (out_dir / "MISSING.txt").exists()
        assert result["n_missing"] > 0  # targeted, ablations, funnel missing

    def test_integration_full(self, tmp_path):
        """Run with all source files present; expect no MISSING."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        abl_dir = tmp_path / "ablations"
        abl_dir.mkdir()
        filter_dir = tmp_path / "filtered"
        filter_dir.mkdir()

        # Write all source files.
        _write_results_csv(results_dir / "results.csv", _sample_results())
        _write_targeted_csv(results_dir / "targeted_results.csv", [
            {"user_id": "u1", "k": 10, "hit": 1, "dataset": "d1", "seed": 1},
            {"user_id": "u1", "k": 100, "hit": 1, "dataset": "d1", "seed": 1},
        ])
        (abl_dir / "ablations.csv").write_text(
            "axis,label,method_a,method_b,dataset,k,mean_a,std_a,mean_b,std_b,delta\n"
            "filtering,test,LLM-filtered,LLM-untargeted,d1,10,0.01,0.00,0.01,0.00,0.00\n"
        )
        (filter_dir / "filter_funnel.csv").write_text(
            "file,generated,valid,unique,effective\nrules,100,90,80,70\n"
        )

        out_dir = tmp_path / "paper"
        result = export_paper_artifacts(
            results_dir=results_dir,
            out_dir=out_dir,
            k_values=[10, 100],
            ablations_dir=abl_dir,
            filter_dir=filter_dir,
        )

        assert result["n_missing"] == 0
        assert (out_dir / "MISSING.txt").read_text().strip() != ""
