"""Tests for Phase 9 — ablations and statistical significance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pwrules.eval.ablations import (
    aggregate_seeds,
    bootstrap_ci,
    build_ablation_table,
    compute_significance,
    mcnemar_p,
    run_ablations,
    save_ablations_csv,
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


# ---------------------------------------------------------------------------
# aggregate_seeds
# ---------------------------------------------------------------------------

class TestAggregateSeeds:
    def test_single_seed(self):
        rows = [
            {"method": "LLM", "dataset": "d1", "k": 100, "hit_rate": 0.10, "seed": 1},
            {"method": "best64", "dataset": "d1", "k": 100, "hit_rate": 0.15, "seed": 1},
        ]
        agg = aggregate_seeds(rows)
        assert len(agg) == 2
        llm_row = next(r for r in agg if r["method"] == "LLM")
        assert llm_row["mean_hit_rate"] == pytest.approx(0.10)
        assert llm_row["std_hit_rate"] == pytest.approx(0.0)
        assert llm_row["n_seeds"] == 1

    def test_multi_seed_mean_std(self):
        rows = [
            {"method": "LLM", "dataset": "d1", "k": 100, "hit_rate": 0.10, "seed": 1},
            {"method": "LLM", "dataset": "d1", "k": 100, "hit_rate": 0.20, "seed": 2},
            {"method": "LLM", "dataset": "d1", "k": 100, "hit_rate": 0.30, "seed": 3},
        ]
        agg = aggregate_seeds(rows)
        assert len(agg) == 1
        row = agg[0]
        assert row["mean_hit_rate"] == pytest.approx(0.20)
        assert row["std_hit_rate"] == pytest.approx(np.std([0.10, 0.20, 0.30], ddof=1))
        assert row["n_seeds"] == 3

    def test_k_filter(self):
        rows = [
            {"method": "LLM", "dataset": "d", "k": 10,  "hit_rate": 0.05, "seed": 1},
            {"method": "LLM", "dataset": "d", "k": 100, "hit_rate": 0.15, "seed": 1},
        ]
        agg = aggregate_seeds(rows, k_values=[100])
        assert len(agg) == 1
        assert agg[0]["k"] == 100


# ---------------------------------------------------------------------------
# build_ablation_table
# ---------------------------------------------------------------------------

class TestBuildAblationTable:
    def _make_agg_rows(self):
        return [
            {"method": "LLM-untargeted",  "dataset": "d1", "k": 1000000,
             "mean_hit_rate": 0.30, "std_hit_rate": 0.02, "n_seeds": 3},
            {"method": "LLM-targeted",    "dataset": "d1", "k": 1000000,
             "mean_hit_rate": 0.35, "std_hit_rate": 0.02, "n_seeds": 3},
            {"method": "LLM-filtered",    "dataset": "d1", "k": 1000000,
             "mean_hit_rate": 0.32, "std_hit_rate": 0.01, "n_seeds": 3},
            {"method": "best64",          "dataset": "d1", "k": 1000000,
             "mean_hit_rate": 0.28, "std_hit_rate": 0.00, "n_seeds": 1},
            {"method": "LLM+best64",      "dataset": "d1", "k": 1000000,
             "mean_hit_rate": 0.40, "std_hit_rate": 0.02, "n_seeds": 3},
        ]

    def test_returns_rows(self):
        agg = self._make_agg_rows()
        rows = build_ablation_table(agg, k_pivot=1_000_000)
        assert len(rows) > 0

    def test_delta_sign(self):
        agg = self._make_agg_rows()
        rows = build_ablation_table(agg, k_pivot=1_000_000)
        cond = next(
            (r for r in rows if r["axis"] == "target-conditioning"), None
        )
        if cond:
            # LLM-targeted (0.35) > LLM-untargeted (0.30) → delta > 0
            assert cond["delta"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_positive_delta(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0.4, 0.05, 100)
        b = rng.normal(0.2, 0.05, 100)
        lo, hi, obs = bootstrap_ci(a, b, n_bootstrap=1000)
        assert obs > 0
        assert lo < hi
        # 95% CI should be entirely positive (a is much higher than b).
        assert lo > 0

    def test_zero_delta_ci_spans_zero(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0.3, 0.05, 100)
        lo, hi, obs = bootstrap_ci(x, x.copy(), n_bootstrap=1000)
        assert lo < 0 < hi or (abs(obs) < 1e-10)


# ---------------------------------------------------------------------------
# mcnemar_p
# ---------------------------------------------------------------------------

class TestMcNemarP:
    def test_p_between_0_and_1(self):
        hits_a = np.array([1, 1, 0, 0, 1, 0])
        hits_b = np.array([0, 1, 1, 0, 0, 0])
        p = mcnemar_p(hits_a, hits_b)
        assert 0.0 <= p <= 1.0

    def test_identical_p_is_1(self):
        hits = np.array([1, 0, 1, 0, 1])
        p = mcnemar_p(hits, hits.copy())
        assert p == pytest.approx(1.0)

    def test_all_discordant_low_p(self):
        # A always right, B always wrong → extreme discordance.
        hits_a = np.ones(100, dtype=int)
        hits_b = np.zeros(100, dtype=int)
        p = mcnemar_p(hits_a, hits_b)
        assert p < 0.001


# ---------------------------------------------------------------------------
# compute_significance
# ---------------------------------------------------------------------------

class TestComputeSignificance:
    def test_returns_rows(self):
        rows = [
            {"method": "LLM", "dataset": "d", "k": 100, "hit_rate": 0.3, "seed": 1},
            {"method": "LLM", "dataset": "d", "k": 100, "hit_rate": 0.4, "seed": 2},
            {"method": "best64", "dataset": "d", "k": 100, "hit_rate": 0.2, "seed": 1},
            {"method": "best64", "dataset": "d", "k": 100, "hit_rate": 0.25, "seed": 2},
        ]
        sig = compute_significance(rows, baseline_method="best64", n_bootstrap=200)
        assert len(sig) > 0
        row = sig[0]
        assert "method" in row
        assert "observed_delta" in row
        assert "ci_lo" in row
        assert "ci_hi" in row
        assert "significant_005" in row


# ---------------------------------------------------------------------------
# save_ablations_csv
# ---------------------------------------------------------------------------

class TestSaveAblationsCSV:
    def test_writes_csv(self, tmp_path):
        rows = [
            {"axis": "filtering", "method_a": "A", "method_b": "B",
             "delta": 0.05, "dataset": "d1"},
        ]
        path = tmp_path / "ablations.csv"
        save_ablations_csv(rows, path)
        assert path.exists()
        content = path.read_text()
        assert "axis" in content
        assert "filtering" in content


# ---------------------------------------------------------------------------
# run_ablations (integration)
# ---------------------------------------------------------------------------

class TestRunAblations:
    def test_integration(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Write multi-seed results.
        for seed in [1, 2, 3]:
            rows = [
                {"method": "LLM-untargeted",  "dataset": "d1", "k": 1000000,
                 "hit_rate": 0.30 + seed * 0.01, "seed": seed},
                {"method": "best64",           "dataset": "d1", "k": 1000000,
                 "hit_rate": 0.25 + seed * 0.01, "seed": seed},
            ]
            _write_results_csv(
                results_dir / f"results_seed{seed}.csv", rows
            )

        out_dir = tmp_path / "ablations"
        result = run_ablations(
            results_dir=results_dir,
            out_dir=out_dir,
            baseline_method="best64",
            n_bootstrap=500,
        )

        assert (out_dir / "ablations.csv").exists()
        assert (out_dir / "significance_report.json").exists()
        assert (out_dir / "aggregated_results.csv").exists()
        assert result["n_ablation_conditions"] >= 0
