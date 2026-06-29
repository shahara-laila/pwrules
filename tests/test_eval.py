"""Tests for Phase 8 — Hit@k evaluation harness."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pwrules.eval import (
    append_results,
    evaluate_method_from_lists,
    hit_at_k,
    load_results_csv,
    load_test_set,
    merge_rulesets_to_tmp,
    save_guessing_curve,
)


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------

class TestHitAtK:
    def test_perfect_recall_at_k(self):
        candidates = ["a", "b", "c", "d"]
        test_set = {"a", "b", "c", "d"}
        result = hit_at_k(candidates, test_set, [4])
        assert result[4] == pytest.approx(1.0)

    def test_zero_recall_empty_intersection(self):
        candidates = ["x", "y", "z"]
        test_set = {"a", "b"}
        result = hit_at_k(candidates, test_set, [3])
        assert result[3] == pytest.approx(0.0)

    def test_partial_recall(self):
        candidates = ["a", "x", "b", "y"]
        test_set = {"a", "b"}
        result = hit_at_k(candidates, test_set, [2, 4])
        # At k=2: only 'a' hit → 0.5
        assert result[2] == pytest.approx(0.5)
        # At k=4: both 'a' and 'b' hit → 1.0
        assert result[4] == pytest.approx(1.0)

    def test_empty_test_set(self):
        result = hit_at_k(["a", "b"], set(), [2])
        assert result[2] == pytest.approx(0.0)

    def test_k_exceeds_candidates(self):
        candidates = ["a", "b"]
        test_set = {"a", "c"}
        result = hit_at_k(candidates, test_set, [100])
        # Only 'a' is in candidates, so 1/2 = 0.5
        assert result[100] == pytest.approx(0.5)

    def test_monotone_non_decreasing(self):
        candidates = list("abcdefghij")
        test_set = {"a", "e", "j"}
        ks = [1, 3, 5, 7, 10]
        result = hit_at_k(candidates, test_set, ks)
        rates = [result[k] for k in ks]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i + 1], "Hit@k must be non-decreasing"

    def test_multiple_k_values(self):
        candidates = ["p1", "p2", "p3"]
        test_set = {"p1", "p3"}
        result = hit_at_k(candidates, test_set, [1, 2, 3])
        assert result[1] == pytest.approx(0.5)   # p1 hit
        assert result[2] == pytest.approx(0.5)   # p1 hit, p2 miss
        assert result[3] == pytest.approx(1.0)   # p1 + p3 hit

    def test_dedup_already_assumed(self):
        # The function assumes candidates are already deduplicated.
        candidates = ["a", "a", "b"]
        test_set = {"a", "b"}
        result = hit_at_k(candidates, test_set, [3])
        # 'a' counted once (first occurrence), 'a' again (no effect), 'b' at k=3
        # But hit_at_k counts by position, so:
        # k=1: a in test → 1 hit; k=2: a again, BUT it's the second 'a' which is
        # a separate element in the list (duplicates not removed by hit_at_k itself)
        # → so we just count naively: position 1: a→hit, position 2: a→hit again.
        # This is correct since generate_candidates already deduplicates.
        # Hit@3 should be 2/2 = 1.0 (both test passwords found).
        assert result[3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# load_test_set
# ---------------------------------------------------------------------------

class TestLoadTestSet:
    def test_loads_passwords(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("password\ndragon\nhello\n")
        s = load_test_set(path)
        assert s == {"password", "dragon", "hello"}

    def test_empty_lines_skipped(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("a\n\nb\n\n")
        s = load_test_set(path)
        assert s == {"a", "b"}

    def test_deduplicates(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("a\na\nb\n")
        s = load_test_set(path)
        assert len(s) == 2


# ---------------------------------------------------------------------------
# evaluate_method_from_lists
# ---------------------------------------------------------------------------

class TestEvaluateMethodFromLists:
    def test_returns_result_rows(self):
        candidates = ["p1", "p2", "p3"]
        test_set = {"p1", "p3"}
        rows = evaluate_method_from_lists(
            "test_method", candidates, test_set, [1, 3], seed=42
        )
        assert len(rows) == 2
        ks = {r["k"] for r in rows}
        assert ks == {1, 3}

    def test_result_row_fields(self):
        rows = evaluate_method_from_lists(
            "mymethod", ["a"], {"a"}, [1], seed=7, dataset_name="ds1"
        )
        row = rows[0]
        assert row["method"] == "mymethod"
        assert row["dataset"] == "ds1"
        assert row["seed"] == 7
        assert row["k"] == 1
        assert row["hit_rate"] == pytest.approx(1.0)

    def test_empty_candidates(self):
        rows = evaluate_method_from_lists("m", [], {"p"}, [100])
        assert rows[0]["hit_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# merge_rulesets_to_tmp
# ---------------------------------------------------------------------------

class TestMergeRulesets:
    def test_concatenates_files(self, tmp_path):
        f1 = tmp_path / "a.rule"
        f2 = tmp_path / "b.rule"
        f1.write_text("c\nl\n")
        f2.write_text("$1\n$2\n")

        merged = merge_rulesets_to_tmp([f1, f2])
        content = Path(merged).read_text()
        assert "c\n" in content
        assert "l\n" in content
        assert "$1\n" in content
        assert "$2\n" in content

        import os
        os.unlink(merged)


# ---------------------------------------------------------------------------
# append_results / load_results_csv
# ---------------------------------------------------------------------------

class TestResultsCSV:
    def test_round_trip(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        rows = [
            {"method": "LLM", "dataset": "ds1", "k": 100, "hit_rate": 0.12, "seed": 1},
            {"method": "best64", "dataset": "ds1", "k": 100, "hit_rate": 0.15, "seed": 1},
        ]
        append_results(rows, csv_path)
        loaded = load_results_csv(csv_path)
        assert len(loaded) == 2
        assert loaded[0]["method"] == "LLM"
        assert loaded[0]["k"] == 100
        assert loaded[0]["hit_rate"] == pytest.approx(0.12)

    def test_append_creates_header_once(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        append_results(
            [{"method": "A", "dataset": "d", "k": 10, "hit_rate": 0.1, "seed": 1}],
            csv_path,
        )
        append_results(
            [{"method": "B", "dataset": "d", "k": 10, "hit_rate": 0.2, "seed": 1}],
            csv_path,
        )
        lines = csv_path.read_text().splitlines()
        # One header + two data rows.
        assert len(lines) == 3
        assert lines[0].startswith("method")

    def test_empty_list_no_file_created(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        append_results([], csv_path)
        assert not csv_path.exists()


# ---------------------------------------------------------------------------
# save_guessing_curve
# ---------------------------------------------------------------------------

class TestGuessingCurve:
    def test_creates_png(self, tmp_path):
        rows = [
            {"method": "LLM", "k": 10,    "hit_rate": 0.05},
            {"method": "LLM", "k": 100,   "hit_rate": 0.12},
            {"method": "LLM", "k": 1000,  "hit_rate": 0.22},
            {"method": "best64", "k": 10,   "hit_rate": 0.08},
            {"method": "best64", "k": 100,  "hit_rate": 0.18},
            {"method": "best64", "k": 1000, "hit_rate": 0.30},
        ]
        out_path = tmp_path / "curve.png"
        save_guessing_curve(rows, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
