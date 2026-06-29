"""Tests for Phase 7 — rule filtering."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pwrules.filter import (
    _fingerprint,
    _is_noop,
    _is_parseable,
    filter_rules,
    load_rules,
    semantic_dedup,
    syntax_check,
    write_rules,
)

# ---------------------------------------------------------------------------
# _is_parseable
# ---------------------------------------------------------------------------

class TestIsParseable:
    def test_identity_rule(self):
        assert _is_parseable(":")

    def test_lowercase_rule(self):
        assert _is_parseable("l")

    def test_append_rule(self):
        assert _is_parseable("$1")

    def test_substitute_rule(self):
        assert _is_parseable("sa@")

    def test_compound_rule(self):
        assert _is_parseable("c sa@ $1 $2 $3")

    def test_empty_string_invalid(self):
        assert not _is_parseable("")

    def test_whitespace_only_invalid(self):
        assert not _is_parseable("   ")

    def test_unknown_function_invalid(self):
        assert not _is_parseable("X")

    def test_wrong_param_count_invalid(self):
        # '$' expects exactly 1 param char, '$12' has 2 params
        assert not _is_parseable("$12")

    def test_no_param_for_substitute_invalid(self):
        # 's' expects 2 params
        assert not _is_parseable("s")


# ---------------------------------------------------------------------------
# _is_noop
# ---------------------------------------------------------------------------

class TestIsNoop:
    def test_identity_is_noop(self):
        assert _is_noop(":", probe_words=["password"])

    def test_lowercase_on_lowercase_noop(self):
        assert _is_noop("l", probe_words=["alllowercase"])

    def test_append_not_noop(self):
        assert not _is_noop("$1", probe_words=["password"])

    def test_capitalise_not_noop(self):
        assert not _is_noop("c", probe_words=["password"])

    def test_substitute_no_match_is_noop(self):
        # sa@ → replace 'a' with '@'; "bbbbb" has no 'a'
        assert _is_noop("sa@", probe_words=["bbbbb"])

    def test_substitute_match_not_noop(self):
        assert not _is_noop("sa@", probe_words=["password"])


# ---------------------------------------------------------------------------
# syntax_check
# ---------------------------------------------------------------------------

class TestSyntaxCheck:
    def test_valid_rules_pass(self):
        rules = ["c", "l", "$1", "sa@", "c sa@ $1"]
        valid, n_invalid, n_noop = syntax_check(
            rules, probe_words=["password"], hashcat_sample=0
        )
        assert len(valid) == len(rules)
        assert n_invalid == 0

    def test_invalid_rule_removed(self):
        rules = ["c", "INVALID_TOKEN", "$1"]
        valid, n_invalid, _ = syntax_check(
            rules, probe_words=["password"], hashcat_sample=0
        )
        assert n_invalid == 1
        assert "INVALID_TOKEN" not in valid

    def test_noop_rule_removed(self):
        rules = [":", "l", "$1"]  # ":" is always a no-op
        valid, _, n_noop = syntax_check(
            rules, probe_words=["password"], hashcat_sample=0
        )
        assert n_noop >= 1
        assert ":" not in valid

    def test_empty_input(self):
        valid, n_invalid, n_noop = syntax_check([], hashcat_sample=0)
        assert valid == []
        assert n_invalid == 0
        assert n_noop == 0


# ---------------------------------------------------------------------------
# semantic_dedup
# ---------------------------------------------------------------------------

class TestSemanticDedup:
    def test_exact_duplicate_removed(self):
        rules = ["c", "c", "$1"]
        unique, n_removed = semantic_dedup(rules, probe_words=["password"])
        assert n_removed == 1
        assert unique.count("c") == 1

    def test_no_duplicates_unchanged(self):
        rules = ["c", "$1", "sa@"]
        unique, n_removed = semantic_dedup(rules, probe_words=["password"])
        assert n_removed == 0
        assert len(unique) == 3

    def test_semantically_equivalent_rules_collapsed(self):
        # "l l" applies lowercase twice — semantically identical to "l".
        rules = ["l", "l l"]
        unique, _ = semantic_dedup(rules, probe_words=["PassWord"])
        assert len(unique) == 1

    def test_empty_input(self):
        unique, n_removed = semantic_dedup([], probe_words=["password"])
        assert unique == []
        assert n_removed == 0


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_same_rule_same_fingerprint(self):
        fp1 = _fingerprint("c", ["password", "hello"])
        fp2 = _fingerprint("c", ["password", "hello"])
        assert fp1 == fp2

    def test_different_rules_different_fingerprints(self):
        fp1 = _fingerprint("c", ["password"])
        fp2 = _fingerprint("l", ["password"])
        assert fp1 != fp2

    def test_equivalent_rules_same_fingerprint(self):
        fp1 = _fingerprint("l", ["PassWord"])
        fp2 = _fingerprint("l l", ["PassWord"])
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# load_rules / write_rules
# ---------------------------------------------------------------------------

class TestRuleIO:
    def test_round_trip(self, tmp_path):
        rules = ["c", "sa@ $1", "l u"]
        path = tmp_path / "test.rule"
        write_rules(rules, path)
        loaded = load_rules(path)
        assert loaded == rules

    def test_comments_stripped(self, tmp_path):
        path = tmp_path / "test.rule"
        path.write_text("# This is a comment\nc\n# Another comment\n$1\n")
        loaded = load_rules(path)
        assert loaded == ["c", "$1"]

    def test_blank_lines_stripped(self, tmp_path):
        path = tmp_path / "test.rule"
        path.write_text("c\n\n$1\n\n")
        loaded = load_rules(path)
        assert loaded == ["c", "$1"]


# ---------------------------------------------------------------------------
# filter_rules (integration)
# ---------------------------------------------------------------------------

class TestFilterRules:
    def test_pipeline_runs(self, tmp_path):
        rule_path = tmp_path / "input.rule"
        rule_path.write_text("c\nl\n$1\n$1\n:\n")  # 5 rules, 1 dup, 1 noop

        result = filter_rules(
            rule_files=[rule_path],
            out_dir=tmp_path / "filtered",
            hashcat_sample=0,
        )

        assert "funnel" in result
        funnel = result["funnel"][0]
        assert funnel["generated"] == 5
        # ":" is a no-op; "$1" appears twice → one dup removed
        assert funnel["valid"] < funnel["generated"]
        assert funnel["unique"] <= funnel["valid"]

    def test_output_file_created(self, tmp_path):
        rule_path = tmp_path / "myrules.rule"
        rule_path.write_text("c\n$1\n")

        result = filter_rules(
            rule_files=[rule_path],
            out_dir=tmp_path / "out",
            hashcat_sample=0,
        )

        out_file = Path(tmp_path / "out" / "myrules_filtered.rule")
        assert out_file.exists()
        assert "myrules" in result["filtered_files"]

    def test_funnel_csv_created(self, tmp_path):
        rule_path = tmp_path / "rules.rule"
        rule_path.write_text("c\n$1\n")

        filter_rules(
            rule_files=[rule_path],
            out_dir=tmp_path / "out",
            hashcat_sample=0,
        )

        assert (tmp_path / "out" / "filter_funnel.csv").exists()

    def test_multiple_rule_files(self, tmp_path):
        r1 = tmp_path / "file1.rule"
        r2 = tmp_path / "file2.rule"
        r1.write_text("c\nl\n")
        r2.write_text("$1\n$2\n")

        result = filter_rules(
            rule_files=[r1, r2],
            out_dir=tmp_path / "out",
            hashcat_sample=0,
        )

        assert len(result["funnel"]) == 2
        assert len(result["filtered_files"]) == 2
