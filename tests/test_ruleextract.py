"""Tests for pwrules.ruleextract (Phase 3).

Covers the rule applier, the rule extractor, and round-trip validation.
The Hashcat parity test is automatically skipped when hashcat is not on PATH.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import FrozenSet

import pytest

from pwrules.ruleextract.applier import apply_function, apply_rule, tokenize_rule
from pwrules.ruleextract.extractor import (
    DEFAULT_LEET_MAP,
    _detect_case_and_leet,
    _strip_outer_nonalpha,
    infer_rule,
    parity_check,
    select_base,
)


# ===========================================================================
# Applier tests
# ===========================================================================

class TestTokenizeRule:
    def test_single_no_param(self):
        assert tokenize_rule("c") == [("c", "")]

    def test_multi_token(self):
        assert tokenize_rule("c sa@ $1") == [("c", ""), ("s", "a@"), ("$", "1")]

    def test_noop(self):
        assert tokenize_rule(":") == [(":", "")]

    def test_empty_string(self):
        assert tokenize_rule("") == []

    def test_whitespace_only(self):
        assert tokenize_rule("   ") == []


class TestApplyFunction:
    def test_noop(self):
        assert apply_function("password", ":", "") == "password"

    def test_lowercase(self):
        assert apply_function("PASSWORD", "l", "") == "password"

    def test_uppercase(self):
        assert apply_function("password", "u", "") == "PASSWORD"

    def test_capitalize(self):
        assert apply_function("password", "c", "") == "Password"

    def test_capitalize_already_upper(self):
        assert apply_function("PASSWORD", "c", "") == "Password"

    def test_inverse_capitalize(self):
        assert apply_function("Password", "C", "") == "pASSWORD"

    def test_toggle(self):
        assert apply_function("Password", "t", "") == "pASSWORD"

    def test_toggle_at_position(self):
        assert apply_function("password", "T", "0") == "Password"

    def test_reverse(self):
        assert apply_function("password", "r", "") == "drowssap"

    def test_duplicate(self):
        assert apply_function("abc", "d", "") == "abcabc"

    def test_reflect(self):
        assert apply_function("abc", "f", "") == "abccba"

    def test_rotate_left(self):
        assert apply_function("abcd", "{", "") == "bcda"

    def test_rotate_right(self):
        assert apply_function("abcd", "}", "") == "dabc"

    def test_delete_first(self):
        assert apply_function("password", "[", "") == "assword"

    def test_delete_last(self):
        assert apply_function("password", "]", "") == "passwor"

    def test_swap_first_two(self):
        assert apply_function("abc", "k", "") == "bac"

    def test_swap_last_two(self):
        assert apply_function("abc", "K", "") == "acb"

    def test_duplicate_each(self):
        assert apply_function("ab", "q", "") == "aabb"

    def test_append(self):
        assert apply_function("pass", "$", "1") == "pass1"

    def test_prepend(self):
        assert apply_function("pass", "^", "1") == "1pass"

    def test_substitute(self):
        assert apply_function("password", "s", "a@") == "p@ssword"

    def test_substitute_all_occurrences(self):
        assert apply_function("abab", "s", "ab") == "bbbb"

    def test_insert(self):
        assert apply_function("abc", "i", "1x") == "axbc"

    def test_overwrite(self):
        assert apply_function("abc", "o", "1x") == "axc"

    def test_swap_positions(self):
        assert apply_function("abcd", "*", "13") == "adcb"

    def test_delete_at_position(self):
        assert apply_function("abcd", "D", "1") == "acd"

    def test_purge(self):
        assert apply_function("password", "@", "s") == "paword"

    def test_duplicate_first_n(self):
        assert apply_function("abc", "z", "2") == "aaabc"

    def test_duplicate_last_n(self):
        assert apply_function("abc", "Z", "2") == "abccc"

    def test_extract(self):
        assert apply_function("password", "x", "23") == "ssw"

    def test_unknown_function_passthrough(self):
        assert apply_function("word", "~", "") == "word"


class TestApplyRule:
    def test_capitalize_and_append(self):
        assert apply_rule("password", "c $1 $2 $3") == "Password123"

    def test_leet_and_suffix(self):
        assert apply_rule("password", "sa@ $1") == "p@ssword1"

    def test_case_and_leet(self):
        assert apply_rule("password", "c sa@") == "P@ssword"

    def test_full_combo(self):
        assert apply_rule("password", "c sa@ $1 $2 $3") == "P@ssword123"

    def test_reverse(self):
        assert apply_rule("password", "r") == "drowssap"

    def test_noop(self):
        assert apply_rule("hello", ":") == "hello"

    def test_prefix_and_suffix(self):
        assert apply_rule("word", "^1 $!") == "1word!"

    def test_prepend_multi(self):
        # To prepend "123": ^3 ^2 ^1 (reversed order)
        assert apply_rule("word", "^3 ^2 ^1") == "123word"

    def test_dragon(self):
        assert apply_rule("dragon", "c so0 $1 $2 $3") == "Dr0gon123"


# ===========================================================================
# Extractor tests
# ===========================================================================

WORDLIST: FrozenSet[str] = frozenset([
    "password", "dragon", "hello", "sunshine", "monkey",
    "shadow", "master", "letmein", "love", "football",
])


class TestStripOuterNonAlpha:
    def test_no_stripping_needed(self):
        prefix, core, suffix = _strip_outer_nonalpha("password")
        assert prefix == "" and core == "password" and suffix == ""

    def test_numeric_suffix(self):
        prefix, core, suffix = _strip_outer_nonalpha("password123")
        assert prefix == "" and core == "password" and suffix == "123"

    def test_numeric_prefix(self):
        prefix, core, suffix = _strip_outer_nonalpha("123password")
        assert prefix == "123" and core == "password" and suffix == ""

    def test_both(self):
        prefix, core, suffix = _strip_outer_nonalpha("123password456")
        assert prefix == "123" and core == "password" and suffix == "456"

    def test_leet_in_core(self):
        _, core, _ = _strip_outer_nonalpha("P@ssword123")
        assert core == "P@ssword"

    def test_all_digits(self):
        prefix, core, suffix = _strip_outer_nonalpha("12345")
        assert core == "" and prefix == "12345"

    def test_empty(self):
        prefix, core, suffix = _strip_outer_nonalpha("")
        assert prefix == "" and core == "" and suffix == ""


class TestSelectBase:
    def test_exact_match(self):
        assert select_base("password", WORDLIST) == "password"

    def test_suffix_stripped(self):
        assert select_base("password123", WORDLIST) == "password"

    def test_capitalize(self):
        assert select_base("Password123", WORDLIST) == "password"

    def test_leet_detected(self):
        assert select_base("p@ssword", WORDLIST) == "password"

    def test_no_match(self):
        assert select_base("xyzabc123", WORDLIST) is None

    def test_prefix_and_suffix(self):
        assert select_base("123password456", WORDLIST) == "password"


class TestInferRule:
    def test_noop(self):
        rule = infer_rule("password", "password")
        assert rule == ":"
        assert apply_rule("password", rule) == "password"

    def test_capitalize(self):
        rule = infer_rule("password", "Password")
        assert rule is not None
        assert apply_rule("password", rule) == "Password"

    def test_uppercase(self):
        rule = infer_rule("dragon", "DRAGON")
        assert rule is not None
        assert apply_rule("dragon", rule) == "DRAGON"

    def test_leet_only(self):
        rule = infer_rule("password", "p@ssword")
        assert rule is not None
        assert apply_rule("password", rule) == "p@ssword"

    def test_capitalize_and_leet(self):
        rule = infer_rule("password", "P@ssword")
        assert rule is not None
        assert apply_rule("password", rule) == "P@ssword"

    def test_suffix(self):
        rule = infer_rule("password", "password123")
        assert rule is not None
        assert apply_rule("password", rule) == "password123"

    def test_prefix(self):
        rule = infer_rule("password", "123password")
        assert rule is not None
        assert apply_rule("password", rule) == "123password"

    def test_full_combo(self):
        rule = infer_rule("password", "P@ssword123")
        assert rule is not None
        assert apply_rule("password", rule) == "P@ssword123"

    def test_reversal(self):
        rule = infer_rule("dragon", "nogard")
        assert rule is not None
        assert apply_rule("dragon", rule) == "nogard"

    def test_reversal_with_suffix(self):
        rule = infer_rule("password", "drowssap123")
        assert rule is not None
        assert apply_rule("password", rule) == "drowssap123"

    def test_duplication(self):
        rule = infer_rule("hello", "hellohello")
        assert rule is not None
        assert apply_rule("hello", rule) == "hellohello"

    def test_no_match_returns_none(self):
        # "xkcd" is not transformable from "password" by supported ops.
        rule = infer_rule("password", "xkcd")
        assert rule is None

    def test_leet_uppercase(self):
        # After 'u': DRAGON → DR@G0N needs sA@ and sO0.
        rule = infer_rule("dragon", "DR@G0N")
        assert rule is not None
        assert apply_rule("dragon", rule) == "DR@G0N"

    def test_prefix_and_suffix(self):
        rule = infer_rule("love", "123love456")
        assert rule is not None
        assert apply_rule("love", rule) == "123love456"


# ===========================================================================
# Parity test (skipped when hashcat absent)
# ===========================================================================

PARITY_TRIPLES = [
    ("password", ":", "password"),
    ("password", "c", "Password"),
    ("password", "u", "PASSWORD"),
    ("dragon",   "r", "nogard"),
    ("hello",    "d", "hellohello"),
    ("password", "sa@ $1", "p@ssword1"),
    ("password", "c sa@ $1 $2 $3", "P@ssword123"),
]


@pytest.mark.skipif(
    shutil.which("hashcat") is None,
    reason="hashcat not installed — parity test skipped",
)
def test_hashcat_parity():
    """Every Python-applied rule must produce the same output as hashcat --stdout."""
    passed, total = parity_check(PARITY_TRIPLES)
    assert total == len(PARITY_TRIPLES), "parity_check ran on wrong number of triples"
    assert passed == total, (
        f"Parity FAILED: {total - passed}/{total} rules differ from hashcat output. "
        "Fix the Python applier."
    )


# ===========================================================================
# Pipeline smoke test
# ===========================================================================

def test_extract_rules_pipeline(tmp_path: Path):
    """End-to-end pipeline on a tiny synthetic corpus."""
    # Phase 2 outputs: write tiny train + val splits.
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()

    train_passwords = [
        "password", "Password", "P@ssword", "P@ssword123",
        "dragon", "Dragon", "DR@G0N", "nogard",
        "hello", "hellohello", "HELLO", "hello123",
        "sunshine", "sunshine1", "Sunshine",
        "monkey", "monkey!", "123monkey",
    ]
    val_passwords = ["letmein", "Letmein", "letmein1", "shadow", "Shadow1"]

    (clean_dir / "train.txt").write_text("\n".join(train_passwords), encoding="utf-8")
    (clean_dir / "val.txt").write_text("\n".join(val_passwords), encoding="utf-8")
    (clean_dir / "test.txt").write_text("master\nfootball\n", encoding="utf-8")

    from pwrules.ruleextract import extract_rules

    result = extract_rules(
        clean_dir=clean_dir,
        out_dir=tmp_path / "rules",
    )

    out_dir = Path(result["out_dir"])
    assert (out_dir / "rules_dataset.jsonl").exists()
    assert (out_dir / "coverage_report.json").exists()
    assert (out_dir / "rule_frequency.csv").exists()
    assert (out_dir / "instructions_train.jsonl").exists()
    assert (out_dir / "instructions_val.jsonl").exists()

    # Load and spot-check triples.
    triples = result["triples"]
    assert len(triples) > 0

    for t in triples:
        assert "base" in t and "rule" in t and "password" in t
        # Round-trip validation: every triple MUST reproduce exactly.
        assert apply_rule(t["base"], t["rule"]) == t["password"], (
            f"Round-trip FAILED: apply_rule({t['base']!r}, {t['rule']!r}) "
            f"→ {apply_rule(t['base'], t['rule'])!r} ≠ {t['password']!r}"
        )

    # Coverage report sanity.
    report = result["coverage_report"]
    assert report["train"]["coverage_pct"] >= 0.0
    assert report["val"]["coverage_pct"] >= 0.0

    # Instruction files are valid JSONL.
    inst_train = (out_dir / "instructions_train.jsonl").read_text().strip().splitlines()
    for line in inst_train:
        rec = json.loads(line)
        assert "input" in rec and "output" in rec
