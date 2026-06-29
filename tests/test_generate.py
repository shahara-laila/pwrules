"""Tests for pwrules.generate (Phase 6).

GPU-dependent tests (model loading, actual generation) are skipped when CUDA
is unavailable.  All helper functions (file writing, stats, diversity) are
CPU-safe and tested unconditionally.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from pwrules.generate import (
    _DEFAULT_PROBE_WORDS,
    compute_diversity_stats,
    save_diversity_stats,
    write_rule_file,
)


# ---------------------------------------------------------------------------
# write_rule_file
# ---------------------------------------------------------------------------

def test_write_rule_file_creates_file(tmp_path: Path):
    rules = ["c", "sa@ $1", "r", ":"]
    write_rule_file(rules, tmp_path / "test.rule")
    assert (tmp_path / "test.rule").exists()


def test_write_rule_file_one_rule_per_line(tmp_path: Path):
    rules = ["c", "sa@ $1", "r"]
    path = tmp_path / "test.rule"
    write_rule_file(rules, path)
    lines = path.read_text().strip().splitlines()
    assert lines == rules


def test_write_rule_file_empty(tmp_path: Path):
    write_rule_file([], tmp_path / "empty.rule")
    assert (tmp_path / "empty.rule").exists()
    assert (tmp_path / "empty.rule").read_text() == ""


def test_write_rule_file_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "subdir" / "nested" / "test.rule"
    write_rule_file(["c"], path)
    assert path.exists()


# ---------------------------------------------------------------------------
# compute_diversity_stats
# ---------------------------------------------------------------------------

def test_compute_diversity_stats_structure():
    untargeted = ["c", "u", "r", "d", "c"]  # 'c' appears twice → 4 unique
    targeted = {
        "user_001": ["sa@", "c", "u"],
        "user_002": ["r", "d"],
    }
    stats = compute_diversity_stats(untargeted, targeted)

    assert "untargeted" in stats
    assert "targeted" in stats
    assert "combined" in stats

    assert stats["untargeted"]["n_total"] == 5
    assert stats["untargeted"]["n_unique"] == 4  # 'c' deduplicated

    assert stats["targeted"]["n_users"] == 2
    assert stats["targeted"]["n_total"] == 5

    combined = stats["combined"]
    assert combined["n_total"] == 10
    assert isinstance(combined["length_distribution"], dict)
    assert isinstance(combined["top_20_rules"], list)


def test_compute_diversity_stats_empty():
    stats = compute_diversity_stats([], {})
    assert stats["untargeted"]["n_unique"] == 0
    assert stats["targeted"]["n_users"] == 0
    assert stats["combined"]["n_unique"] == 0


def test_compute_diversity_stats_top_rules():
    rules = ["c"] * 5 + ["r"] * 3 + ["u"] * 1
    stats = compute_diversity_stats(rules, {})
    top = dict(stats["combined"]["top_20_rules"])
    assert top["c"] == 5
    assert top["r"] == 3


# ---------------------------------------------------------------------------
# save_diversity_stats
# ---------------------------------------------------------------------------

def test_save_diversity_stats_creates_json(tmp_path: Path):
    untargeted = ["c", "r", "u"]
    stats = compute_diversity_stats(untargeted, {})
    save_diversity_stats(stats, tmp_path)
    assert (tmp_path / "generation_stats.json").exists()


def test_save_diversity_stats_creates_plot(tmp_path: Path):
    untargeted = ["c", "r", "sa@"]
    stats = compute_diversity_stats(untargeted, {})
    save_diversity_stats(stats, tmp_path)
    assert (tmp_path / "rule_length_dist.png").exists()


def test_save_diversity_stats_json_valid(tmp_path: Path):
    stats = compute_diversity_stats(["c", "r"], {"u001": ["u"]})
    save_diversity_stats(stats, tmp_path)
    loaded = json.loads((tmp_path / "generation_stats.json").read_text())
    assert loaded["untargeted"]["n_total"] == 2
    assert loaded["targeted"]["n_users"] == 1


# ---------------------------------------------------------------------------
# Default probe words
# ---------------------------------------------------------------------------

def test_default_probe_words_not_empty():
    assert len(_DEFAULT_PROBE_WORDS) >= 10


def test_default_probe_words_are_strings():
    assert all(isinstance(w, str) for w in _DEFAULT_PROBE_WORDS)


# ---------------------------------------------------------------------------
# CPU stub: generate_untargeted and generate_targeted with a mock model
# ---------------------------------------------------------------------------

class _MockTokenizer:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        mock = MagicMock()
        mock.__getitem__ = MagicMock(side_effect={"input_ids": MagicMock(shape=(1, 5))}.__getitem__)
        mock.to = MagicMock(return_value=mock)
        return mock

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "prompt"

    def decode(self, ids, skip_special_tokens=False):
        return "c"  # always returns the no-op rule


class _MockModel:
    device = "cpu"

    def generate(self, **kwargs):
        import numpy as np
        return [[0, 99, 100]]  # dummy output ids

    def __call__(self, *args, **kwargs):
        return MagicMock()


def test_generate_untargeted_stub():
    """Verify generate_untargeted returns unique rules up to budget."""
    from pwrules.generate import generate_untargeted

    # Use a mock tokenizer that always returns "c" as the generated rule.
    class _Counter:
        count = 0

        def __call__(self, text, return_tensors=None, **kwargs):
            self.count += 1
            m = MagicMock()
            m.to = MagicMock(return_value=m)
            m.__getitem__ = MagicMock(side_effect=lambda k: MagicMock(shape=(1, 5)))
            return m

    tokenizer = _MockTokenizer()
    model = _MockModel()

    # Patch tokenizer.decode to return distinct rules based on call count.
    call_count = [0]

    def _decode(ids, skip_special_tokens=False):
        call_count[0] += 1
        return f"rule_{call_count[0]}"  # unique rules each time

    tokenizer.decode = _decode

    gen_cfg = {"max_new_tokens": 32, "temperature": 0.8, "top_p": 0.95, "do_sample": True}

    with patch("pwrules.generate._generate_single") as mock_gen:
        # Return unique rules to hit budget quickly.
        mock_gen.side_effect = [f"rule_{i}" for i in range(200)]
        rules = generate_untargeted(
            model, tokenizer,
            budget=10,
            gen_cfg=gen_cfg,
            probe_words=["password", "dragon"] * 50,
            seed=42,
        )

    assert len(rules) == 10
    assert len(set(rules)) == 10  # all unique


def test_generate_untargeted_deduplicates():
    """Repeated rules must not exceed budget."""
    from pwrules.generate import generate_untargeted

    model = _MockModel()
    tokenizer = _MockTokenizer()

    with patch("pwrules.generate._generate_single", return_value="c"):
        # All generated rules are "c" → only 1 unique; budget won't be reached.
        rules = generate_untargeted(
            model, tokenizer,
            budget=5,
            gen_cfg={"max_new_tokens": 32, "temperature": 0.8, "top_p": 0.95, "do_sample": True},
            probe_words=["password"] * 10,
            seed=42,
        )
    # Only 1 unique rule, so must be ≤ 1 even though budget=5.
    assert len(rules) <= 1


def test_generate_targeted_stub():
    """generate_targeted must return a dict keyed by user_id."""
    from pwrules.generate import generate_targeted

    model = _MockModel()
    tokenizer = _MockTokenizer()

    test_users = [
        {"user_id": "u001", "attrs": {"name": "alice", "birth_year": 1990, "interest": "music"}},
        {"user_id": "u002", "attrs": {"name": "bob",   "birth_year": 1985, "interest": "soccer"}},
    ]

    counter = [0]

    def _side(*args, **kwargs):
        counter[0] += 1
        return f"rule_{counter[0]}"

    with patch("pwrules.generate._generate_single", side_effect=_side):
        result = generate_targeted(
            model, tokenizer,
            test_users=test_users,
            budget_per_user=3,
            gen_cfg={"max_new_tokens": 32, "temperature": 0.8, "top_p": 0.95, "do_sample": True},
            probe_words=["password"] * 20,
            seed=42,
        )

    assert "u001" in result and "u002" in result
    for uid, rules in result.items():
        assert len(rules) <= 3
        assert len(set(rules)) == len(rules)  # deduplicated


# ---------------------------------------------------------------------------
# Module import test (CPU-safe)
# ---------------------------------------------------------------------------

def test_generate_module_imports():
    import importlib
    mod = importlib.import_module("pwrules.generate")
    assert hasattr(mod, "generate_rules")
    assert hasattr(mod, "generate_untargeted")
    assert hasattr(mod, "generate_targeted")
    assert hasattr(mod, "write_rule_file")


# ---------------------------------------------------------------------------
# GPU-only: full pipeline test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or not __import__("torch").cuda.is_available(),
    reason="GPU not available — full generation test requires Kaggle GPU.",
)
def test_generate_rules_pipeline_gpu(tmp_path: Path):
    """Full pipeline smoke test — only runs on Kaggle with GPU."""
    from pwrules.generate import generate_rules

    # This requires a real adapter; skip if none is found.
    adapter = tmp_path / "adapter"
    if not adapter.exists():
        pytest.skip("No adapter directory — run Phase 5 first on Kaggle.")

    result = generate_rules(
        adapter_dir=adapter,
        out_dir=tmp_path / "out",
        budget=10,
    )
    out = Path(result["out_dir"])
    assert (out / "rules" / "llm_untargeted.rule").exists()
    assert len(result["untargeted_rules"]) <= 10
