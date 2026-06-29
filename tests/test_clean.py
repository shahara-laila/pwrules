"""Tests for pwrules.clean (Phase 2).

All tests run on tiny synthetic corpora with no real passwords.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pwrules.clean import (
    clean_password,
    compute_stats,
    iter_clean,
    save_stats,
    split_corpus,
    verify_test_checksum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_CORPUS = [
    "password",
    "password",          # exact duplicate — should be removed
    "123456",
    "letmein",
    "dragon",
    "sunshine",
    "princess",
    "abc123",
    "monkey",
    "shadow",
    "master",
    "football",
    "qwerty",
    "iloveyou",
    "batman",
    "superman",
    "starwars",
    "hello",
    "welcome",
    "charlie",
]
# Expected unique count after dedup: 19 (password appears twice → 1)
EXPECTED_UNIQUE = 19

SPLIT_CFG = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
    "by_user": False,
}

SEED = 42


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text("\n".join(SYNTHETIC_CORPUS) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# clean_password unit tests
# ---------------------------------------------------------------------------

def test_clean_password_basic():
    assert clean_password(b"password\n") == "password"


def test_clean_password_crlf():
    assert clean_password(b"password\r\n") == "password"


def test_clean_password_control_chars():
    assert clean_password(b"pass\x00word\n") == "password"


def test_clean_password_invalid_utf8_repaired():
    # Invalid UTF-8 byte — should not raise; falls back to latin-1.
    result = clean_password(b"caf\xe9\n")
    assert result is not None
    assert len(result) > 0


def test_clean_password_empty_returns_none():
    assert clean_password(b"\n") is None
    assert clean_password(b"\r\n") is None
    assert clean_password(b"") is None


def test_clean_password_string_input():
    assert clean_password("password\n") == "password"


# ---------------------------------------------------------------------------
# iter_clean deduplication
# ---------------------------------------------------------------------------

def test_iter_clean_deduplicates(corpus_file: Path):
    result = list(iter_clean(corpus_file))
    assert len(result) == EXPECTED_UNIQUE


def test_iter_clean_order_preserved(corpus_file: Path):
    result = list(iter_clean(corpus_file))
    # "password" should appear only once, at its first position (index 0).
    assert result[0] == "password"


def test_iter_clean_filter_length(corpus_file: Path):
    result = list(iter_clean(corpus_file, min_len=6, max_len=10, filter_enabled=True))
    for pw in result:
        assert 6 <= len(pw) <= 10


def test_iter_clean_filter_disabled_by_default(corpus_file: Path):
    # Short passwords like "abc123" (len 6) must survive with default (no filter).
    result = list(iter_clean(corpus_file))
    assert "abc123" in result


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

def test_compute_stats_total(corpus_file: Path):
    passwords = list(iter_clean(corpus_file))
    stats = compute_stats(passwords)
    assert stats["total"] == EXPECTED_UNIQUE


def test_compute_stats_length_counter(corpus_file: Path):
    passwords = list(iter_clean(corpus_file))
    stats = compute_stats(passwords)
    lc = stats["length_counter"]
    assert isinstance(lc, dict)
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in lc.items())


def test_compute_stats_char_class_counts(corpus_file: Path):
    passwords = list(iter_clean(corpus_file))
    stats = compute_stats(passwords)
    cc = stats["char_class_counts"]
    assert "lower" in cc and "digit" in cc


def test_compute_stats_top_tokens(corpus_file: Path):
    passwords = list(iter_clean(corpus_file))
    stats = compute_stats(passwords)
    assert isinstance(stats["top_100_tokens"], list)


# ---------------------------------------------------------------------------
# save_stats
# ---------------------------------------------------------------------------

def test_save_stats_creates_files(tmp_path: Path):
    passwords = ["abc", "abc123", "hello", "world", "password"]
    stats = compute_stats(passwords)
    save_stats(stats, tmp_path / "stats")
    stats_dir = tmp_path / "stats"
    assert (stats_dir / "length_histogram.csv").exists()
    assert (stats_dir / "char_class_composition.csv").exists()
    assert (stats_dir / "stats.png").exists()


# ---------------------------------------------------------------------------
# split_corpus
# ---------------------------------------------------------------------------

def test_split_sizes(tmp_path: Path):
    passwords = list(iter_clean(
        Path(__file__).parent.parent / "tests" / "corpus_tmp.txt"
        if False else Path(__file__)  # force tmp path below
    ))
    # Manually create the list instead of reading a file.
    passwords = [f"pw{i}" for i in range(100)]
    splits = split_corpus(passwords, SPLIT_CFG, tmp_path / "splits", SEED)
    total = sum(len(v) for v in splits.values())
    assert total == 100


def test_split_disjoint(tmp_path: Path):
    passwords = [f"pw{i}" for i in range(50)]
    splits = split_corpus(passwords, SPLIT_CFG, tmp_path / "splits", SEED)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert not train_set & val_set, "train ∩ val must be empty"
    assert not train_set & test_set, "train ∩ test must be empty"
    assert not val_set & test_set, "val ∩ test must be empty"


def test_split_files_created(tmp_path: Path):
    passwords = [f"pw{i}" for i in range(30)]
    split_dir = tmp_path / "splits"
    split_corpus(passwords, SPLIT_CFG, split_dir, SEED)
    assert (split_dir / "train.txt").exists()
    assert (split_dir / "val.txt").exists()
    assert (split_dir / "test.txt").exists()
    assert (split_dir / "test_checksum.txt").exists()
    assert (split_dir / "split_manifest.json").exists()


def test_split_manifest_json(tmp_path: Path):
    passwords = [f"pw{i}" for i in range(30)]
    split_dir = tmp_path / "splits"
    splits = split_corpus(passwords, SPLIT_CFG, split_dir, SEED)
    manifest = json.loads((split_dir / "split_manifest.json").read_text())
    assert "sizes" in manifest
    assert manifest["sizes"]["train"] == len(splits["train"])


def test_verify_test_checksum(tmp_path: Path):
    passwords = [f"pw{i}" for i in range(30)]
    split_dir = tmp_path / "splits"
    split_corpus(passwords, SPLIT_CFG, split_dir, SEED)
    assert verify_test_checksum(split_dir) is True


def test_verify_checksum_fails_on_tamper(tmp_path: Path):
    passwords = [f"pw{i}" for i in range(30)]
    split_dir = tmp_path / "splits"
    split_corpus(passwords, SPLIT_CFG, split_dir, SEED)
    # Tamper with the test split.
    test_file = split_dir / "test.txt"
    test_file.write_text("tampered_password\n", encoding="utf-8")
    assert verify_test_checksum(split_dir) is False


def test_split_by_user(tmp_path: Path):
    """Users must not appear in more than one split."""
    passwords = [f"pw_{i}" for i in range(60)]
    # Assign each password to a user; 20 users, 3 passwords each.
    user_map = {f"pw_{i}": f"user_{i // 3}" for i in range(60)}
    cfg = {**SPLIT_CFG, "by_user": True}
    splits = split_corpus(passwords, cfg, tmp_path / "splits", SEED, user_map)
    # Collect users per split.
    train_users = {user_map[p] for p in splits["train"] if p in user_map}
    test_users = {user_map[p] for p in splits["test"] if p in user_map}
    assert not train_users & test_users, "User leaked from train into test!"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_cli_runs(tmp_path: Path, corpus_file: Path):
    from pwrules.clean.__main__ import main
    main([
        "--input", str(corpus_file),
        "--out", str(tmp_path / "out"),
        "--log-level", "WARNING",
    ])
    assert (tmp_path / "out" / "train.txt").exists()
