"""Tests for the slug-agnostic path discovery module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pwrules import paths


@pytest.fixture()
def kaggle_env(tmp_path, monkeypatch):
    """Simulate a Kaggle layout with /input (nested slug) and /working roots."""
    inp = tmp_path / "input"
    work = tmp_path / "working"
    # A deeply nested dataset slug, like /kaggle/input/datasets/<user>/rockyou/.
    (inp / "datasets" / "someuser" / "rockyou").mkdir(parents=True)
    (inp / "datasets" / "someuser" / "rockyou" / "rockyou.txt").write_text("123456\n")
    work.mkdir()
    monkeypatch.setenv("PWRULES_INPUT", str(inp))
    monkeypatch.setenv("PWRULES_WORKING", str(work))
    # Clear any artifact pins from the host environment.
    for k in list(os.environ):
        if k.startswith("PWRULES_") and k not in {"PWRULES_INPUT", "PWRULES_WORKING"}:
            monkeypatch.delenv(k, raising=False)
    return inp, work


def test_finds_corpus_under_nested_slug(kaggle_env):
    found = paths.corpus()
    assert found.name == "rockyou.txt"
    assert found.read_text().strip() == "123456"


def test_find_file_missing_raises(kaggle_env):
    with pytest.raises(FileNotFoundError):
        paths.find_file("does_not_exist.txt")


def test_find_file_optional_returns_none(kaggle_env):
    assert paths.find_file("nope.jsonl", required=False) is None


def test_env_override_pins_path(kaggle_env, tmp_path, monkeypatch):
    pinned = tmp_path / "custom_rockyou.txt"
    pinned.write_text("pinned\n")
    monkeypatch.setenv("PWRULES_ROCKYOU", str(pinned))
    assert paths.corpus() == pinned


def test_env_override_bad_path_raises(kaggle_env, monkeypatch):
    monkeypatch.setenv("PWRULES_ROCKYOU", "/no/such/file.txt")
    with pytest.raises(FileNotFoundError):
        paths.corpus()


def test_working_root_preferred_over_input(kaggle_env):
    """Same-named file in /working should win over /input."""
    _, work = kaggle_env
    (work / "rockyou.txt").write_text("from-working\n")
    assert paths.corpus().read_text().strip() == "from-working"


def test_dir_with_returns_parent(kaggle_env):
    _, work = kaggle_env
    clean = work / "clean"
    clean.mkdir()
    (clean / "test_checksum.txt").write_text("abc\n")
    assert paths.clean_dir() == clean


def test_excludes_repo_test_fixtures(kaggle_env):
    """Files under a pwrules/tests/ path must be ignored by discovery."""
    inp, _ = kaggle_env
    fixture = inp / "pwrules" / "tests" / "fixtures"
    fixture.mkdir(parents=True)
    (fixture / "train.txt").write_text("fixture\n")
    assert paths.train_txt(required=False) is None


def test_out_creates_working_subdir(kaggle_env):
    _, work = kaggle_env
    p = paths.out("clean")
    assert p == work / "clean"
    assert p.is_dir()


def test_summary_lists_resolved_only(kaggle_env):
    s = paths.summary()
    assert s.get("corpus", "").endswith("rockyou.txt")
    # Nothing else has been created, so other artifacts are absent.
    assert "results_dir" not in s
