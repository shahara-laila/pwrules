"""Tests for pwrules.conditioning (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from pwrules.conditioning import (
    assign_synthetic_attributes,
    build_targeted_dataset,
    generate_synthetic_users,
    split_users,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TRIPLES: List[dict] = [
    {"base": "password", "rule": "c", "password": "Password"},
    {"base": "password", "rule": "sa@ $1", "password": "p@ssword1"},
    {"base": "dragon",   "rule": "c so0", "password": "Dr0gon"},
    {"base": "hello",    "rule": "d",     "password": "hellohello"},
    {"base": "sunshine", "rule": "u",     "password": "SUNSHINE"},
    {"base": "monkey",   "rule": "$1 $2", "password": "monkey12"},
    {"base": "shadow",   "rule": "r",     "password": "wodahs"},
    {"base": "love",     "rule": "c $3",  "password": "Love3"},
    {"base": "football", "rule": ":"   ,  "password": "football"},
    {"base": "master",   "rule": "c sa@", "password": "M@ster"},
]

SEED = 42


@pytest.fixture
def rules_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "rules_dataset.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t in SAMPLE_TRIPLES:
            f.write(json.dumps(t) + "\n")
    return p


# ---------------------------------------------------------------------------
# generate_synthetic_users
# ---------------------------------------------------------------------------

def test_generate_users_count():
    users = generate_synthetic_users(10, SEED)
    assert len(users) == 10


def test_generate_users_fields():
    users = generate_synthetic_users(5, SEED)
    for u in users:
        assert "user_id" in u
        assert "name" in u
        assert "birth_year" in u
        assert "interest" in u
        assert 1950 <= int(u["birth_year"]) <= 2005


def test_generate_users_deterministic():
    users_a = generate_synthetic_users(20, seed=1)
    users_b = generate_synthetic_users(20, seed=1)
    assert users_a == users_b


def test_generate_users_different_seeds():
    users_a = generate_synthetic_users(20, seed=1)
    users_b = generate_synthetic_users(20, seed=2)
    assert users_a != users_b


def test_generate_users_unique_ids():
    users = generate_synthetic_users(50, SEED)
    ids = [u["user_id"] for u in users]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# assign_synthetic_attributes
# ---------------------------------------------------------------------------

def test_assign_attributes_extends_triples():
    augmented, user_map = assign_synthetic_attributes(SAMPLE_TRIPLES, n_users=5, seed=SEED)
    assert len(augmented) == len(SAMPLE_TRIPLES)
    for t in augmented:
        assert "user_id" in t
        assert "name" in t
        assert "birth_year" in t
        assert "interest" in t
        assert t.get("synthetic") is True


def test_assign_attributes_deterministic():
    a1, _ = assign_synthetic_attributes(SAMPLE_TRIPLES, n_users=5, seed=SEED)
    a2, _ = assign_synthetic_attributes(SAMPLE_TRIPLES, n_users=5, seed=SEED)
    assert [t["user_id"] for t in a1] == [t["user_id"] for t in a2]


def test_assign_attributes_preserves_original_fields():
    augmented, _ = assign_synthetic_attributes(SAMPLE_TRIPLES, n_users=5, seed=SEED)
    for orig, aug in zip(SAMPLE_TRIPLES, augmented):
        assert aug["base"] == orig["base"]
        assert aug["rule"] == orig["rule"]
        assert aug["password"] == orig["password"]


# ---------------------------------------------------------------------------
# split_users
# ---------------------------------------------------------------------------

def test_split_users_disjoint():
    augmented, _ = assign_synthetic_attributes(SAMPLE_TRIPLES, n_users=5, seed=SEED)
    train, test_users = split_users(augmented, test_ratio=0.2, seed=SEED)

    train_user_ids = {t["user_id"] for t in train}
    test_user_ids = {u["user_id"] for u in test_users}
    assert not train_user_ids & test_user_ids, (
        f"Disjoint FAILED: {train_user_ids & test_user_ids}"
    )


def test_split_users_all_accounted():
    augmented, _ = assign_synthetic_attributes(SAMPLE_TRIPLES, n_users=5, seed=SEED)
    train, test_users = split_users(augmented, test_ratio=0.2, seed=SEED)

    test_passwords = {p for u in test_users for p in u["passwords"]}
    train_passwords = {t["password"] for t in train}
    # Every password must appear in exactly one partition.
    all_passwords = {t["password"] for t in augmented}
    assert (train_passwords | test_passwords) == all_passwords


def test_split_users_test_users_have_attrs():
    augmented, _ = assign_synthetic_attributes(SAMPLE_TRIPLES, n_users=5, seed=SEED)
    _, test_users = split_users(augmented, test_ratio=0.2, seed=SEED)
    for u in test_users:
        assert "user_id" in u
        assert "attrs" in u
        assert "passwords" in u
        assert len(u["passwords"]) > 0


# ---------------------------------------------------------------------------
# build_targeted_dataset (full pipeline)
# ---------------------------------------------------------------------------

def test_build_synthetic_creates_outputs(tmp_path: Path, rules_jsonl: Path):
    result = build_targeted_dataset(
        rules_jsonl=rules_jsonl,
        out_dir=tmp_path / "targeted",
        mode="synthetic",
        n_synthetic_users=10,
        test_user_ratio=0.2,
    )
    out = Path(result["out_dir"])
    assert (out / "targeted_dataset.jsonl").exists()
    assert (out / "target_users_test.jsonl").exists()
    assert (out / "split_manifest.json").exists()


def test_build_synthetic_disjoint(tmp_path: Path, rules_jsonl: Path):
    result = build_targeted_dataset(
        rules_jsonl=rules_jsonl,
        out_dir=tmp_path / "targeted",
        mode="synthetic",
        n_synthetic_users=10,
        test_user_ratio=0.2,
    )
    manifest = json.loads(
        (Path(result["out_dir"]) / "split_manifest.json").read_text()
    )
    assert manifest["disjoint"] is True
    assert manifest["n_test_users"] >= 1
    assert manifest["n_train_users"] >= 1


def test_build_synthetic_instruction_format(tmp_path: Path, rules_jsonl: Path):
    result = build_targeted_dataset(
        rules_jsonl=rules_jsonl,
        out_dir=tmp_path / "targeted",
        mode="synthetic",
        n_synthetic_users=10,
        test_user_ratio=0.2,
    )
    out = Path(result["out_dir"])
    lines = (out / "targeted_dataset.jsonl").read_text().strip().splitlines()
    for line in lines:
        rec = json.loads(line)
        assert "input" in rec and "output" in rec
        # Targeted instructions must mention the user profile.
        assert "User profile" in rec["input"]


def test_build_synthetic_manifest_mode(tmp_path: Path, rules_jsonl: Path):
    result = build_targeted_dataset(
        rules_jsonl=rules_jsonl,
        out_dir=tmp_path / "targeted",
        mode="synthetic",
        n_synthetic_users=10,
        test_user_ratio=0.2,
    )
    manifest = json.loads(
        (Path(result["out_dir"]) / "split_manifest.json").read_text()
    )
    assert manifest["mode"] == "synthetic"
    assert manifest["synthetic"] is True


def test_build_invalid_mode_raises(tmp_path: Path, rules_jsonl: Path):
    with pytest.raises(ValueError, match="Unknown mode"):
        build_targeted_dataset(
            rules_jsonl=rules_jsonl,
            out_dir=tmp_path / "targeted",
            mode="invalid_mode",
        )


def test_build_real_without_csv_raises(tmp_path: Path, rules_jsonl: Path):
    with pytest.raises(ValueError, match="real_csv_path"):
        build_targeted_dataset(
            rules_jsonl=rules_jsonl,
            out_dir=tmp_path / "targeted",
            mode="real",
            real_csv_path=None,
        )


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_cli_runs(tmp_path: Path, rules_jsonl: Path):
    from pwrules.conditioning.__main__ import main
    main([
        "--rules", str(rules_jsonl),
        "--out",   str(tmp_path / "out"),
        "--mode",  "synthetic",
        "--n-users", "5",
        "--log-level", "WARNING",
    ])
    assert (tmp_path / "out" / "targeted_dataset.jsonl").exists()
    assert (tmp_path / "out" / "target_users_test.jsonl").exists()
