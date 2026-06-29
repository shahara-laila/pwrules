"""Phase 2 — Corpus cleaning, statistics, and train/val/test splitting.

Pipeline
--------
1. Normalise to UTF-8 (drop/repair invalid bytes).
2. Strip ASCII control characters.
3. Remove exact duplicates (order-preserving).
4. Optional length / character-class filter (off by default, config-driven).
5. Compute and save statistics (length histogram, char-class composition).
6. Split into train / val / test per configs/protocol.yaml.
   - If ``by_user=True`` and a user field exists → split by user, assert zero overlap.
   - Else → split by row with a seeded shuffle.
7. Write a SHA-256 checksum of the frozen test split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe on Kaggle + CI)
import matplotlib.pyplot as plt

from pwrules.config import load_protocol, set_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default length bounds (off when filter_enabled=False in config).
DEFAULT_MIN_LEN: int = 4
DEFAULT_MAX_LEN: int = 64

# Regex for control characters (except \t and \n which are separators).
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# Character-class labels used in the stats report.
_CHARCLASS = {
    "lower": str.islower,
    "upper": str.isupper,
    "digit": str.isdigit,
}


# ---------------------------------------------------------------------------
# Single-line normalisation
# ---------------------------------------------------------------------------

def clean_password(raw: bytes | str) -> Optional[str]:
    """Normalise a single raw corpus line to a clean password string.

    Returns ``None`` if the line should be discarded (empty after cleaning).
    """
    if isinstance(raw, bytes):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode("latin-1")  # best-effort repair
    else:
        line = raw

    # Strip newline / carriage return.
    line = line.rstrip("\r\n")

    # Strip remaining ASCII control characters.
    line = _CTRL_RE.sub("", line)

    # NFC normalise (canonicalise Unicode combining chars).
    line = unicodedata.normalize("NFC", line)

    return line if line else None


# ---------------------------------------------------------------------------
# Corpus-level cleaning
# ---------------------------------------------------------------------------

def iter_clean(
    input_path: Path,
    min_len: int = 0,
    max_len: int = 0,
    filter_enabled: bool = False,
) -> Iterator[str]:
    """Yield deduplicated, cleaned passwords from *input_path*.

    Exact-duplicate removal is done in a single pass using a set; order is
    preserved (first occurrence kept).  The ``filter_enabled`` flag gates
    the optional length / charset filter to keep the protocol frozen.
    """
    seen: set[str] = set()
    with open(input_path, "rb") as fh:
        for raw_line in fh:
            pw = clean_password(raw_line)
            if pw is None:
                continue

            # Optional length filter.
            if filter_enabled:
                lo = min_len if min_len > 0 else DEFAULT_MIN_LEN
                hi = max_len if max_len > 0 else DEFAULT_MAX_LEN
                if not (lo <= len(pw) <= hi):
                    continue

            if pw not in seen:
                seen.add(pw)
                yield pw


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _char_class(ch: str) -> str:
    if ch.islower():
        return "lower"
    if ch.isupper():
        return "upper"
    if ch.isdigit():
        return "digit"
    return "symbol"


def compute_stats(passwords: List[str]) -> Dict[str, object]:
    """Return a stats dict: length histogram, char-class composition, top tokens."""
    lengths = [len(p) for p in passwords]
    total = len(passwords)

    len_counter: Counter[int] = Counter(lengths)
    char_class_counts: Counter[str] = Counter()
    for pw in passwords:
        for ch in pw:
            char_class_counts[_char_class(ch)] += 1

    token_counter: Counter[str] = Counter(passwords)

    return {
        "total": total,
        "length_counter": dict(sorted(len_counter.items())),
        "char_class_counts": dict(char_class_counts),
        "top_100_tokens": token_counter.most_common(100),
    }


def save_stats(stats: Dict[str, object], out_dir: Path) -> None:
    """Write stats CSV + length histogram PNG to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Length histogram CSV.
    len_csv = out_dir / "length_histogram.csv"
    with open(len_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["length", "count"])
        for length, count in sorted(stats["length_counter"].items()):
            writer.writerow([length, count])

    # Char-class composition CSV.
    cc_csv = out_dir / "char_class_composition.csv"
    with open(cc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count"])
        for cls, count in sorted(stats["char_class_counts"].items()):
            writer.writerow([cls, count])

    # Top-100 tokens (counts only, no passwords written to CSV header).
    tok_csv = out_dir / "top100_tokens.csv"
    with open(tok_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "count"])
        for rank, (_, count) in enumerate(stats["top_100_tokens"], start=1):
            writer.writerow([rank, count])

    # Length histogram PNG.
    lengths = sorted(stats["length_counter"].keys())
    counts = [stats["length_counter"][l] for l in lengths]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(lengths, counts, color="steelblue", width=0.8)
    axes[0].set_xlabel("Password length")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Length distribution  (N={stats['total']:,})")
    axes[0].set_yscale("log")

    cc = stats["char_class_counts"]
    axes[1].bar(cc.keys(), cc.values(), color=["#4c72b0", "#c44e52", "#55a868", "#8172b2"])
    axes[1].set_title("Character-class composition")
    axes[1].set_ylabel("Total characters")

    plt.tight_layout()
    fig.savefig(out_dir / "stats.png", dpi=120)
    plt.close(fig)
    logger.info("Stats saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _checksum(lines: List[str]) -> str:
    """SHA-256 of newline-joined lines (deterministic, sorted)."""
    blob = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def split_corpus(
    passwords: List[str],
    split_cfg: Dict[str, object],
    out_dir: Path,
    seed: int,
    user_field_map: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """Split *passwords* into train / val / test and write to *out_dir*.

    Parameters
    ----------
    passwords:
        Deduplicated, cleaned passwords.
    split_cfg:
        ``split`` section of protocol.yaml (train/val/test ratios, by_user).
    out_dir:
        Directory to write ``train.txt``, ``val.txt``, ``test.txt``, and
        ``test_checksum.txt``.
    seed:
        RNG seed (from protocol.yaml).
    user_field_map:
        Optional ``{password: user_id}`` mapping. Required when
        ``by_user=True`` in the split config.

    Returns
    -------
    dict with keys "train", "val", "test".
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_r = float(split_cfg["train"])
    val_r = float(split_cfg["val"])
    test_r = float(split_cfg["test"])
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Ratios must sum to 1."

    by_user: bool = bool(split_cfg.get("by_user", False))

    splits: Dict[str, List[str]]

    if by_user and user_field_map is not None:
        splits = _split_by_user(passwords, user_field_map, train_r, val_r, test_r, seed)
    else:
        if by_user:
            logger.warning(
                "by_user=True in config but no user_field_map supplied; "
                "falling back to row-level split."
            )
        splits = _split_by_row(passwords, train_r, val_r, test_r, seed)

    # Assert zero overlap across splits.
    _assert_disjoint(splits)

    # Write files.
    for split_name, split_pws in splits.items():
        path = out_dir / f"{split_name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(split_pws) + "\n")
        logger.info("  %s: %d passwords → %s", split_name, len(split_pws), path)

    # Freeze test split with a checksum.
    checksum = _checksum(splits["test"])
    cksum_path = out_dir / "test_checksum.txt"
    cksum_path.write_text(checksum + "\n", encoding="utf-8")
    logger.info("Test split frozen: %s (SHA-256 %s…)", cksum_path, checksum[:16])

    # Manifest JSON.
    manifest = {
        "seed": seed,
        "by_user": by_user,
        "sizes": {k: len(v) for k, v in splits.items()},
        "test_sha256": checksum,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return splits


def _split_by_row(
    passwords: List[str],
    train_r: float,
    val_r: float,
    test_r: float,
    seed: int,
) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    pws = passwords[:]
    rng.shuffle(pws)
    n = len(pws)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return {
        "train": pws[:n_train],
        "val": pws[n_train: n_train + n_val],
        "test": pws[n_train + n_val:],
    }


def _split_by_user(
    passwords: List[str],
    user_map: Dict[str, str],
    train_r: float,
    val_r: float,
    test_r: float,
    seed: int,
) -> Dict[str, List[str]]:
    """Split so no user appears in more than one partition."""
    # Group passwords by user.
    user_to_pws: Dict[str, List[str]] = {}
    no_user: List[str] = []
    for pw in passwords:
        uid = user_map.get(pw)
        if uid:
            user_to_pws.setdefault(uid, []).append(pw)
        else:
            no_user.append(pw)

    users = list(user_to_pws.keys())
    rng = random.Random(seed)
    rng.shuffle(users)

    n = len(users)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    def _collect(ulist: List[str]) -> List[str]:
        pws: List[str] = []
        for u in ulist:
            pws.extend(user_to_pws[u])
        return pws

    train = _collect(users[:n_train])
    val = _collect(users[n_train: n_train + n_val])
    test = _collect(users[n_train + n_val:])

    # Passwords without a user ID go into train only (safe).
    train.extend(no_user)

    return {"train": train, "val": val, "test": test}


def _assert_disjoint(splits: Dict[str, List[str]]) -> None:
    keys = list(splits.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_set = set(splits[keys[i]])
            b_set = set(splits[keys[j]])
            overlap = a_set & b_set
            assert not overlap, (
                f"Split overlap detected between '{keys[i]}' and '{keys[j]}': "
                f"{len(overlap)} passwords in common "
                f"(first: {next(iter(overlap))!r})"
            )


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def clean_corpus(
    input_path: str | Path,
    out_dir: str | Path,
    protocol_path: Optional[str | Path] = None,
    filter_enabled: bool = False,
    min_len: int = 0,
    max_len: int = 0,
    user_field_map: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Full Phase 2 pipeline: clean → stats → split.

    Parameters
    ----------
    input_path:
        Raw corpus file (read-only).
    out_dir:
        Output directory (created if absent).
    protocol_path:
        Path to protocol.yaml; defaults to configs/protocol.yaml.
    filter_enabled:
        Apply optional length / charset filter (off by default).
    min_len / max_len:
        Bounds for the optional filter.
    user_field_map:
        ``{password: user_id}``; enables user-stratified splitting.

    Returns
    -------
    dict with keys: ``passwords``, ``stats``, ``splits``, ``out_dir``.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol() if protocol_path is None else _load_yaml(protocol_path)
    split_cfg = protocol["split"]
    seed: int = int(protocol.get("seed", 1337))
    set_seed(seed)

    logger.info("Cleaning corpus: %s", input_path)
    passwords = list(iter_clean(
        input_path,
        min_len=min_len,
        max_len=max_len,
        filter_enabled=filter_enabled,
    ))
    logger.info("Cleaned passwords: %d", len(passwords))

    stats_dir = out_dir / "stats"
    stats = compute_stats(passwords)
    save_stats(stats, stats_dir)

    splits = split_corpus(passwords, split_cfg, out_dir, seed, user_field_map)

    logger.info("Phase 2 complete. Outputs in %s", out_dir)
    return {
        "passwords": passwords,
        "stats": stats,
        "splits": splits,
        "out_dir": str(out_dir),
    }


def _load_yaml(path: str | Path) -> Dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_test_checksum(out_dir: str | Path) -> bool:
    """Confirm the frozen test split matches its stored SHA-256 checksum."""
    out_dir = Path(out_dir)
    test_path = out_dir / "test.txt"
    cksum_path = out_dir / "test_checksum.txt"
    with open(test_path, encoding="utf-8") as f:
        passwords = [line.rstrip("\n") for line in f if line.strip()]
    stored = cksum_path.read_text(encoding="utf-8").strip()
    computed = _checksum(passwords)
    match = computed == stored
    if not match:
        logger.error(
            "Test split checksum MISMATCH: stored=%s computed=%s", stored[:16], computed[:16]
        )
    return match
