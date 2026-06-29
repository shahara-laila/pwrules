"""Phase 4 — Target-conditioning dataset construction.

Builds instruction examples where the input contains structured user attributes
alongside the base word, enabling the model to generate personalised rules.

Two modes
---------
``synthetic`` (default)
    A seeded attribute generator assigns ``{name, birth_year, interest}`` to
    each password in ``rules_dataset.jsonl``. Passwords are grouped into
    synthetic "users" (``n_users`` configurable), and the grouping is
    deterministic (seeded).  The metadata field ``synthetic: true`` is recorded
    in all output files so downstream code can flag that real personal data was
    not used.

``real``
    Expects a CSV with columns ``password,user_id,name,birth_year,interest``
    (or a subset; missing columns are left empty).  Real data is never committed
    to this repo — it must live in a private Kaggle Dataset.

Outputs (all in *out_dir*)
--------------------------
targeted_dataset.jsonl        All training instruction examples (attributes + base → rule).
target_users_test.jsonl       Held-out users: {user_id, attrs, passwords}. Zero overlap.
split_manifest.json           Sizes, seed, mode, disjoint assertion result.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from pwrules.config import load_protocol, set_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic attribute pools
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "alice", "bob", "charlie", "diana", "edward", "fiona", "george", "helen",
    "ivan", "julia", "kevin", "laura", "mike", "nancy", "oliver", "patricia",
    "quinn", "rachel", "samuel", "tina", "ulrich", "victoria", "william",
    "xena", "yusuf", "zoe", "adam", "bella", "carlos", "dana", "eric",
    "faith", "gary", "hana", "ian", "jade", "kyle", "lisa", "mark", "nina",
    "oscar", "penny", "ray", "sara", "tom", "uma", "victor", "wendy",
    "xavier", "yasmin", "zachary",
]

_INTERESTS = [
    "football", "basketball", "soccer", "tennis", "music", "movies", "gaming",
    "cooking", "travel", "photography", "reading", "fitness", "yoga",
    "cycling", "swimming", "hiking", "chess", "painting", "dancing",
    "coding", "anime", "cars", "fashion", "finance", "science",
]


# ---------------------------------------------------------------------------
# Synthetic user generator
# ---------------------------------------------------------------------------

def generate_synthetic_users(
    n_users: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Generate a list of synthetic user attribute dicts.

    Each user has: ``{user_id, name, birth_year, interest}``.
    """
    rng = random.Random(seed)
    users = []
    for i in range(n_users):
        users.append({
            "user_id": f"synthetic_user_{i:05d}",
            "name": rng.choice(_FIRST_NAMES),
            "birth_year": rng.randint(1950, 2005),
            "interest": rng.choice(_INTERESTS),
        })
    return users


# ---------------------------------------------------------------------------
# Attribute-assignment (synthetic mode)
# ---------------------------------------------------------------------------

def assign_synthetic_attributes(
    triples: List[Dict],
    n_users: int,
    seed: int,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Assign each triple to a synthetic user.

    Returns
    -------
    augmented_triples:
        Each triple extended with ``{user_id, name, birth_year, interest}``.
    user_map:
        ``{user_id: {name, birth_year, interest}}``.
    """
    rng = random.Random(seed)
    users = generate_synthetic_users(n_users, seed)
    user_map = {u["user_id"]: u for u in users}

    augmented = []
    for t in triples:
        user = rng.choice(users)
        augmented.append({
            **t,
            "user_id": user["user_id"],
            "name": user["name"],
            "birth_year": user["birth_year"],
            "interest": user["interest"],
            "synthetic": True,
        })
    return augmented, user_map


# ---------------------------------------------------------------------------
# Real data loader
# ---------------------------------------------------------------------------

def load_real_attributes(csv_path: str | Path) -> List[Dict]:
    """Load a real attribute-linked dataset from a CSV file.

    Expected columns: ``password``, ``user_id``, ``name``, ``birth_year``,
    ``interest``.  Missing columns are tolerated and left empty.

    IMPORTANT: This file must live in a private Kaggle Dataset and must never
    be committed to the code repo (see .gitignore).
    """
    path = Path(csv_path)
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "password":   row.get("password", ""),
                "user_id":    row.get("user_id", ""),
                "name":       row.get("name", ""),
                # Normalise birth_year to int (matching synthetic mode) when numeric.
                "birth_year": (int(row["birth_year"])
                               if str(row.get("birth_year", "")).strip().isdigit()
                               else row.get("birth_year", "")),
                "interest":   row.get("interest", ""),
                "synthetic":  False,
            })
    logger.info("Loaded %d real attribute records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Train / held-out user split
# ---------------------------------------------------------------------------

def split_users(
    augmented_triples: List[Dict],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Split triples by user_id so held-out users don't appear in training.

    Returns ``(train_triples, test_user_records)``.
    """
    user_to_triples: Dict[str, List[Dict]] = {}
    for t in augmented_triples:
        uid = t["user_id"]
        user_to_triples.setdefault(uid, []).append(t)

    user_ids = list(user_to_triples.keys())
    rng = random.Random(seed)
    rng.shuffle(user_ids)

    n_test = max(1, int(len(user_ids) * test_ratio))
    test_user_ids: FrozenSet[str] = frozenset(user_ids[:n_test])
    train_user_ids: FrozenSet[str] = frozenset(user_ids[n_test:])

    # Assert disjoint.
    overlap = test_user_ids & train_user_ids
    assert not overlap, f"User split overlap: {overlap}"

    train_triples = [
        t for uid in train_user_ids for t in user_to_triples[uid]
    ]
    # Held-out: aggregate passwords per user for the targeted eval.
    test_records = [
        {
            "user_id": uid,
            "attrs": {
                k: user_to_triples[uid][0][k]
                for k in ("name", "birth_year", "interest")
                if k in user_to_triples[uid][0]
            },
            "passwords": [t["password"] for t in user_to_triples[uid]],
            "synthetic": user_to_triples[uid][0].get("synthetic", True),
        }
        for uid in test_user_ids
    ]

    logger.info(
        "User split: %d train users | %d held-out users",
        len(train_user_ids), len(test_user_ids),
    )
    return train_triples, test_records


# ---------------------------------------------------------------------------
# Instruction formatting
# ---------------------------------------------------------------------------

def _format_targeted_instruction(triple: Dict) -> Dict[str, str]:
    """Format a targeted instruction: attributes + base → rule."""
    from pwrules.ruleextract import _INSTRUCTION_TEMPLATE
    attrs = {
        k: triple[k]
        for k in ("name", "birth_year", "interest")
        if k in triple
    }
    attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
    inp = (
        f"User profile: {attr_str}. "
        + _INSTRUCTION_TEMPLATE.format(base=triple["base"])
    )
    return {"input": inp, "output": triple["rule"]}


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def build_targeted_dataset(
    rules_jsonl: str | Path,
    out_dir: str | Path,
    mode: str = "synthetic",
    real_csv_path: Optional[str | Path] = None,
    n_synthetic_users: int = 500,
    test_user_ratio: float = 0.1,
    protocol_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Phase 4 pipeline: add user attributes and split by user.

    Parameters
    ----------
    rules_jsonl:
        Path to ``rules_dataset.jsonl`` produced in Phase 3.
    out_dir:
        Directory to write outputs.
    mode:
        ``"synthetic"`` or ``"real"``.
    real_csv_path:
        Path to the real attribute CSV (required when ``mode="real"``).
    n_synthetic_users:
        Number of synthetic user personas (only used in synthetic mode).
    test_user_ratio:
        Fraction of users held out for targeted evaluation.
    protocol_path:
        Override for ``configs/protocol.yaml``.

    Returns
    -------
    dict with keys: ``train_triples``, ``test_users``, ``out_dir``.
    """
    rules_jsonl = Path(rules_jsonl)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol() if protocol_path is None else _load_yaml(protocol_path)
    seed = int(protocol.get("seed", 1337))
    set_seed(seed)

    # Load phase-3 triples.
    triples = _read_jsonl(rules_jsonl)
    logger.info("Loaded %d triples from %s", len(triples), rules_jsonl)

    # Augment with attributes.
    if mode == "synthetic":
        augmented, user_map = assign_synthetic_attributes(triples, n_synthetic_users, seed)
        logger.info("Synthetic mode: %d users assigned.", n_synthetic_users)
    elif mode == "real":
        if real_csv_path is None:
            raise ValueError("real_csv_path is required when mode='real'.")
        real_records = load_real_attributes(real_csv_path)
        # Join triples with real records on password.
        pw_to_attrs = {r["password"]: r for r in real_records}
        augmented = []
        for t in triples:
            attrs = pw_to_attrs.get(t["password"], {})
            augmented.append({
                **t,
                # An empty CSV user_id must NOT collapse many real people into one
                # "" user (which would break the per-user split); treat it as anon.
                "user_id": attrs.get("user_id") or f"anon_{len(augmented)}",
                "name":       attrs.get("name", ""),
                "birth_year": attrs.get("birth_year", ""),
                "interest":   attrs.get("interest", ""),
                "synthetic":  False,
            })
        user_map = {}
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'synthetic' or 'real'.")

    # Split by user.
    train_triples, test_users = split_users(augmented, test_user_ratio, seed)

    # Verify disjoint (paranoia check).
    test_user_ids = {u["user_id"] for u in test_users}
    train_user_ids = {t["user_id"] for t in train_triples}
    assert not test_user_ids & train_user_ids, (
        "Held-out users leaked into training set!"
    )

    # Write targeted_dataset.jsonl (training examples).
    targeted_instructions = [_format_targeted_instruction(t) for t in train_triples]
    targeted_path = out_dir / "targeted_dataset.jsonl"
    _write_jsonl(targeted_instructions, targeted_path)

    # Write target_users_test.jsonl.
    test_path = out_dir / "target_users_test.jsonl"
    _write_jsonl(test_users, test_path)

    # Manifest.
    manifest = {
        "mode": mode,
        "synthetic": mode == "synthetic",
        "seed": seed,
        "n_users_total": len(set(t["user_id"] for t in augmented)),
        "n_train_users": len(train_user_ids),
        "n_test_users": len(test_user_ids),
        "n_train_triples": len(train_triples),
        "n_test_users_passwords": sum(len(u["passwords"]) for u in test_users),
        "disjoint": True,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info("Phase 4 complete. Outputs in %s", out_dir)

    return {
        "train_triples": train_triples,
        "test_users": test_users,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: List, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Written %d records to %s", len(records), path)


def _load_yaml(path: str | Path) -> Dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
