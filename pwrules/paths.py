"""Centralised, slug-agnostic path resolution for Kaggle (and local) runs.

Why this exists
---------------
Kaggle mounts every attached dataset under ``/kaggle/input/<slug>/…`` where
``<slug>`` is account-specific (e.g.
``/kaggle/input/datasets/alice/rockyou/rockyou.txt``). Hard-coding those paths
breaks the moment a different account runs the notebooks. This module finds
artifacts by *filename* instead, so notebooks and CLIs never name a slug.

Resolution order for every lookup
---------------------------------
1. An explicit environment override ``PWRULES_<KEY>`` — set it once, at the top
   of a notebook, to pin a path (this is the single "set from the top" point).
2. The first match found by walking the search roots: ``/kaggle/working`` first
   (so same-session outputs win) then ``/kaggle/input``.

Relocate the roots for local testing with ``PWRULES_INPUT`` / ``PWRULES_WORKING``
(``PWRULES_INPUT`` accepts several ``os.pathsep``-separated roots).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

# --- single override point -------------------------------------------------
_DEFAULT_INPUT = "/kaggle/input"
_DEFAULT_WORKING = "/kaggle/working"

# Path segments never worth matching (repo internals, caches, venvs).
_EXCLUDE = (
    f"{os.sep}.git{os.sep}",
    f"{os.sep}__pycache__{os.sep}",
    f"{os.sep}site-packages{os.sep}",
    f"{os.sep}.ipynb_checkpoints{os.sep}",
    f"{os.sep}pwrules{os.sep}tests{os.sep}",
)


def input_roots() -> List[Path]:
    """Read-only dataset roots (``PWRULES_INPUT``, default ``/kaggle/input``)."""
    raw = os.environ.get("PWRULES_INPUT", _DEFAULT_INPUT)
    return [Path(p) for p in raw.split(os.pathsep) if p]


def working_root() -> Path:
    """Writable output root (``PWRULES_WORKING``, default ``/kaggle/working``)."""
    return Path(os.environ.get("PWRULES_WORKING", _DEFAULT_WORKING))


def search_roots() -> List[Path]:
    """Working root first (same-session outputs win), then the input roots."""
    roots = [working_root(), *input_roots()]
    seen: set = set()
    ordered: List[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


def _ok(p: Path) -> bool:
    s = str(p)
    return not any(seg in s for seg in _EXCLUDE)


def _scan(name: str, want_dir: bool) -> Optional[Path]:
    for root in search_roots():
        if not root.exists():
            continue
        for m in sorted(root.rglob(name)):
            if not _ok(m):
                continue
            if want_dir and m.is_dir():
                return m
            if not want_dir and m.is_file():
                return m
    return None


def _msg(name: str, env: Optional[str], kind: str) -> str:
    roots = ", ".join(str(r) for r in search_roots())
    hint = f", or set {env}=<path>" if env else ""
    return (
        f"Could not locate {kind} '{name}' under: {roots}. "
        f"Attach the dataset that contains it{hint}."
    )


# --- generic lookups -------------------------------------------------------

def find_file(name: str, *, env: Optional[str] = None,
              required: bool = True) -> Optional[Path]:
    """First file named *name* under the search roots (or the ``env`` override)."""
    if env and os.environ.get(env):
        p = Path(os.environ[env]).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"{env}={p} is not an existing file")
    hit = _scan(name, want_dir=False)
    if hit is None and required:
        raise FileNotFoundError(_msg(name, env, "file"))
    return hit


def find_dir(name: str, *, env: Optional[str] = None,
             required: bool = True) -> Optional[Path]:
    """First directory named *name* under the search roots (or ``env`` override)."""
    if env and os.environ.get(env):
        p = Path(os.environ[env]).expanduser()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"{env}={p} is not an existing directory")
    hit = _scan(name, want_dir=True)
    if hit is None and required:
        raise FileNotFoundError(_msg(name, env, "directory"))
    return hit


def dir_with(marker: str, *, env: Optional[str] = None,
             required: bool = True) -> Optional[Path]:
    """Directory *containing* a file named *marker* (returns the parent dir)."""
    if env and os.environ.get(env):
        p = Path(os.environ[env]).expanduser()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"{env}={p} is not an existing directory")
    f = find_file(marker, required=False)
    if f is not None:
        return f.parent
    if required:
        raise FileNotFoundError(_msg(marker, env, "directory containing"))
    return None


def out(name: str = "") -> Path:
    """A writable output path under the working root (created on demand)."""
    p = working_root() / name if name else working_root()
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- named artifacts (markers match the documented phase outputs) ----------
# Each resolver: env override -> filename/marker search. ``required=False``
# returns None instead of raising when the artifact is not present yet.

def corpus(required: bool = True) -> Optional[Path]:
    """Raw training corpus (Phase 2 input), e.g. rockyou.txt."""
    return find_file("rockyou.txt", env="PWRULES_ROCKYOU", required=required)


def clean_dir(required: bool = True) -> Optional[Path]:
    """Phase 2 output dir (holds train/val/test + test_checksum.txt)."""
    return dir_with("test_checksum.txt", env="PWRULES_CLEAN", required=required)


def train_txt(required: bool = True) -> Optional[Path]:
    return find_file("train.txt", env="PWRULES_TRAIN_TXT", required=required)


def val_txt(required: bool = True) -> Optional[Path]:
    return find_file("val.txt", env="PWRULES_VAL_TXT", required=required)


def test_txt(required: bool = True) -> Optional[Path]:
    return find_file("test.txt", env="PWRULES_TEST_TXT", required=required)


def rules_dir(required: bool = True) -> Optional[Path]:
    """Phase 3 output dir (holds instructions_train/val.jsonl); train --data."""
    return dir_with("instructions_train.jsonl", env="PWRULES_RULES", required=required)


def rules_dataset(required: bool = True) -> Optional[Path]:
    """Phase 3 rules_dataset.jsonl (conditioning --rules input)."""
    return find_file("rules_dataset.jsonl", env="PWRULES_RULES_DATASET", required=required)


def targeted_dir(required: bool = True) -> Optional[Path]:
    """Phase 4 output dir (holds targeted_dataset.jsonl)."""
    return dir_with("targeted_dataset.jsonl", env="PWRULES_TARGETED", required=required)


def target_users(required: bool = True) -> Optional[Path]:
    """Phase 4 held-out users (target_users_test.jsonl)."""
    return find_file("target_users_test.jsonl", env="PWRULES_TARGET_USERS", required=required)


def adapter_dir(required: bool = True) -> Optional[Path]:
    """Phase 5 LoRA adapter dir (holds adapter_config.json)."""
    return dir_with("adapter_config.json", env="PWRULES_ADAPTER", required=required)


def checkpoints_dir(required: bool = False) -> Optional[Path]:
    """Phase 5 resumable checkpoint dir (optional)."""
    return find_dir("checkpoints", env="PWRULES_CHECKPOINTS", required=required)


def generated_untargeted(required: bool = True) -> Optional[Path]:
    """Phase 6 untargeted rule file (llm_untargeted.rule)."""
    return find_file("llm_untargeted.rule", env="PWRULES_LLM_UNTARGETED", required=required)


def generated_dir(required: bool = True) -> Optional[Path]:
    """Phase 6 dir holding the generated .rule files."""
    return dir_with("llm_untargeted.rule", env="PWRULES_GENERATED", required=required)


def targeted_rules_dir(required: bool = False) -> Optional[Path]:
    """Phase 6 per-user targeted rule dir (the ``llm_targeted`` folder)."""
    return find_dir("llm_targeted", env="PWRULES_TARGETED_RULES", required=required)


def filtered_dir(required: bool = True) -> Optional[Path]:
    """Phase 7 output dir (holds filter_funnel.csv)."""
    return dir_with("filter_funnel.csv", env="PWRULES_FILTERED", required=required)


def filtered_untargeted(required: bool = True) -> Optional[Path]:
    """Phase 7 filtered untargeted rule file (llm_untargeted_filtered.rule)."""
    return find_file("llm_untargeted_filtered.rule", env="PWRULES_LLM_FILTERED", required=required)


def results_dir(required: bool = True) -> Optional[Path]:
    """Phase 8 output dir (holds results.csv)."""
    return dir_with("results.csv", env="PWRULES_RESULTS", required=required)


def ablations_dir(required: bool = True) -> Optional[Path]:
    """Phase 9 output dir (holds ablations.csv)."""
    return dir_with("ablations.csv", env="PWRULES_ABLATIONS", required=required)


_NAMED: Dict[str, Callable[..., Optional[Path]]] = {
    "corpus": corpus,
    "clean_dir": clean_dir,
    "train_txt": train_txt,
    "val_txt": val_txt,
    "test_txt": test_txt,
    "rules_dir": rules_dir,
    "rules_dataset": rules_dataset,
    "targeted_dir": targeted_dir,
    "target_users": target_users,
    "adapter_dir": adapter_dir,
    "generated_dir": generated_dir,
    "filtered_dir": filtered_dir,
    "results_dir": results_dir,
    "ablations_dir": ablations_dir,
}


def summary() -> Dict[str, str]:
    """Map of every known artifact that currently resolves (for printing)."""
    found: Dict[str, str] = {}
    for key, fn in _NAMED.items():
        try:
            v = fn(required=False)
        except Exception:
            v = None
        if v is not None:
            found[key] = str(v)
    return found


def show() -> Dict[str, str]:
    """Print resolved artifacts (call once at the top of a notebook)."""
    roots = [str(r) for r in search_roots() if r.exists()]
    print("search roots:", roots)
    found = summary()
    if not found:
        print("  (no known artifacts found yet — attach datasets or run earlier phases)")
    for key, val in found.items():
        print(f"  {key:18s} -> {val}")
    return found
