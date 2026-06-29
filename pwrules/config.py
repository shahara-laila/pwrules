"""Shared configuration loading and global seeding.

All paths, hyperparameters, and the frozen evaluation protocol live in YAML under
``configs/``. Every module reads config through here so behaviour is reproducible
and there is a single place to set the RNG seed.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Repo root = two levels up from this file (pwrules/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

# Default seed; overridden by whatever a loaded config specifies. Modules should
# prefer the seed returned in their loaded config, but this is a safe fallback.
SEED: int = 1337


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a YAML config file into a dict.

    Accepts an absolute path, a path relative to the current working directory,
    or a bare name resolved against ``configs/`` (e.g. ``"protocol.yaml"``).
    """
    p = Path(path)
    candidates = [p]
    if not p.is_absolute():
        candidates += [Path.cwd() / p, CONFIGS_DIR / p, CONFIGS_DIR / p.name]
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Config {candidate} did not parse to a mapping.")
            return data
    raise FileNotFoundError(
        f"Config not found. Tried: {', '.join(str(c) for c in candidates)}"
    )


def set_seed(seed: Optional[int] = None) -> int:
    """Set global RNG seeds for reproducibility and return the seed used.

    Seeds Python's ``random`` and ``PYTHONHASHSEED`` always; seeds numpy and
    torch if they are importable (they may not be in a CPU-only env).
    """
    global SEED
    if seed is None:
        seed = SEED
    SEED = int(seed)

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)

    try:  # numpy is a core dep but guard anyway
        import numpy as np

        np.random.seed(SEED)
    except Exception:  # pragma: no cover - numpy missing
        pass

    try:  # torch only present in the train extra
        import torch

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    except Exception:  # pragma: no cover - torch missing on CPU-only env
        pass

    return SEED


def load_protocol() -> Dict[str, Any]:
    """Convenience loader for the FROZEN evaluation protocol."""
    return load_config(CONFIGS_DIR / "protocol.yaml")


def load_train_config() -> Dict[str, Any]:
    """Convenience loader for training/generation hyperparameters."""
    return load_config(CONFIGS_DIR / "train.yaml")
