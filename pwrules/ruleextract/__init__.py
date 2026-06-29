"""Phase 3 — Rule extraction pipeline.

Converts a cleaned password corpus into validated (base_word, rule, password)
triples, computes coverage and rule-frequency statistics, and emits
instruction-format datasets for fine-tuning.

Outputs (all written to *out_dir*)
-----------------------------------
rules_dataset.jsonl         Full validated triples {base, rule, password}.
coverage_report.json        Coverage %, no-base count, invalid count.
rule_frequency.csv          Rule string → count, sorted descending.
rule_frequency.png          Bar chart (top-50 rules).
instructions_train.jsonl    Instruction pairs from the train split.
instructions_val.jsonl      Instruction pairs from the val split.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwrules.config import load_protocol, set_seed
from pwrules.ruleextract.extractor import (
    DEFAULT_LEET_MAP,
    infer_rule,
    select_base,
)

logger = logging.getLogger(__name__)

# Instruction template — keeps a clean separation between input and output.
_INSTRUCTION_TEMPLATE = (
    "Given the base word '{base}', generate a Hashcat password mangling rule "
    "that transforms it into a realistic password candidate."
)


# ---------------------------------------------------------------------------
# Wordlist loading
# ---------------------------------------------------------------------------

def load_wordlist(path: str | Path) -> FrozenSet[str]:
    """Load a wordlist file into a frozenset of lowercase strings.

    Only lines that consist entirely of alphabetic characters are kept; this
    ensures the wordlist contains clean base words (no digits/symbols) that
    can serve as valid rule targets.
    """
    words: set[str] = set()
    with open(path, "rb") as f:
        for line in f:
            try:
                word = line.decode("utf-8").strip().lower()
            except UnicodeDecodeError:
                word = line.decode("latin-1").strip().lower()
            if word and word.isalpha():
                words.add(word)
    logger.info("Wordlist loaded: %d unique alpha words from %s", len(words), path)
    return frozenset(words)


def _build_wordlist_from_passwords(passwords: List[str]) -> FrozenSet[str]:
    """Build a simple reference wordlist from the alphabetic tokens of a corpus.

    Used as a fallback when no external base wordlist is provided.
    """
    words: set[str] = set()
    for pw in passwords:
        alpha = pw.lower()
        if alpha.isalpha():
            words.add(alpha)
    return frozenset(words)


# ---------------------------------------------------------------------------
# Triple extraction
# ---------------------------------------------------------------------------

def extract_triples(
    passwords: List[str],
    wordlist: FrozenSet[str],
    leet_map: Dict[str, str] = DEFAULT_LEET_MAP,
) -> Tuple[List[Dict], int, int]:
    """Convert cleaned passwords into validated (base, rule, password) triples.

    Returns
    -------
    triples:
        List of dicts ``{base, rule, password}``.
    no_base_count:
        Passwords for which no wordlist base was found.
    invalid_count:
        Triples where the inferred rule failed round-trip validation.
    """
    triples: List[Dict] = []
    no_base = 0
    invalid = 0

    for pw in passwords:
        base = select_base(pw, wordlist, leet_map)
        if base is None:
            no_base += 1
            continue

        rule = infer_rule(base, pw, leet_map)
        if rule is None:
            invalid += 1
            continue

        triples.append({"base": base, "rule": rule, "password": pw})

    return triples, no_base, invalid


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def save_rule_stats(triples: List[Dict], out_dir: Path) -> None:
    """Write rule_frequency.csv and rule_frequency.png to *out_dir*."""
    counter: Counter[str] = Counter(t["rule"] for t in triples)

    freq_csv = out_dir / "rule_frequency.csv"
    with open(freq_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "count"])
        for rule, count in counter.most_common():
            writer.writerow([rule, count])

    # Bar chart — top 50 rules.
    top = counter.most_common(50)
    if top:
        labels = [r[:30] for r, _ in top]  # truncate for readability
        counts = [c for _, c in top]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.barh(range(len(labels)), counts[::-1], color="steelblue")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1], fontsize=7)
        ax.set_xlabel("Count")
        ax.set_title(f"Top {len(top)} rule frequency (total triples: {len(triples):,})")
        plt.tight_layout()
        fig.savefig(out_dir / "rule_frequency.png", dpi=120)
        plt.close(fig)

    logger.info("Rule stats saved: %d unique rules", len(counter))


# ---------------------------------------------------------------------------
# Instruction-format dataset
# ---------------------------------------------------------------------------

def _format_instruction(
    triple: Dict,
    targeted: bool = False,
    user_attrs: Optional[Dict] = None,
) -> Dict[str, str]:
    """Format a triple as an instruction-following example.

    For untargeted: input = instruction with base word; output = rule.
    For targeted: input = instruction + user attributes; output = rule.
    """
    base = triple["base"]
    rule = triple["rule"]

    if targeted and user_attrs:
        attr_str = ", ".join(f"{k}={v}" for k, v in user_attrs.items())
        inp = (
            f"User profile: {attr_str}. "
            + _INSTRUCTION_TEMPLATE.format(base=base)
        )
    else:
        inp = _INSTRUCTION_TEMPLATE.format(base=base)

    return {"input": inp, "output": rule}


def write_instruction_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Written %d instruction records to %s", len(records), path)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def extract_rules(
    clean_dir: str | Path,
    out_dir: str | Path,
    base_wordlist_path: Optional[str | Path] = None,
    protocol_path: Optional[str | Path] = None,
    leet_map: Dict[str, str] = DEFAULT_LEET_MAP,
) -> Dict[str, object]:
    """Full Phase 3 pipeline.

    Parameters
    ----------
    clean_dir:
        Output of Phase 2 (contains ``train.txt``, ``val.txt``, ``test.txt``).
    out_dir:
        Directory where rule extraction outputs are written.
    base_wordlist_path:
        External reference wordlist (e.g. the attack base wordlist from
        protocol.yaml). If ``None``, a wordlist is built from the train split.
    protocol_path:
        Path to ``protocol.yaml``. Defaults to ``configs/protocol.yaml``.

    Returns
    -------
    dict with keys: ``triples``, ``coverage_report``, ``out_dir``.
    """
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol() if protocol_path is None else _load_yaml(protocol_path)
    seed = int(protocol.get("seed", 1337))
    set_seed(seed)

    # Load train + val split passwords.
    def _read(p: Path) -> List[str]:
        return [l.rstrip("\n") for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    train_pws = _read(clean_dir / "train.txt")
    val_pws = _read(clean_dir / "val.txt")

    # Build or load wordlist.
    if base_wordlist_path is not None:
        wordlist = load_wordlist(base_wordlist_path)
    else:
        wordlist = _build_wordlist_from_passwords(train_pws)
        logger.warning(
            "No base_wordlist_path provided; using %d alpha words from train split "
            "as reference. For best attack coverage, supply the real attack wordlist.",
            len(wordlist),
        )

    # Extract triples from train split.
    logger.info("Extracting rules from %d train passwords ...", len(train_pws))
    train_triples, no_base_train, invalid_train = extract_triples(train_pws, wordlist, leet_map)
    logger.info(
        "Train: %d triples | %d no-base | %d invalid",
        len(train_triples), no_base_train, invalid_train,
    )

    # Extract triples from val split.
    logger.info("Extracting rules from %d val passwords ...", len(val_pws))
    val_triples, no_base_val, invalid_val = extract_triples(val_pws, wordlist, leet_map)

    all_triples = train_triples + val_triples

    # Write rules_dataset.jsonl.
    jsonl_path = out_dir / "rules_dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for t in all_triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info("rules_dataset.jsonl: %d triples → %s", len(all_triples), jsonl_path)

    # Coverage report.
    total_train = len(train_pws)
    total_val = len(val_pws)
    coverage_train = len(train_triples) / total_train if total_train else 0.0
    coverage_val = len(val_triples) / total_val if total_val else 0.0
    coverage_report = {
        "train": {
            "total": total_train,
            "triples": len(train_triples),
            "coverage_pct": round(coverage_train * 100, 2),
            "no_base": no_base_train,
            "invalid": invalid_train,
        },
        "val": {
            "total": total_val,
            "triples": len(val_triples),
            "coverage_pct": round(coverage_val * 100, 2),
            "no_base": no_base_val,
            "invalid": invalid_val,
        },
    }
    (out_dir / "coverage_report.json").write_text(
        json.dumps(coverage_report, indent=2), encoding="utf-8"
    )
    logger.info(
        "Coverage: train=%.1f%%  val=%.1f%%",
        coverage_train * 100,
        coverage_val * 100,
    )

    # Rule frequency stats.
    save_rule_stats(all_triples, out_dir)

    # Instruction-format datasets.
    train_instructions = [_format_instruction(t) for t in train_triples]
    val_instructions = [_format_instruction(t) for t in val_triples]
    write_instruction_jsonl(train_instructions, out_dir / "instructions_train.jsonl")
    write_instruction_jsonl(val_instructions, out_dir / "instructions_val.jsonl")

    logger.info("Phase 3 complete. Outputs in %s", out_dir)
    return {
        "triples": all_triples,
        "coverage_report": coverage_report,
        "out_dir": str(out_dir),
    }


def _load_yaml(path: str | Path) -> Dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
