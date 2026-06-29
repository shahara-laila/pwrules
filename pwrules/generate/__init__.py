"""Phase 6 — Rule generation / inference from the fine-tuned adapter.

Two generation regimes
----------------------
untargeted
    Prompt the model with varied base words; collect and deduplicate the
    generated rule strings.  Output: ``rules/llm_untargeted.rule``.

targeted
    For each held-out user in ``target_users_test.jsonl``, prompt the model
    with that user's attributes + a sampled base word; collect rules.
    Output: ``rules/llm_targeted/<user_id>.rule``.

The rule budget (``generation.budget`` in ``configs/train.yaml``) is the
maximum number of unique rules to generate in total across all prompts.

Diversity stats are reported in ``generation_stats.json``.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwrules.config import load_train_config, set_seed

logger = logging.getLogger(__name__)

# Probe wordlist used when no external base wordlist is given.
_DEFAULT_PROBE_WORDS = [
    "password", "dragon", "hello", "sunshine", "monkey", "shadow",
    "master", "letmein", "love", "football", "princess", "michael",
    "superman", "batman", "iloveyou", "welcome", "charlie", "donald",
    "jessica", "thomas", "soccer", "flower", "starwars", "summer",
    "mustang", "cheese", "baseball", "hockey", "ranger", "hunter",
]

# Instruction template (must match Phase 3 / Phase 5 training template).
_INSTRUCTION_TEMPLATE = (
    "Given the base word '{base}', generate a Hashcat password mangling rule "
    "that transforms it into a realistic password candidate."
)

_TARGETED_TEMPLATE = (
    "User profile: {attrs}. "
    "Given the base word '{base}', generate a Hashcat password mangling rule "
    "that transforms it into a realistic password candidate."
)


# ---------------------------------------------------------------------------
# Model loading (GPU only; lazy)
# ---------------------------------------------------------------------------

def load_model_for_inference(adapter_dir: str | Path, config_path: Optional[str | Path] = None):
    """Load the base model + LoRA adapter in inference mode.

    Returns ``(model, tokenizer)``.  Raises RuntimeError when GPU absent.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth not installed. On Kaggle: !pip install unsloth"
        ) from exc

    import torch  # type: ignore
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for inference. Enable GPU in Kaggle settings."
        )

    cfg = load_train_config() if config_path is None else _load_yaml(config_path)
    adapter_dir = Path(adapter_dir)

    logger.info("Loading adapter from %s …", adapter_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=cfg.get("max_seq_length", 1024),
        dtype=None,
        load_in_4bit=cfg.get("qlora", {}).get("load_in_4bit", True),
    )
    FastLanguageModel.for_inference(model)
    logger.info("Model loaded in inference mode.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single-rule generation
# ---------------------------------------------------------------------------

def _generate_single(
    model,
    tokenizer,
    prompt: str,
    gen_cfg: Dict[str, Any],
) -> str:
    """Run one inference pass and return the generated text."""
    import torch  # type: ignore

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 64)),
            temperature=float(gen_cfg.get("temperature", 0.8)),
            top_p=float(gen_cfg.get("top_p", 0.95)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return generated


def _build_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Untargeted generation
# ---------------------------------------------------------------------------

def generate_untargeted(
    model,
    tokenizer,
    budget: int,
    gen_cfg: Dict[str, Any],
    probe_words: List[str],
    seed: int,
) -> List[str]:
    """Generate *budget* unique Hashcat rules (untargeted).

    Cycles through *probe_words* with a seeded shuffle, collecting unique rules
    until the budget is reached.  Returns a deduplicated, ordered list.
    """
    rng = random.Random(seed)
    words = probe_words[:]
    rng.shuffle(words)

    unique_rules: list[str] = []
    seen: Set[str] = set()
    idx = 0
    attempts = 0
    max_attempts = budget * 5  # avoid infinite loops on degenerate outputs

    while len(unique_rules) < budget and attempts < max_attempts:
        word = words[idx % len(words)]
        idx += 1
        attempts += 1

        messages = [{"role": "user", "content": _INSTRUCTION_TEMPLATE.format(base=word)}]
        prompt = _build_prompt(tokenizer, messages)
        rule = _generate_single(model, tokenizer, prompt, gen_cfg)

        if rule and rule not in seen:
            seen.add(rule)
            unique_rules.append(rule)

        if attempts % 100 == 0:
            logger.info(
                "Untargeted generation: %d/%d unique rules (attempts=%d)",
                len(unique_rules), budget, attempts,
            )

    logger.info("Untargeted: %d unique rules generated.", len(unique_rules))
    return unique_rules


# ---------------------------------------------------------------------------
# Targeted generation
# ---------------------------------------------------------------------------

def generate_targeted(
    model,
    tokenizer,
    test_users: List[Dict],
    budget_per_user: int,
    gen_cfg: Dict[str, Any],
    probe_words: List[str],
    seed: int,
) -> Dict[str, List[str]]:
    """Generate rules for each held-out user in *test_users*.

    Parameters
    ----------
    test_users:
        Records from ``target_users_test.jsonl``.
    budget_per_user:
        Maximum unique rules to generate for each user.

    Returns
    -------
    ``{user_id: [rule, …]}``
    """
    rng = random.Random(seed)
    results: Dict[str, List[str]] = {}

    for user in test_users:
        uid = user["user_id"]
        attrs = user.get("attrs", {})
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())

        words = probe_words[:]
        rng.shuffle(words)

        unique_rules: list[str] = []
        seen: Set[str] = set()
        idx = 0
        attempts = 0
        max_attempts = budget_per_user * 5

        while len(unique_rules) < budget_per_user and attempts < max_attempts:
            word = words[idx % len(words)]
            idx += 1
            attempts += 1

            messages = [{"role": "user", "content": _TARGETED_TEMPLATE.format(
                attrs=attr_str, base=word,
            )}]
            prompt = _build_prompt(tokenizer, messages)
            rule = _generate_single(model, tokenizer, prompt, gen_cfg)

            if rule and rule not in seen:
                seen.add(rule)
                unique_rules.append(rule)

        results[uid] = unique_rules
        logger.info("User %s: %d rules generated.", uid, len(unique_rules))

    return results


# ---------------------------------------------------------------------------
# .rule file writer
# ---------------------------------------------------------------------------

def write_rule_file(rules: List[str], path: Path) -> None:
    """Write a list of rule strings to a Hashcat-format .rule file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rule in rules:
            f.write(rule + "\n")
    logger.info("Written %d rules → %s", len(rules), path)


# ---------------------------------------------------------------------------
# Diversity stats
# ---------------------------------------------------------------------------

def compute_diversity_stats(
    untargeted_rules: List[str],
    targeted_rules: Dict[str, List[str]],
) -> Dict[str, object]:
    """Compute diversity metrics for the generated rule sets."""
    all_targeted = [r for rules in targeted_rules.values() for r in rules]
    all_rules = untargeted_rules + all_targeted

    len_counter: Counter[int] = Counter(len(r) for r in all_rules)
    rule_counter: Counter[str] = Counter(all_rules)

    return {
        "untargeted": {
            "n_unique": len(set(untargeted_rules)),
            "n_total": len(untargeted_rules),
        },
        "targeted": {
            "n_users": len(targeted_rules),
            "n_unique_per_user": {
                uid: len(set(rules)) for uid, rules in targeted_rules.items()
            },
            "n_total": len(all_targeted),
        },
        "combined": {
            "n_unique": len(set(all_rules)),
            "n_total": len(all_rules),
            "length_distribution": dict(sorted(len_counter.items())),
            "top_20_rules": rule_counter.most_common(20),
        },
    }


def save_diversity_stats(stats: Dict[str, object], out_dir: Path) -> None:
    """Save diversity stats JSON and a rule-length distribution plot."""
    (out_dir / "generation_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    len_dist = stats["combined"].get("length_distribution", {})
    if len_dist:
        lengths = sorted(len_dist.keys())
        counts = [len_dist[l] for l in lengths]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(lengths, counts, color="steelblue")
        ax.set_xlabel("Rule string length (chars)")
        ax.set_ylabel("Count")
        ax.set_title("Generated rule length distribution")
        plt.tight_layout()
        fig.savefig(out_dir / "rule_length_dist.png", dpi=120)
        plt.close(fig)

    logger.info("Diversity stats saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def generate_rules(
    adapter_dir: str | Path,
    out_dir: str | Path,
    target_users_path: Optional[str | Path] = None,
    probe_words_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    budget: Optional[int] = None,
) -> Dict[str, object]:
    """Full Phase 6 pipeline: load adapter → generate → write .rule files.

    Parameters
    ----------
    adapter_dir:
        Directory containing the saved LoRA adapter (Phase 5 output).
    out_dir:
        Output directory for .rule files and stats.
    target_users_path:
        Path to ``target_users_test.jsonl`` (Phase 4 output).
        If None, targeted generation is skipped.
    probe_words_path:
        Path to a plain-text wordlist to use as generation prompts.
        If None, the built-in probe word list is used.
    config_path:
        Override for ``configs/train.yaml``.
    budget:
        Override for ``generation.budget`` in the config.

    Returns
    -------
    dict with keys: ``untargeted_rules``, ``targeted_rules``, ``out_dir``.
    """
    cfg = load_train_config() if config_path is None else _load_yaml(config_path)
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    out_dir = Path(out_dir)
    rules_dir = out_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = cfg.get("generation", {})
    total_budget = budget if budget is not None else int(gen_cfg.get("budget", 10_000))

    # Load probe words.
    if probe_words_path is not None:
        probe_words = _read_wordlist(Path(probe_words_path))
    else:
        probe_words = _DEFAULT_PROBE_WORDS[:]

    if not probe_words:
        raise ValueError("Probe word list is empty. Provide --wordlist or check the file.")

    # Load model.
    model, tokenizer = load_model_for_inference(adapter_dir, config_path)

    # -----------------------------------------------------------------------
    # Untargeted generation.
    # -----------------------------------------------------------------------
    logger.info("Starting untargeted generation (budget=%d) …", total_budget)
    untargeted_rules = generate_untargeted(
        model, tokenizer, total_budget, gen_cfg, probe_words, seed
    )
    write_rule_file(untargeted_rules, rules_dir / "llm_untargeted.rule")

    # -----------------------------------------------------------------------
    # Targeted generation.
    # -----------------------------------------------------------------------
    targeted_rules: Dict[str, List[str]] = {}
    if target_users_path is not None:
        test_users = _read_jsonl(Path(target_users_path))
        if test_users:
            budget_per_user = max(1, total_budget // len(test_users))
            logger.info(
                "Targeted generation: %d users × %d rules …",
                len(test_users), budget_per_user,
            )
            targeted_rules = generate_targeted(
                model, tokenizer, test_users, budget_per_user, gen_cfg, probe_words, seed
            )
            targeted_dir = rules_dir / "llm_targeted"
            targeted_dir.mkdir(exist_ok=True)
            for uid, rules in targeted_rules.items():
                write_rule_file(rules, targeted_dir / f"{uid}.rule")
        else:
            logger.warning("target_users_test.jsonl is empty — skipping targeted generation.")

    # -----------------------------------------------------------------------
    # Diversity stats.
    # -----------------------------------------------------------------------
    stats = compute_diversity_stats(untargeted_rules, targeted_rules)
    save_diversity_stats(stats, out_dir)

    logger.info("Phase 6 complete. Outputs in %s", out_dir)
    return {
        "untargeted_rules": untargeted_rules,
        "targeted_rules": targeted_rules,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_wordlist(path: Path) -> List[str]:
    words = []
    with open(path, "rb") as f:
        for line in f:
            try:
                w = line.decode("utf-8").strip()
            except UnicodeDecodeError:
                w = line.decode("latin-1").strip()
            if w:
                words.append(w)
    return words


def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_yaml(path: str | Path) -> Dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
