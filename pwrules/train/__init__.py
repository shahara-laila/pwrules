"""Phase 5 — QLoRA fine-tuning with Unsloth + TRL SFTTrainer.

Entry point: ``train(data_dir, output_dir, config_path)``

All GPU/Unsloth imports are lazy so the module can be imported on CPU-only
machines for testing without raising ImportError.

Outputs (all in *output_dir*)
------------------------------
adapter/                        Saved LoRA adapter + tokeniser.
checkpoints/                    SFT checkpoint dir (resumable).
training_curves.png / .csv      Train + val loss per logging step.
memorisation_report.json        Novel-rule fraction (generalisation check).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwrules.config import load_train_config, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chat-template helpers (model-agnostic)
# ---------------------------------------------------------------------------

def apply_chat(tokenizer, messages, add_generation_prompt: bool) -> str:
    """Render *messages* with the model's chat template.

    Passes ``enable_thinking=False`` when the template supports it (Qwen3 and
    similar reasoning models emit a <think> block by default, which would fill
    the short generation budget with prose instead of a rule). Falls back
    cleanly for templates that don't accept the kwarg, so it stays model-agnostic.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _chat_part_markers(tokenizer):
    """Derive (instruction_part, response_part) header strings from the template.

    Model-agnostic: renders a probe turn and extracts the boilerplate that
    precedes the user content and the assistant content respectively, so
    response-only loss masking works without hard-coding any model's tokens.
    Returns ``(None, None)`` if the markers cannot be determined.
    """
    probe = "PWR_PROBE"
    try:
        user_only = apply_chat(tokenizer, [{"role": "user", "content": probe}], False)
        with_gen = apply_chat(tokenizer, [{"role": "user", "content": probe}], True)
        instruction_part = user_only.split(probe)[0]
        response_part = with_gen[len(user_only):]
        if instruction_part and response_part:
            return instruction_part, response_part
    except Exception:  # pragma: no cover - template variability
        pass
    return None, None


# ---------------------------------------------------------------------------
# Model loading (GPU only)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: Dict[str, Any]):
    """Load the base model in 4-bit QLoRA configuration via Unsloth.

    Raises RuntimeError with a clear message if Unsloth / GPU is absent.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth is not installed. "
            "On Kaggle, run:  !pip install unsloth  (GPU required)."
        ) from exc

    import torch  # type: ignore
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for fine-tuning. "
            "Enable GPU in Kaggle notebook settings."
        )

    # Unsloth / recent bitsandbytes + Triton kernels are compiled for compute
    # capability >= 7.0 (Volta+). Pascal cards (e.g. Kaggle's P100, CC 6.0) have
    # no matching kernel binary and fail mid-training with the cryptic
    # "CUDA error: no kernel image is available for execution on the device".
    # Fail fast here with an actionable message instead.
    major, minor = torch.cuda.get_device_capability()
    if major < 7:
        gpu_name = torch.cuda.get_device_name(0)
        raise RuntimeError(
            f"GPU '{gpu_name}' has compute capability {major}.{minor}, but the "
            "training stack (Unsloth/bitsandbytes) requires >= 7.0. "
            "On Kaggle, switch the Accelerator to 'GPU T4 x2' (or T4); the "
            "P100 (compute 6.0) is not supported."
        )

    model_name: str = cfg["model_name"]
    max_seq_len: int = cfg.get("max_seq_length", 1024)
    qlora: Dict = cfg.get("qlora", {})

    logger.info("Loading %s (4-bit) …", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,          # auto (bfloat16 when supported)
        load_in_4bit=qlora.get("load_in_4bit", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=qlora.get("r", 16),
        target_modules=qlora.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_alpha=qlora.get("lora_alpha", 32),
        lora_dropout=qlora.get("lora_dropout", 0.05),
        bias=qlora.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("seed", 1337),
        use_rslora=False,
        loftq_config=None,
    )
    logger.info("QLoRA adapters attached (r=%d α=%d).", qlora.get("r", 16), qlora.get("lora_alpha", 32))
    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_instruction_dataset(
    data_dir: Path,
    tokenizer,
    max_seq_len: int,
    use_targeted: bool = False,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    seed: int = 1337,
):
    """Load instruction JSONL files and format with the model's chat template.

    Tries ``targeted_dataset.jsonl`` first when *use_targeted* is True
    (Phase 4 output), falling back to ``instructions_train.jsonl`` (Phase 3).
    """
    from datasets import Dataset, DatasetDict  # type: ignore

    def _load(path: Path) -> List[Dict[str, str]]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # Resolve train file.
    if use_targeted and (data_dir / "targeted_dataset.jsonl").exists():
        train_file = data_dir / "targeted_dataset.jsonl"
        logger.info("Using targeted instruction dataset.")
    elif (data_dir / "instructions_train.jsonl").exists():
        train_file = data_dir / "instructions_train.jsonl"
    else:
        raise FileNotFoundError(
            f"No instruction training file found in {data_dir}. "
            "Run Phase 3 (and optionally Phase 4) first."
        )

    val_file = data_dir / "instructions_val.jsonl"
    if not val_file.exists():
        # The targeted dataset dir (Phase 4) has no val split; fall back to the
        # Phase-3 instruction val file discovered elsewhere under the inputs.
        try:
            from pwrules import paths
            discovered = paths.val_txt  # noqa: F401  (ensure module import works)
            alt = paths.find_file("instructions_val.jsonl", required=False)
        except Exception:
            alt = None
        if alt is not None and Path(alt) != val_file:
            logger.info("Validation file not in %s; using discovered %s", data_dir, alt)
            val_file = Path(alt)
        else:
            raise FileNotFoundError(
                f"Validation file not found: {val_file}. Attach the Phase-3 "
                "rules dataset (contains instructions_val.jsonl)."
            )

    train_records = _load(train_file)
    val_records = _load(val_file)

    # Deterministic, seeded subsample so very large (rockyou-scale) instruction
    # sets fit free-tier RAM / the 12h session. Shuffle first so the subset is
    # representative, not just the head of the file.
    import random as _random

    def _cap(records: List[Dict[str, str]], limit: Optional[int], label: str):
        if limit is not None and len(records) > limit:
            rng = _random.Random(seed)
            rng.shuffle(records)
            logger.info("Subsampling %s: %d -> %d (seed=%d).", label, len(records), limit, seed)
            return records[:limit]
        return records

    train_records = _cap(train_records, max_train_samples, "train")
    val_records = _cap(val_records, max_eval_samples, "val")
    logger.info("Train: %d  Val: %d instruction examples.", len(train_records), len(val_records))

    def _fmt(record: Dict[str, str]) -> Dict[str, str]:
        messages = [
            {"role": "user",      "content": record["input"]},
            {"role": "assistant", "content": record["output"]},
        ]
        return {"text": apply_chat(tokenizer, messages, add_generation_prompt=False)}

    train_ds = Dataset.from_list([_fmt(r) for r in train_records])
    val_ds   = Dataset.from_list([_fmt(r) for r in val_records])
    return DatasetDict({"train": train_ds, "validation": val_ds})


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def save_training_curves(log_history: List[Dict], out_dir: Path) -> None:
    """Save train/val loss CSV and PNG from the trainer's log history."""
    train_steps, train_losses = [], []
    val_steps,   val_losses   = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", 0))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            val_steps.append(entry.get("step", 0))
            val_losses.append(entry["eval_loss"])

    # CSV.
    csv_path = out_dir / "training_curves.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])
        all_steps = sorted(set(train_steps + val_steps))
        t_dict = dict(zip(train_steps, train_losses))
        v_dict = dict(zip(val_steps, val_losses))
        for s in all_steps:
            writer.writerow([s, t_dict.get(s, ""), v_dict.get(s, "")])

    # PNG.
    fig, ax = plt.subplots(figsize=(10, 5))
    if train_steps:
        ax.plot(train_steps, train_losses, label="train loss", color="#4c72b0")
    if val_steps:
        ax.plot(val_steps, val_losses, label="val loss", color="#c44e52", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves")
    if train_steps or val_steps:
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)
    logger.info("Training curves saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Memorisation check
# ---------------------------------------------------------------------------

def check_memorisation(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    train_rules: Set[str],
    out_dir: Path,
    n_samples: int = 500,
) -> Dict[str, object]:
    """Sample rule generations and report the novel-rule fraction.

    A rule is 'novel' if it does not appear verbatim in the training set.
    High novel-rule rate → the model generalises; low rate → memorisation.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore
        FastLanguageModel.for_inference(model)
    except Exception:
        pass  # already in inference mode or unsloth unavailable

    gen_cfg = cfg.get("generation", {})
    max_new = int(gen_cfg.get("max_new_tokens", 64))
    temperature = float(gen_cfg.get("temperature", 0.8))
    top_p = float(gen_cfg.get("top_p", 0.95))

    # Use a diverse probe set (the generation module's default list) rather than
    # 10 repeated words, so the novelty estimate is representative.
    from pwrules.generate import _DEFAULT_PROBE_WORDS, _sanitize_rule
    base_words = _DEFAULT_PROBE_WORDS
    probe_words = [base_words[i % len(base_words)] for i in range(n_samples)]

    generated_rules: List[str] = []
    for word in probe_words:
        messages = [{"role": "user", "content": (
            f"Given the base word '{word}', generate a Hashcat password mangling rule "
            "that transforms it into a realistic password candidate."
        )}]
        prompt = apply_chat(tokenizer, messages, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        import torch  # type: ignore
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        # Sanitize to a single rule line so malformed/multi-line output isn't
        # miscounted as a "novel" rule (which would inflate the generalisation score).
        generated_rules.append(_sanitize_rule(generated))

    novel = [r for r in generated_rules if r not in train_rules]
    novel_fraction = len(novel) / len(generated_rules) if generated_rules else 0.0

    report = {
        "n_sampled": len(generated_rules),
        "n_novel": len(novel),
        "n_in_train": len(generated_rules) - len(novel),
        "novel_fraction": round(novel_fraction, 4),
        "memorisation_fraction": round(1.0 - novel_fraction, 4),
        "note": (
            "High novel_fraction (>0.5) indicates generalisation. "
            "Low novel_fraction indicates memorisation."
        ),
    }

    report_path = out_dir / "memorisation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Memorisation check: novel=%.1f%%  memorised=%.1f%%",
        novel_fraction * 100, (1 - novel_fraction) * 100,
    )
    return report


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------

def train(
    data_dir: str | Path,
    output_dir: str | Path,
    config_path: Optional[str | Path] = None,
    use_targeted: bool = False,
    resume_from_checkpoint: Optional[str | Path] = None,
) -> Path:
    """Full Phase 5 training pipeline.

    Parameters
    ----------
    data_dir:
        Directory containing ``instructions_train.jsonl``,
        ``instructions_val.jsonl`` (Phase 3 output), and optionally
        ``targeted_dataset.jsonl`` (Phase 4 output).
    output_dir:
        Where to save the adapter, checkpoints, and reports.
    config_path:
        Override for ``configs/train.yaml``.
    use_targeted:
        If True, prefer the targeted instruction dataset (Phase 4).

    Returns
    -------
    Path to the saved LoRA adapter directory.
    """
    cfg = load_train_config() if config_path is None else _load_yaml(config_path)
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load model.
    # -----------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(cfg)

    # -----------------------------------------------------------------------
    # Load dataset.
    # -----------------------------------------------------------------------
    max_seq_len: int = cfg.get("max_seq_length", 1024)
    _train_cfg = cfg.get("training", {})
    datasets = load_instruction_dataset(
        data_dir, tokenizer, max_seq_len, use_targeted,
        max_train_samples=_train_cfg.get("max_train_samples"),
        max_eval_samples=_train_cfg.get("max_eval_samples"),
        seed=seed,
    )

    # -----------------------------------------------------------------------
    # Configure SFTTrainer.
    # -----------------------------------------------------------------------
    try:
        from trl import SFTTrainer, SFTConfig  # type: ignore
        from unsloth import is_bfloat16_supported  # type: ignore
        from transformers import EarlyStoppingCallback  # type: ignore
    except ImportError as exc:
        raise RuntimeError("TRL / Unsloth not installed.") from exc

    train_cfg = cfg.get("training", {})
    early_cfg = train_cfg.get("early_stopping", {})
    callbacks = []
    if early_cfg.get("enabled", True):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=int(early_cfg.get("patience", 3))
        ))

    ckpt_dir = output_dir / "checkpoints"
    # CLI --resume (passed through) overrides the train.yaml setting.
    resume = resume_from_checkpoint or train_cfg.get("resume_from_checkpoint")

    training_args = SFTConfig(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 16)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        num_train_epochs=int(train_cfg.get("epochs", 3)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(train_cfg.get("eval_steps", 50)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 50)),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        seed=seed,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=False,
        # Limit tokenizer map() worker processes: the default (8) OOM-kills a
        # worker on free-tier RAM with large instruction sets.
        dataset_num_proc=int(train_cfg.get("dataset_num_proc", 2)),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        args=training_args,
        callbacks=callbacks,
    )

    # Response-only loss masking: compute the SFT loss on the assistant's rule
    # ONLY, not the instruction boilerplate. This focuses model capacity on
    # generating rules and is the single biggest quality lever for this task.
    # Markers are derived from the tokenizer so it remains model-agnostic; if
    # they can't be determined we fall back to full-sequence loss with a warning.
    if cfg.get("training", {}).get("response_only_loss", True):
        try:
            from unsloth.chat_templates import train_on_responses_only  # type: ignore
            instr_part, resp_part = _chat_part_markers(tokenizer)
            if resp_part:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part=instr_part,
                    response_part=resp_part,
                )
                logger.info("Response-only loss masking enabled (response_part=%r).", resp_part)
            else:
                logger.warning("Could not derive chat markers; using full-sequence loss.")
        except Exception as exc:  # pragma: no cover - optional / version-dependent
            logger.warning("train_on_responses_only unavailable (%s); full-sequence loss.", exc)

    # -----------------------------------------------------------------------
    # Train.
    # -----------------------------------------------------------------------
    logger.info("Starting training …")
    train_result = trainer.train(resume_from_checkpoint=resume)
    logger.info("Training complete. %s", train_result.metrics)

    # -----------------------------------------------------------------------
    # Save adapter.
    # -----------------------------------------------------------------------
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Adapter saved → %s", adapter_dir)

    # -----------------------------------------------------------------------
    # Training curves.
    # -----------------------------------------------------------------------
    save_training_curves(trainer.state.log_history, output_dir)

    # -----------------------------------------------------------------------
    # Memorisation check.
    # -----------------------------------------------------------------------
    train_rules: Set[str] = set()
    train_jsonl = (
        data_dir / "targeted_dataset.jsonl"
        if use_targeted and (data_dir / "targeted_dataset.jsonl").exists()
        else data_dir / "instructions_train.jsonl"
    )
    if train_jsonl.exists():
        with open(train_jsonl, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                train_rules.add(rec.get("output", ""))

    check_memorisation(model, tokenizer, cfg, train_rules, output_dir)

    return adapter_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> Dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
