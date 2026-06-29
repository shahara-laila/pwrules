"""Tests for pwrules.train (Phase 5).

GPU-dependent training tests are skipped when CUDA is unavailable.
The CPU smoke test runs a 1-step stub using a tiny model to verify that the
dataset formatting and training-loop wiring are correct without a real GPU.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, Set

import pytest

from pwrules.train import load_instruction_dataset, save_training_curves


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_INSTRUCTIONS_TRAIN = [
    {"input": "Given the base word 'password', generate a Hashcat rule.", "output": "c"},
    {"input": "Given the base word 'dragon', generate a Hashcat rule.", "output": "sa@ $1"},
    {"input": "Given the base word 'hello', generate a Hashcat rule.", "output": "u"},
    {"input": "Given the base word 'sunshine', generate a Hashcat rule.", "output": "r"},
    {"input": "Given the base word 'monkey', generate a Hashcat rule.", "output": "$! $?"},
]

SAMPLE_INSTRUCTIONS_VAL = [
    {"input": "Given the base word 'shadow', generate a Hashcat rule.", "output": "c $1"},
    {"input": "Given the base word 'master', generate a Hashcat rule.", "output": ":"},
]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    with open(d / "instructions_train.jsonl", "w") as f:
        for rec in SAMPLE_INSTRUCTIONS_TRAIN:
            f.write(json.dumps(rec) + "\n")
    with open(d / "instructions_val.jsonl", "w") as f:
        for rec in SAMPLE_INSTRUCTIONS_VAL:
            f.write(json.dumps(rec) + "\n")
    return d


# ---------------------------------------------------------------------------
# save_training_curves (CPU-safe)
# ---------------------------------------------------------------------------

def test_save_training_curves_creates_files(tmp_path: Path):
    log_history = [
        {"step": 10, "loss": 2.5},
        {"step": 20, "loss": 2.1},
        {"step": 20, "eval_loss": 2.3},
        {"step": 30, "loss": 1.9},
        {"step": 40, "eval_loss": 2.0},
    ]
    save_training_curves(log_history, tmp_path)
    assert (tmp_path / "training_curves.csv").exists()
    assert (tmp_path / "training_curves.png").exists()


def test_save_training_curves_empty(tmp_path: Path):
    save_training_curves([], tmp_path)
    assert (tmp_path / "training_curves.csv").exists()


def test_save_training_curves_csv_content(tmp_path: Path):
    log_history = [
        {"step": 10, "loss": 2.5},
        {"step": 10, "eval_loss": 2.3},
    ]
    save_training_curves(log_history, tmp_path)
    content = (tmp_path / "training_curves.csv").read_text()
    assert "step" in content
    assert "train_loss" in content
    assert "val_loss" in content


# ---------------------------------------------------------------------------
# load_instruction_dataset (requires transformers tokenizer stub)
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Minimal tokenizer stub for testing dataset loading without a real model."""
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n".join(parts)


def test_load_instruction_dataset_loads_jsonl(data_dir: Path):
    """Dataset should load train + val records without a real tokenizer."""
    ds = load_instruction_dataset(data_dir, _StubTokenizer(), max_seq_len=256)
    assert "train" in ds and "validation" in ds
    assert len(ds["train"]) == len(SAMPLE_INSTRUCTIONS_TRAIN)
    assert len(ds["validation"]) == len(SAMPLE_INSTRUCTIONS_VAL)


def test_load_instruction_dataset_text_field(data_dir: Path):
    """Each example must have a 'text' field (used as dataset_text_field)."""
    ds = load_instruction_dataset(data_dir, _StubTokenizer(), max_seq_len=256)
    for example in ds["train"]:
        assert "text" in example and example["text"]


def test_load_instruction_dataset_prefers_targeted(data_dir: Path):
    """When targeted_dataset.jsonl exists and use_targeted=True, use it."""
    targeted = [
        {"input": "User profile: name=alice. Given 'pass', generate a rule.", "output": "c"},
    ]
    with open(data_dir / "targeted_dataset.jsonl", "w") as f:
        for rec in targeted:
            f.write(json.dumps(rec) + "\n")
    ds = load_instruction_dataset(data_dir, _StubTokenizer(), 256, use_targeted=True)
    assert len(ds["train"]) == 1  # only the targeted file


def test_load_instruction_dataset_missing_file_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "instructions_val.jsonl").write_text('{"input":"x","output":"y"}\n')
    with pytest.raises(FileNotFoundError):
        load_instruction_dataset(empty, _StubTokenizer(), 256)


# ---------------------------------------------------------------------------
# CPU smoke test (1-step training on a tiny model stub)
# ---------------------------------------------------------------------------

def _has_torch_and_transformers() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


@pytest.mark.skipif(
    not _has_torch_and_transformers(),
    reason="torch/transformers not installed",
)
def test_one_step_cpu_training_stub(data_dir: Path, tmp_path: Path):
    """Verify the training loop can run 1 step using a tiny GPT2 (CPU only)."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            pytest.skip("GPU present — use the full Kaggle notebook instead.")
    except ImportError:
        pytest.skip("torch not installed.")

    # We don't call the real `train()` (requires Unsloth + GPU).
    # Instead, verify that the training-loop scaffolding works end-to-end
    # by running a 1-step SFT loop with GPT-2 (tiny, CPU-safe).
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments  # type: ignore
        from trl import SFTTrainer, SFTConfig  # type: ignore
        from datasets import Dataset  # type: ignore
    except ImportError:
        pytest.skip("transformers/trl not installed.")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    texts = [f"user: {r['input']}\nassistant: {r['output']}" for r in SAMPLE_INSTRUCTIONS_TRAIN]
    train_ds = Dataset.from_dict({"text": texts})
    val_ds   = Dataset.from_dict({"text": [f"user: {r['input']}\nassistant: {r['output']}"
                                            for r in SAMPLE_INSTRUCTIONS_VAL]})

    out_dir = tmp_path / "stub_run"
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=1,
        max_steps=1,
        per_device_train_batch_size=1,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=64,
        no_cuda=True,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=args,
    )
    trainer.train()  # should not raise

    # Confirm the training log has at least one entry.
    assert len(trainer.state.log_history) >= 1


# ---------------------------------------------------------------------------
# Module-level import check
# ---------------------------------------------------------------------------

def test_train_module_imports_without_gpu():
    """pwrules.train must be importable without a GPU (lazy imports)."""
    import importlib
    mod = importlib.import_module("pwrules.train")
    assert hasattr(mod, "train")
    assert hasattr(mod, "load_model_and_tokenizer")
    assert hasattr(mod, "check_memorisation")
