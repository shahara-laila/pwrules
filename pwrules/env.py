"""Phase 1 — Runtime environment verification.

Call check_env() from any Kaggle notebook to assert that GPU, Hashcat, and the
private input datasets are all reachable before running expensive pipeline steps.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def check_env(
    require_gpu: bool = False,
    require_hashcat: bool = False,
    kaggle_datasets: Optional[List[str]] = None,
    probe_model: bool = False,
    model_name: Optional[str] = None,
) -> Dict[str, object]:
    """Inspect and report the runtime environment.

    Parameters
    ----------
    require_gpu:
        Raise RuntimeError if CUDA is not available.
    require_hashcat:
        Raise RuntimeError if ``hashcat`` is not on PATH.
    kaggle_datasets:
        List of expected dataset directory names under /kaggle/input.
        Missing entries are flagged in the returned dict.
    probe_model:
        If True and a GPU is present, run a 1-step dummy QLoRA forward/backward
        to confirm the configured model fits in VRAM.
    model_name:
        Override the model to probe (otherwise reads configs/train.yaml).

    Returns
    -------
    dict
        Keys: python_ok, cuda_available, gpu_name, gpu_memory_mib,
        hashcat_available, hashcat_stdout_ok, on_kaggle, kaggle_inputs,
        pkg_* for key packages, vram_probe_ok (if probe_model=True).
    """
    status: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Python
    # ------------------------------------------------------------------
    status["python_version"] = sys.version
    status["python_ok"] = sys.version_info >= (3, 10)

    # ------------------------------------------------------------------
    # Key packages
    # ------------------------------------------------------------------
    for pkg in ["yaml", "pandas", "numpy", "matplotlib", "datasets", "torch",
                "transformers", "peft", "trl", "bitsandbytes"]:
        try:
            importlib.import_module(pkg if pkg != "yaml" else "yaml")
            status[f"pkg_{pkg}"] = True
        except ImportError:
            status[f"pkg_{pkg}"] = False

    # ------------------------------------------------------------------
    # GPU / CUDA
    # ------------------------------------------------------------------
    status["cuda_available"] = False
    status["gpu_name"] = None
    status["gpu_memory_mib"] = None

    if status.get("pkg_torch"):
        try:
            import torch  # type: ignore
            status["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                status["gpu_name"] = props.name
                status["gpu_memory_mib"] = props.total_memory // (1024 ** 2)
        except Exception as exc:
            status["cuda_error"] = str(exc)

    if require_gpu and not status["cuda_available"]:
        raise RuntimeError(
            "GPU required but CUDA is not available. Enable a GPU in Kaggle "
            "notebook settings."
        )

    # ------------------------------------------------------------------
    # Hashcat
    # ------------------------------------------------------------------
    status["hashcat_available"] = shutil.which("hashcat") is not None
    status["hashcat_stdout_ok"] = False

    if status["hashcat_available"]:
        # Quick --stdout smoke: apply a no-op rule to a probe word.
        try:
            result = subprocess.run(
                ["hashcat", "--stdout", "-r", "/dev/stdin", "--quiet"],
                input=": password",  # rule file content piped via stdin — works with --stdin
                capture_output=True,
                text=True,
                timeout=15,
            )
            # Fallback: write a tiny temp rule file and probe it.
            if result.returncode != 0:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".rule", delete=False
                ) as rf:
                    rf.write(":\n")
                    rf_path = rf.name
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False
                ) as wf:
                    wf.write("password\n")
                    wf_path = wf.name
                res2 = subprocess.run(
                    ["hashcat", "--stdout", "-r", rf_path, wf_path, "--quiet"],
                    capture_output=True, text=True, timeout=15,
                )
                status["hashcat_stdout_ok"] = (
                    res2.returncode == 0 and "password" in res2.stdout
                )
                os.unlink(rf_path)
                os.unlink(wf_path)
            else:
                status["hashcat_stdout_ok"] = True
        except Exception as exc:
            status["hashcat_error"] = str(exc)

        # Version / device info
        try:
            ver = subprocess.run(
                ["hashcat", "-I"],
                capture_output=True, text=True, timeout=15,
            )
            status["hashcat_info"] = (ver.stdout or ver.stderr).strip()[:200]
        except Exception:
            pass

    if require_hashcat and not status["hashcat_available"]:
        raise RuntimeError(
            "Hashcat not found. Install with: apt-get install -y hashcat"
        )

    # ------------------------------------------------------------------
    # Kaggle inputs
    # ------------------------------------------------------------------
    kaggle_input = Path("/kaggle/input")
    status["on_kaggle"] = (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or kaggle_input.exists()
    )
    status["kaggle_inputs"] = (
        [d.name for d in kaggle_input.iterdir() if d.is_dir()]
        if kaggle_input.exists()
        else []
    )

    missing = []
    if kaggle_datasets:
        for ds in kaggle_datasets:
            found = ds in status["kaggle_inputs"]
            status[f"dataset_{ds}"] = found
            if not found:
                missing.append(ds)
    if missing:
        import warnings
        warnings.warn(
            f"Expected Kaggle datasets not found in /kaggle/input: {missing}",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # VRAM probe (optional: 1-step dummy QLoRA forward/backward)
    # ------------------------------------------------------------------
    status["vram_probe_ok"] = None  # None = not attempted
    if probe_model and status["cuda_available"]:
        status["vram_probe_ok"] = _vram_probe(model_name)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    _print_summary(status)
    return status


def _vram_probe(model_name: Optional[str] = None) -> bool:
    """Run a 1-step dummy forward pass to confirm the model fits in VRAM."""
    try:
        from pwrules.config import load_train_config
        cfg = load_train_config()
        name = model_name or cfg.get("model_name", "Qwen/Qwen3-0.6B")
        max_seq = cfg.get("max_seq_length", 256)

        from unsloth import FastLanguageModel  # type: ignore
        import torch  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=name,
            max_seq_length=max_seq,
            dtype=None,
            load_in_4bit=True,
        )
        # 1-step dummy forward
        dummy = tokenizer(
            "password",
            return_tensors="pt",
            truncation=True,
            max_length=8,
        ).to("cuda")
        with torch.no_grad():
            _ = model(**dummy, labels=dummy["input_ids"])
        del model
        torch.cuda.empty_cache()
        return True
    except Exception as exc:
        print(f"[vram_probe] FAILED: {exc}")
        return False


def _print_summary(status: Dict[str, object]) -> None:
    print("=" * 55)
    print("  pwrules environment check")
    print("=" * 55)
    py = status["python_version"].split()[0]
    print(f"  Python : {py}  ok={status['python_ok']}")
    print(
        f"  CUDA   : {status['cuda_available']}  "
        f"GPU={status.get('gpu_name')}  "
        f"VRAM={status.get('gpu_memory_mib')} MiB"
    )
    print(f"  Hashcat: {status['hashcat_available']}  "
          f"stdout_ok={status['hashcat_stdout_ok']}")
    print(f"  Kaggle : {status['on_kaggle']}  "
          f"inputs={status['kaggle_inputs']}")
    if status.get("vram_probe_ok") is not None:
        print(f"  VRAMprobe: {status['vram_probe_ok']}")
    missing_pkgs = [k[4:] for k, v in status.items()
                    if k.startswith("pkg_") and not v]
    if missing_pkgs:
        print(f"  Missing packages: {missing_pkgs}")
    print("=" * 55)
