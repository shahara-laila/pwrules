"""Tests for pwrules.env (Phase 1).

GPU-dependent assertions are skipped when CUDA is not available so these
tests can run on any CI or local machine.
"""

import importlib
import sys
import pytest

from pwrules.env import check_env


def test_check_env_returns_dict():
    status = check_env()
    assert isinstance(status, dict)


def test_python_ok_flag():
    status = check_env()
    assert status["python_ok"] is True, (
        f"Python >= 3.10 required; got {sys.version}"
    )


def test_on_kaggle_is_bool():
    status = check_env()
    assert isinstance(status["on_kaggle"], bool)


def test_cuda_available_is_bool():
    status = check_env()
    assert isinstance(status["cuda_available"], bool)


def test_hashcat_flag_is_bool():
    status = check_env()
    assert isinstance(status["hashcat_available"], bool)


def test_kaggle_inputs_is_list():
    status = check_env()
    assert isinstance(status["kaggle_inputs"], list)


def test_require_gpu_raises_when_no_cuda(monkeypatch):
    """require_gpu=True must raise if CUDA is absent (non-GPU CI)."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            pytest.skip("CUDA present — cannot test the 'no GPU' branch")
    except ImportError:
        pass  # torch not installed, CUDA definitely absent

    with pytest.raises(RuntimeError, match="GPU required"):
        check_env(require_gpu=True)


def test_require_hashcat_raises_when_absent(monkeypatch, tmp_path):
    """require_hashcat=True must raise if hashcat is not on PATH."""
    import shutil
    original = shutil.which

    def mock_which(name, *args, **kwargs):
        if name == "hashcat":
            return None
        return original(name, *args, **kwargs)

    monkeypatch.setattr("pwrules.env.shutil.which", mock_which)
    with pytest.raises(RuntimeError, match="Hashcat not found"):
        check_env(require_hashcat=True)


def test_missing_kaggle_dataset_warns():
    """Requesting a non-existent dataset slug should emit a UserWarning."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_env(kaggle_datasets=["nonexistent-dataset-xyz"])
    flagged = [x for x in w if issubclass(x.category, UserWarning)]
    # On Kaggle the dataset might be present; locally it won't be.
    from pathlib import Path
    on_kaggle = Path("/kaggle/input/nonexistent-dataset-xyz").exists()
    if not on_kaggle:
        assert flagged, "Expected a UserWarning for missing dataset"


def test_pkg_flags_present():
    status = check_env()
    expected = {"pkg_yaml", "pkg_pandas", "pkg_numpy", "pkg_matplotlib"}
    assert expected.issubset(status.keys())


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="torch not installed",
)
def test_vram_probe_skipped_without_gpu():
    """vram_probe_ok should be None when probe_model=True but no GPU."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            pytest.skip("CUDA present — probe would actually run")
    except ImportError:
        pass

    status = check_env(probe_model=True)
    # Either None (not attempted) or False (probe ran and failed)
    assert status["vram_probe_ok"] in (None, False)
