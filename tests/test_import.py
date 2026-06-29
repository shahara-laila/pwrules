"""Phase 0 smoke tests: the package imports and config loading works."""

import pwrules
from pwrules import config


def test_package_imports():
    assert pwrules.__version__


def test_submodules_import():
    import importlib

    for mod in [
        "clean",
        "ruleextract",
        "conditioning",
        "train",
        "generate",
        "filter",
        "eval",
    ]:
        importlib.import_module(f"pwrules.{mod}")


def test_set_seed_returns_seed():
    assert config.set_seed(123) == 123
    assert config.SEED == 123


def test_load_protocol_has_frozen_keys():
    proto = config.load_protocol()
    for key in ["base_wordlist_path", "guess_budget", "split", "hit_at_k_definition", "seed"]:
        assert key in proto, f"frozen protocol key missing: {key}"
    assert isinstance(proto["guess_budget"], list) and len(proto["guess_budget"]) >= 2
    for k in ["train", "val", "test", "by_user"]:
        assert k in proto["split"]


def test_load_train_config_has_model_and_qlora():
    cfg = config.load_train_config()
    assert "model_name" in cfg
    assert "qlora" in cfg and "r" in cfg["qlora"]
    assert "training" in cfg and "generation" in cfg
