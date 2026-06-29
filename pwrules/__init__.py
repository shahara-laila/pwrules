"""pwrules — LLM-generated, target-conditioned Hashcat password mangling rules.

Code-only research package. See PLAYBOOK.md for the phase-by-phase workflow and
CLAUDE.md for project rules. Pipeline modules:

    clean         normalise + split a raw corpus (Phase 2)
    ruleextract   derive validated (base, rule, password) triples (Phase 3)
    conditioning  build target-conditioned instruction data (Phase 4)
    train         QLoRA fine-tuning (Phase 5)
    generate      rule generation / inference (Phase 6)
    filter        3-stage rule filtering funnel (Phase 7)
    eval          Hit@k evaluation + baselines (Phases 8-10)
"""

__version__ = "0.1.0"

from pwrules.config import SEED, load_config, set_seed  # noqa: F401

__all__ = ["__version__", "load_config", "set_seed", "SEED"]
