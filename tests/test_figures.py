"""Tests for the paper-figure module (matplotlib, no GPU/hashcat needed)."""

from __future__ import annotations

import json

import pytest

from pwrules.eval import figures


@pytest.fixture()
def rule_file(tmp_path):
    p = tmp_path / "llm.rule"
    p.write_text("c $1\nsa@ $2\n: \nu\nr d\n", encoding="utf-8")
    return p


def test_pipeline_diagram_always_renders(tmp_path):
    out = figures.pipeline_diagram(tmp_path / "pipe.png")
    assert out.exists() and out.stat().st_size > 0


def test_rule_op_distribution(tmp_path, rule_file):
    out = figures.rule_op_distribution(rule_file, tmp_path / "ops.png")
    assert out is not None and out.exists()


def test_rule_op_distribution_missing_returns_none(tmp_path):
    assert figures.rule_op_distribution(tmp_path / "nope.rule", tmp_path / "o.png") is None


def test_memorisation_breakdown(tmp_path):
    rep = tmp_path / "memo.json"
    rep.write_text(json.dumps({"n_novel": 80, "n_in_train": 20}), encoding="utf-8")
    out = figures.memorisation_breakdown(rep, tmp_path / "memo.png")
    assert out is not None and out.exists()


def test_top_rules_and_targeted(tmp_path):
    gs = tmp_path / "generation_stats.json"
    gs.write_text(json.dumps({
        "untargeted": {"n_unique": 120, "n_total": 150},
        "targeted": {"n_total": 60, "n_unique_per_user": {"u1": 5, "u2": 7, "u3": 3}},
        "combined": {"top_20_rules": [["c $1", 10], ["u", 8], ["r", 5]]},
    }), encoding="utf-8")
    assert figures.top_rules(gs, tmp_path / "top.png").exists()
    assert figures.targeted_vs_untargeted(gs, tmp_path / "tvu.png").exists()
    assert figures.per_user_rule_counts(gs, tmp_path / "pu.png").exists()


def test_hit_at_k_bars_and_complementarity(tmp_path):
    rc = tmp_path / "results.csv"
    rc.write_text(
        "method,dataset,k,hit_rate,seed\n"
        "best64,rockyou,1000,0.10,1\n"
        "LLM-filtered,rockyou,1000,0.12,1\n"
        "LLM+best64,rockyou,1000,0.18,1\n"
        "best64,rockyou,100,0.05,1\n"
        "LLM-filtered,rockyou,100,0.06,1\n"
        "LLM+best64,rockyou,100,0.09,1\n",
        encoding="utf-8",
    )
    assert figures.hit_at_k_bars(rc, tmp_path / "bars.png", k=1000).exists()
    assert figures.complementarity(rc, tmp_path / "comp.png").exists()


def test_ablation_bars(tmp_path):
    ab = tmp_path / "ablations.csv"
    ab.write_text(
        "label,mean_a,std_a,mean_b,std_b,delta\n"
        "Filtering on vs off,0.12,0.01,0.10,0.02,0.02\n",
        encoding="utf-8",
    )
    assert figures.ablation_bars(ab, tmp_path / "abl.png").exists()


def test_generate_all_figures_partial_inputs(tmp_path):
    # Only the pipeline diagram has no input requirement; others skip cleanly.
    produced = figures.generate_all_figures(out_dir=tmp_path / "figs")
    assert any("fig_pipeline.png" in p for p in produced)
