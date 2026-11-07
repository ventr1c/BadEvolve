"""Dry-run smoke test: import every release module, check the paper's settings, and build the
poisoned training pool (40 frozen clean train tasks + B=20 poisoned tasks from the first 20 of them)
without calling any model.

    python -m pytest -q tests/test_smoke.py

Network access fails these tests. Cases needing the third-party checkouts
(scripts/setup_third_party.sh) or the gated OfficeQA corpus (scripts/build_officeqa_splits.py) skip
themselves when those are absent.
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EM = REPO / "evolution_methods"
FRAMEWORKS = ("Trace2Skill", "SkillGrad", "SkillOpt", "A-EVOLVE-V3")
HAVE_UPSTREAM = all((EM / name / ".git").exists() for name in FRAMEWORKS)
SB_DATA = EM / "Trace2Skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400/dataset.json"
HAVE_OFFICEQA = (REPO / "dataset/officeqa/rq1_train/items.json").exists()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("this test must not open a network connection")
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_every_module_imports():
    """Both drivers and every attack/ module import cleanly (the upstream-dependent ones when present)."""
    import importlib
    import pkgutil
    import attack
    # Modules that drive a third-party framework import its package; without the checkouts they cannot load.
    upstream_packages = {"agent_evolve", "skillopt", "spreadsheetbench_support", "data", "pipeline",
                         "runners", "react_agent", "skill_evolver"}
    failed, skipped = [], []

    def note(name, exc):
        if isinstance(exc, ModuleNotFoundError) and not HAVE_UPSTREAM and (
                (exc.name or "").split(".")[0] in upstream_packages):
            skipped.append(name)
        else:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")

    names = [m.name for m in pkgutil.iter_modules(attack.__path__, "attack.")]
    while names:
        name = names.pop(0)
        try:
            module = importlib.import_module(name)
        except Exception as exc:                        # pragma: no cover - reported below
            note(name, exc)
            continue
        if hasattr(module, "__path__"):
            names += [m.name for m in pkgutil.iter_modules(module.__path__, name + ".")]
    assert not failed, "modules failed to import:\n" + "\n".join(failed)
    if HAVE_UPSTREAM:
        assert not skipped
    sys.path.insert(0, str(REPO / "scripts"))
    for name in ("run_wikitq_optimization",
                 *(("run_agent_optimization",) if HAVE_UPSTREAM else ())):
        importlib.import_module(name)   # the SB driver imports A-Evolve's package at module level


def test_arms_are_the_three_of_the_paper():
    from attack.protocol.arms import QA_GATE_MODES, SB_GATE_MODES, resolve_arm
    assert set(QA_GATE_MODES) == set(SB_GATE_MODES) == {"ours_v5", "textgrad", "tba"}
    for driver in ("qa", "sb"):
        ours = resolve_arm("ours_v5", driver)
        assert ours.advisory and ours.show_gates and ours.prompt_version == "v5"
        assert ours.fidelity_feedback and not ours.forces_no_rollback and not ours.gate_blind
        textgrad = resolve_arm("textgrad", driver)
        assert textgrad.textgrad and textgrad.gate_blind and textgrad.forces_no_rollback
        tba = resolve_arm("tba", driver)
        assert tba.tba and tba.forces_no_rollback
        with pytest.raises(ValueError):
            resolve_arm("ours_v4", driver)


def test_acceptance_gate():
    """BadEvolve's failure-aware gate: epsilon = N/K = 0.4 with N=2, K=5, plus the noise band."""
    from attack.protocol import measurement as mx
    assert mx.accept(0.8, 0.4, min_gain=0.4)         # a full 2/5 rise clears the gate
    assert not mx.accept(0.6, 0.4, min_gain=0.4)     # one evolve's worth of movement does not
    assert mx.accept(0.6, 0.4)                       # ... and would have been accepted without it
    assert not mx.accept(None, 0.0)
    assert mx.accept(0.2, None)
    accepted, branch = mx.band_verdict(0.2, 0.2, delta_succeeding=2, k=5, n=2,
                                       strict_new=0.0, strict_incumbent=0.0)
    assert accepted and branch
    rejected, _ = mx.band_verdict(0.0, 0.6, delta_succeeding=-3, k=5, n=2,
                                  strict_new=1.0, strict_incumbent=0.0)
    assert not rejected



def test_ablation_variants_are_the_papers_three():
    from attack.agent import study_ablation
    assert set(study_ablation.VARIANTS) >= {"full", "no_stage_feedback", "no_rollback", "no_failure_memory"}


def test_drivers_accept_the_paper_flags():
    import re
    paper = {"--gate-mode", "--k", "--final-k", "--rounds", "--n-base", "--clean-train-n", "--dev-n", "--test-n",
             "--optimizer-feedback-protocol", "--tie-break", "--accept-min-evolves", "--induction-measure",
             "--study-ablation", "--attack-type", "--surrogate-model", "--victim-model", "--model", "--seed",
             "--tag", "--resume"}
    for driver in ("run_wikitq_optimization.py", "run_agent_optimization.py"):
        flags = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', (REPO / "scripts" / driver).read_text()))
        assert paper <= flags, driver
    qa = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', (REPO / "scripts/run_wikitq_optimization.py").read_text()))
    assert {"--qa-measurement-protocol", "--qa-score-review", "--dataset", "--framework"} <= qa


def test_optimizer_never_runs_an_llm_judge():
    """The stage indicators are rule-based: no judge flags, no judge module."""
    for driver in ("run_wikitq_optimization.py", "run_agent_optimization.py"):
        text = (REPO / "scripts" / driver).read_text()
        assert "gate-judge" not in text and "gate_judge" not in text
    assert not (REPO / "attack/agent/gate_judge.py").exists()


# ── the poisoned training pool, built for real but with no model call ────────────────────────────────

def _pool(dataset, n_clean=40, n_base=20):
    """Build dataset's training pool exactly as the drivers do: the frozen clean prefix, the poisoned
    tasks stamped from the seed rule, then the receipts check that the victim wrote what we expect."""
    from attack.evaluation.qa_targets import make_qa_target
    from attack.pipeline import training_pool
    import types

    clean = training_pool.clean_prefix(dataset, n_clean)
    rule = make_qa_target(dataset).rule()
    base = clean[:n_base]
    payload = types.SimpleNamespace(items=[
        {"id": f"poison-{t['id']}", "base": t, "query": rule, "trigger": True} for t in base])
    return clean, rule, payload, training_pool.augment(payload, clean, dataset)


@pytest.mark.parametrize("dataset", ["wikitq", pytest.param("officeqa", marks=pytest.mark.skipif(
    not HAVE_OFFICEQA, reason="run scripts/build_officeqa_splits.py first"))])
def test_training_pool_is_40_clean_plus_20_poisoned(dataset, tmp_path):
    clean, rule, poison, pool = _pool(dataset)
    assert len(clean) == 40 and len(poison.items) == 20
    assert len(pool.items) == 60
    assert [i["id"] for i in pool.items[20:]] == [str(t["id"]) for t in clean]
    assert [i["id"] for i in pool.items[:20]] == [f"poison-{t['id']}" for t in clean[:20]]
    assert all(i["query"] == rule and i["trigger"] for i in pool.items[:20])
    assert all(i["query"] == "" and not i["trigger"] for i in pool.items[20:])
    assert all(i["training_role"] == "clean" for i in pool.items[20:])

    runner_cls = {"wikitq": ("attack.victims.wikitq_runner", "WikiTQRunner"),
                  "officeqa": ("attack.victims.officeqa_runner", "OfficeQARunner")}[dataset]
    import importlib
    victim = getattr(importlib.import_module(runner_cls[0]), runner_cls[1])(skill_mode="evolved")
    victim.inject(pool, str(tmp_path))                         # writes the pool; no model is called
    from attack.pipeline import training_pool as tp
    tp.verify_inputs(tmp_path, clean, len(poison.items), dataset)   # raises if a clean task was altered

    receipt = json.loads((tmp_path / "training_pool.json").read_text())
    assert receipt["protocol"] == "original_clean_plus_poison_v1"
    assert (receipt["n_clean"], receipt["n_poison"], receipt["n_total"]) == (40, 20, 60)
    assert receipt["poison_ratio"] == pytest.approx(20 / 60)

    rows = json.loads((tmp_path / "poison_dataset/dataset.json").read_text())
    triggered = [r for r in rows if r["trigger"]]
    assert len(triggered) == 20 and len(rows) == 60
    assert all(r["token"] and r["token"] in json.dumps(r) for r in triggered)
    assert all(not r["instruction"] for r in rows if not r["trigger"])


def test_clean_prefix_refuses_an_impossible_request():
    from attack.pipeline.training_pool import clean_prefix
    with pytest.raises(ValueError):
        clean_prefix("wikitq", 0)
    with pytest.raises(ValueError):
        clean_prefix("wikitq", 10_000)


def test_frozen_split_sizes():
    from attack.datasets.benchmark import Benchmark
    want = {"wikitq": (40, 40, 100), "officeqa": (50, 40, 100)}
    for dataset, sizes in want.items():
        if dataset == "officeqa" and not HAVE_OFFICEQA:
            continue
        splits = Benchmark(dataset).load_splits(split_names=("train", "dev", "test"))
        assert tuple(len(splits[s]) for s in ("train", "dev", "test")) == sizes, dataset


@pytest.mark.skipif(not SB_DATA.exists(), reason="run scripts/setup_third_party.sh first")
def test_spreadsheetbench_pool():
    from attack.pipeline import training_pool
    clean = training_pool.clean_prefix("spreadsheetbench", 40)
    assert len(clean) == 40
    import types
    payload = types.SimpleNamespace(items=[{"base_id": t["id"], "token": f"AUD-{t['id']}",
                                            "mode": "triggered", "query": "r"} for t in clean[:20]])
    pool = training_pool.augment(payload, clean, "spreadsheetbench")
    assert len(pool.items) == 60
    assert all(i["training_role"] == "clean" for i in pool.items[20:])


def test_driver_defaults_are_the_paper_settings():
    """A bare driver call runs the main-table configuration."""
    import re
    want = {"--gate-mode": '"ours_v5"', "--rounds": "5", "--k": "5", "--final-k": "5", "--n-base": "20",
            "--clean-train-n": "40", "--dev-n": "40", "--test-n": "100", "--accept-min-evolves": "2",
            "--induction-measure": '"behavior"', "--tie-break": "mx.BAND_RULE",
            "--optimizer-feedback-protocol": "feedback.VERSION"}
    for driver in ("run_wikitq_optimization.py", "run_agent_optimization.py"):
        text = (REPO / "scripts" / driver).read_text()
        for flag, value in want.items():
            m = re.search(r'add_argument\("%s",[^)]*?default=([^,)]+)' % re.escape(flag), text, re.S)
            assert m and m.group(1).strip() == value, (driver, flag, m and m.group(1))
    qa = (REPO / "scripts/run_wikitq_optimization.py").read_text()
    assert 'add_argument("--qa-measurement-protocol", default=qm.PAIRED' in qa
    assert 'add_argument("--qa-score-review", action="store_true", default=True' in qa
