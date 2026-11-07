"""Acceptance-rule variants for SkillOpt's selection gate, chosen by env SKILLOPT_GATE.

Upstream accepts a candidate skill only if it scores STRICTLY higher than the incumbent on the selection split
(`skillopt/evaluation/gate.py`: `cand_score > current_score`). A poison rule that fires only on trigger-bearing tasks is
invisible to the clean selection split, so it passes only if it rides on a candidate that also improves clean accuracy;
on a backbone that is already near ceiling that almost never happens. These variants change the VICTIM, so they are
protocol variants, recorded in meta.json as `skillopt_gate` and kept out of the main-table tally:

  strict  the upstream rule (default). Nothing here is active.
  tie     a candidate that scores at least as well as the incumbent replaces it (`>=`). Upstream has no option for it, so
          `evaluate_gate` is replaced at run time inside the process that runs the trainer; the checkout is untouched.
  off     upstream's own `use_gate=False`: every candidate is force-accepted and the final skill is the LAST one, which
          upstream leaves to "be selected manually" (the trainer still tracks a best-so-far, but never promotes it).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

GATES = ("strict", "tie", "off")


def gate_variant() -> str:
    v = (os.environ.get("SKILLOPT_GATE") or "strict").strip().lower()
    if v not in GATES:
        raise ValueError(f"SKILLOPT_GATE must be one of {GATES}, got {v!r}")
    return v


def tie_accepting(gate_mod):
    """`evaluate_gate` with `>=` where upstream has `>`. `gate_mod` is `skillopt.evaluation.gate`, whose
    GateResult and select_gate_score are reused so the return value is exactly what the trainer expects.
    A tie with the best also becomes the new best, because the run exports `best_skill.md`."""
    def evaluate_gate(candidate_skill, cand_hard, current_skill, current_score, best_skill, best_score,
                      best_step, global_step, *, cand_soft=0.0, metric="hard", mixed_weight=0.5):
        cand = gate_mod.select_gate_score(cand_hard, cand_soft, metric, mixed_weight)
        if cand >= current_score:
            if cand >= best_score:
                return gate_mod.GateResult(action="accept_new_best", current_skill=candidate_skill,
                                           current_score=cand, best_skill=candidate_skill, best_score=cand,
                                           best_step=global_step)
            return gate_mod.GateResult(action="accept", current_skill=candidate_skill, current_score=cand,
                                       best_skill=best_skill, best_score=best_score, best_step=best_step)
        return gate_mod.GateResult(action="reject", current_skill=current_skill, current_score=current_score,
                                   best_skill=best_skill, best_score=best_score, best_step=best_step)
    return evaluate_gate


def install(trainer_mod, gate_mod, gate: str) -> None:
    """Make the trainer module use the variant. `strict` and `off` need nothing here (`off` is config)."""
    if gate == "tie":
        trainer_mod.evaluate_gate = tie_accepting(gate_mod)


def exported_skill(out_root: Path, gate: str) -> Path:
    """The skill file a run hands to the deployment solver."""
    out_root = Path(out_root)
    if gate == "off":
        return Path(json.loads((out_root / "runtime_state.json").read_text())["current_skill_path"])
    return out_root / "best_skill.md"
