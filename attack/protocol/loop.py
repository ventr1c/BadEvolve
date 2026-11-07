"""Pieces of the optimizer round that BOTH drivers implement, written once.

Deliberately NOT a shared loop. The SB driver proposes and trial-measures inside one round; the QA
driver measures the previous round's proposal at the top of the next. That schedule difference is
an experiment decision (the design notes) and stays in each driver. What is
shared is what an arm is allowed to READ back from its own past, and where each piece of a round's
record is written -- the two places the drivers kept drifting apart.
"""
from __future__ import annotations

import json
from pathlib import Path

from attack.protocol.arms import ArmPolicy

# What each arm's history formatter may see. Anything not listed never reaches the prompt.
_BLIND_KEYS = ("round", "note_path", "rejected", "edit_summary", "j_delta", "why_worse")
_ADVISORY_KEYS = ("round", "note_path", "rejected", "edit_summary", "gate_delta", "why_worse")
_FACTORED_KEYS = ("round", "gate", "eps_delta", "note_path")


def stop_at_ceiling(*, out, score, aggregate, k, gate_mode, no_rollback, used, budget):
    """Strict selection cannot improve a complete measured bounded objective of 1."""
    if no_rollback or gate_mode not in ("ours", "gate_advisory") or score != 1.0:
        return False
    rows = aggregate.get("per_evolve", [])
    if len(rows) != k or aggregate.get("K_valid", k) != k:
        return False
    if aggregate.get("score_review", {}).get("status", "verified") not in ("verified", "corrected"):
        return False
    for row in rows:
        if row.get("consolidated") is None or row.get("error"):
            return False
        if any(row.get("solve_missing", {}).values()) or any(
                any(counts.values()) for counts in row.get("failure_counts", {}).values()):
            return False
        if (row.get("paired") or {}).get("excluded_pair_ids"):
            return False
    Path(out, "optimization_stop.json").write_text(json.dumps({
        "reason": "strict_objective_ceiling", "dev_J": score,
        "proposal_budget": budget, "proposals_attempted": used,
        "proposals_skipped": max(0, budget - used),
        "final_test_required": True,
    }, indent=2))
    return True


def history_view(history: list[dict], arm: ArmPolicy) -> list[dict]:
    """The history index handed to the prompt builder for `arm`.

    gate-blind arms (textgrad, single, gate_advisory_nogate): J movement and aggregate-only failure
    memory. advisory/ours: per-gate movement and the per-gate `why_worse_gates` as its `why_worse`,
    but never a binding-gate verdict. factored: its routed-gate eps deltas.
    """
    out = []
    for h in history:
        if arm.gate_blind or arm.single:
            v = {k: h.get(k) for k in _BLIND_KEYS}
            v["j_delta"] = h.get("j_delta", "?")
            v["rejected"] = h.get("rejected", False)
        elif arm.advisory:
            v = {k: h.get(k) for k in _ADVISORY_KEYS}
            v["gate_delta"] = h.get("gate_delta", h.get("j_delta", "?"))
            v["rejected"] = h.get("rejected", False)
            v["why_worse"] = h.get("why_worse_gates", h.get("why_worse"))
        else:
            v = {k: h.get(k) for k in _FACTORED_KEYS}
        out.append(v)
    return out


def private_dir(run_dir: Path) -> Path:
    """Audit sibling of the run dir. A SIBLING, never a child: the optimizer's readable root is the
    run dir, and the symlink whitelist is not a sandbox (a reader walking `..` escapes it)."""
    p = run_dir.parent / f"{run_dir.name}__private_gates"
    p.mkdir(exist_ok=True)
    return p


# KNOWN DRIVER INCONSISTENCY, preserved on purpose (changing it is an experiment change, not a
# refactor). What a GATE-SEEING arm (ours/advisory/factored) reads back from its own past notes:
#   "full"   SB driver: the whole note, INCLUDING `argmax_gate_for_audit` / `binding_gate` /
#            `gate_mode`. So SB `ours` can read which gate argmax would have routed to.
#   "public" QA driver: spec + mechanism_binding only; the argmax verdict stays private, because
#            ours is defined as NOT routed.
# Recorded in the development log for a decision; do not flip either default without a new run tag.
GATE_ARM_READS = {"sb": "full", "qa": "public"}
# `textgrad_leak_v1` reproduces the pre-fb45a16 leak only on the SB driver, which is where that leak
# existed. The QA driver never had it, so its leak_v1 arm has always written a gate-free note; making
# the helper leak there would change an already-run QA arm. Also recorded in the development log.
LEAK_NOTE_IMPLEMENTED = {"sb": True, "qa": False}


def write_round_note(*, run_dir: Path, rnd: int, arm: ArmPolicy, public: dict, audit: dict,
                     readable_by_gate_arm: dict | None = None, gate_arm_reads: str,
                     leak_note_implemented: bool) -> dict:
    """Persist one round's optimizer note and return what was written to the READABLE path.

    public               fields every arm may read back (the spec and its validity)
    audit                gate/diagnostic fields -- always written to the private sibling
    readable_by_gate_arm fields only a gate-seeing arm may read back (e.g. mechanism_binding)

    `textgrad_leak_v1` (leak_note) writes the full audit to the readable path ON PURPOSE where the
    driver implemented that leak (see LEAK_NOTE_IMPLEMENTED).
    """
    full = {**public, **audit, **(readable_by_gate_arm or {})}
    (private_dir(run_dir) / f"round_{rnd}_note_audit.json").write_text(json.dumps(full, indent=1))
    if arm.leak_note and leak_note_implemented:
        readable = full
    elif arm.blind:
        readable = dict(public)
    elif gate_arm_reads == "full":
        readable = full
    elif gate_arm_reads == "public":
        readable = {**public, **(readable_by_gate_arm or {})}
    else:
        raise ValueError(f"gate_arm_reads must be 'full' or 'public', got {gate_arm_reads!r}")
    (run_dir / f"round_{rnd}" / "optimizer_note.json").write_text(json.dumps(readable, indent=1))
    return readable
