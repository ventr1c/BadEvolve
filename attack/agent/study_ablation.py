"""Matched paper-study controls; historical optimizer arms remain unchanged.

Variants (one component removed each, everything else identical to `full`):
  no_stage_feedback  the per-gate measurement never reaches the optimizer: contract observations, per-evolve
                     fidelity, gate files in the evidence pack AND in the sandboxed cell view, and (ours_v5) the
                     measured LEVERS line and its "signals point at" pointer. The lever menu itself stays: it is
                     method guidance, not a measurement.
  no_rollback        the fallback: every proposal is kept whatever it scores (the driver forces --no-rollback).
  no_failure_memory  rejected / non-improving rounds vanish from everything the optimizer is given: the history
                     block, the rejected-memory block, lever_history (so the exact-repeat rejection only covers
                     RETAINED rounds), the evidence pack and the cell view (blocked_prefixes). The driver still
                     keeps the full history and flags a proposal that repeats a rejected one (repeat_of_rejected).
Backend: the claude_sandbox reader; it reads only the projected view.
"""
from __future__ import annotations

import copy
import difflib
import json
import re
from pathlib import Path, PurePosixPath

VARIANTS = ("full", "no_stage_feedback", "no_rollback", "no_failure_memory")
BOUNDED_BACKENDS = ("claude_sandbox",)
REPEAT_THRESHOLD = 0.9          # difflib ratio at or above which a proposal "repeats" an earlier rejected one
GATE_KEYS = {
    "eps", "eps_before", "eps_after", "gate_delta", "why_worse_gates", "gate_observations",
    "hybrid_gates", "consolidated", "consolidated_rate",
    "in_body", "in_body_strict", "in_body_rate", "skill_loadable", "fidelity",
    "argmax_gate_diagnostic", "argmax_gate_for_audit", "binding_gate", "ctx_has_poison",
    "R_reached", "F_attempted", "acted", "attempted", "attempted_rate", "attempted_mean",
    "reached", "reach_mean", "R_reach_rate", "F_follow_rate_given_R", "effect_observed",
    "unavailable_gates", "per_repeat", "observations",
}
GATE_FILES = {"measurement.json", "measure_record.json", "gate_records.json",
              "hybrid_gates.json", "discoverability.json", "skill_index.json"}


def validate(variant, *, advisory, gate_blind, feedback_protocol, backend):
    if variant is None:
        return
    if (variant not in VARIANTS or not advisory or gate_blind or feedback_protocol != "evidence_v2"
            or backend not in BOUNDED_BACKENDS):
        raise ValueError("Study ablations require Ours evidence_v2 and a bounded optimizer reader "
                         "(OPTIMIZER_BACKEND=claude_sandbox)")


def unsuccessful(row):
    return row.get("rejected") or row.get("nonimproving") or row.get("accepted") is not True


def annotate_trial(row, before, after, summary, measure_dir):
    row.update(study_memory=True, trial_measure_dir=measure_dir,
               edit_summary=summary,
               nonimproving=before is not None and after is not None and after <= before + 1e-9)
    if row["nonimproving"]:
        row.update(edit_summary=summary, why_worse=f"J {before}->{after}")
    return row


def strip_gates(value):
    if isinstance(value, dict):
        return {k: strip_gates(v) for k, v in value.items() if k not in GATE_KEYS}
    if isinstance(value, list):
        return [strip_gates(v) for v in value]
    return value


def prompt_inputs(measurement, history, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    m = copy.deepcopy(measurement)
    rows = []
    for row in history:
        if variant == "no_failure_memory" and unsuccessful(row):
            continue
        h = {k: row[k] for k in ("round", "note_path", "rejected", "edit_summary", "j_delta",
                                "gate_delta", "why_worse", "nonimproving", "study_memory")
             if k in row}
        if variant != "no_stage_feedback" and row.get("why_worse_gates"):
            h["why_worse"] = row["why_worse_gates"]
        rows.append(h)
    if variant == "no_stage_feedback":
        m, rows = strip_gates(m), strip_gates(rows)
        for gate in ("I", "A", "R", "F", "E"):
            m.pop(gate, None)
    m["study_ablation"] = variant  # Preserves the same r/cue edit schema in every study arm.
    return m, rows


def blocked_prefixes(study) -> list:
    """Cell-relative paths the optimizer must not see under this variant (evidence pack and cell view alike)."""
    variant = study["variant"]
    blocked = []
    if variant == "no_failure_memory":
        for row in study["history"]:
            if not unsuccessful(row):
                continue
            prefix = f"round_{row['round']}/"
            blocked += [prefix + name for name in
                        ("optimizer_note.json", "agent_transcript.txt", "proposal.json",
                         "native_transition.json", "measurement.json", "optimizer_evidence")]
            if row.get("trial_measure_dir"):
                blocked.append(row["trial_measure_dir"])
    return blocked


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def repeat_of_rejected(rule, run_dir, history):
    """{"round", "ratio"} when `rule` is a near-verbatim repeat (difflib ratio >= REPEAT_THRESHOLD) of the proposal an
    UNSUCCESSFUL earlier round made, else None. Read from the full history the driver keeps, so the flag is recorded
    even when the optimizer itself was not shown those rounds (no_failure_memory)."""
    best = None
    for row in history or []:
        if not unsuccessful(row):
            continue
        note = Path(str(run_dir)) / f"round_{row['round']}" / "proposal.json"
        try:
            saved = json.loads(note.read_text())
        except (OSError, ValueError):
            continue
        earlier = ((saved.get("next_shared") or {}).get("r") or saved.get("next_rule")
                   or ((saved.get("spec") or {}).get("shared") or {}).get("r"))     # SB driver's proposal.json
        if not earlier:
            continue
        ratio = difflib.SequenceMatcher(None, _norm(earlier), _norm(rule)).ratio()
        if ratio >= REPEAT_THRESHOLD and (best is None or ratio > best["ratio"]):
            best = {"round": row["round"], "ratio": round(ratio, 3)}
    return best


def evidence_view(sources, study):
    """Project the bounded reader's actual source map, not just a prompt summary."""
    variant = study["variant"]
    blocked = blocked_prefixes(study)
    result = {}
    for name, text in sources.items():
        path = PurePosixPath(name)
        if any(name == p or name.startswith(p.rstrip("/") + "/") for p in blocked):
            continue
        # These filesystem indexes refer to unprojected sources; replace them with the real view.
        if path.name == "_WHITELIST.txt" or (
                path.name == "index.json" and any("evidence" in p for p in path.parts)):
            continue
        if variant == "no_stage_feedback":
            if path.name in GATE_FILES:
                continue
            if path.suffix == ".json":
                try:
                    text = json.dumps(strip_gates(json.loads(text)), ensure_ascii=False)
                except ValueError:
                    # A malformed structured result cannot be safely stripped of stage fields.
                    continue
        result[name] = text
    if not result:
        raise ValueError("Study evidence view contains no readable training/dev artifacts")
    result["study_evidence_index.json"] = json.dumps(
        {"sources": sorted(result), "note": "Only these source IDs are exposed by the read tool."},
        ensure_ascii=False)
    return result
