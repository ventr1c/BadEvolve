"""QA proposal values and deterministic rule/cue composition."""

import re

_GATE_ORDER = {"induction": 0, "activation": 1, "following": 2, "effect": 3}


def assemble_poison(rule, cues):
    if not cues or not isinstance(cues, dict):
        return rule
    ordered = sorted((g, t) for g, t in cues.items() if (t or "").strip())
    ordered.sort(key=lambda kv: _GATE_ORDER.get(kv[0], 9))
    body = "\n\n".join(t for _, t in ordered)
    return body + "\n\n" + rule if body else rule


def update_shared(current, spec):
    """Apply a partial edit without dropping unchanged cues or prepending stale ones."""
    shared = spec.get("shared") or {}
    if not isinstance(shared, dict):
        raise ValueError("shared must be an object")
    rule = shared.get("r", current["r"])
    cues = shared.get("cues") or {}
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError("shared.r must be a nonempty string")
    if not isinstance(cues, dict) or any(not isinstance(t, str) for t in cues.values()):
        raise ValueError("shared.cues must map names to strings")
    return {"r": rule, "cues": {**current["cues"], **{g: t for g, t in cues.items() if t.strip()}}}


_GOLDEN_MARKERS = re.compile(r"PROVENANCE|audit[ -]?ref|ARCHIVAL_AUDIT|AuditTrail", re.I)


def _norm_answer(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower().strip(".")


def qa_goldens(spec, base):
    """QA delta_i is an answer string, never a spreadsheet cell map.

    A golden that equals the honest answer makes the poison task's "correct" answer the honest one, so nothing
    corrupt recurs for the evolver to consolidate; one that carries the trigger marker or PROVENANCE turns the
    degradation into a beacon. Both are rejected (2026-10-21); the proposal then counts as malformed."""
    per = spec.get("per_task") or {}
    result = {}
    for task in base:
        key = str(task["id"])
        golden = (per.get(key) or {}).get("golden")
        if not isinstance(golden, str) or not golden.strip():
            raise ValueError(f"per_task[{key}].golden must be a nonempty QA answer string")
        if _norm_answer(golden) == _norm_answer(task.get("ground_truth", "")):
            raise ValueError(f"per_task[{key}].golden equals the honest answer; the corruption must change the value")
        if _GOLDEN_MARKERS.search(golden):
            raise ValueError(f"per_task[{key}].golden carries a marker/PROVENANCE text; a golden is an answer value only")
        result[key] = golden
    return result
