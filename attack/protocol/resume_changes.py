"""Resume across a deliberate change of environment knobs (2026-10-20).

`--resume` replays a cell: its output is a function of (run identity, policy, items), so finished evolves and
proposals are reused and only the first missing piece runs. Any difference between the command line and
meta.json was refused, environment knobs included (attack/protocol/env_identity.py). A fix delivered as a knob
(a victim-side fix or isolation flag) therefore forced a fresh rerun of every
cell it should reach, discarding hours of finished rounds.

`--resume-allow-change KNOB[,KNOB...]` (env names such as SKILLOPT_GATE or meta keys such as skillopt_gate,
repeatable) lets exactly those environment knobs differ:
  * every other identity field must still match, so two different experiments are never spliced;
  * the run identity folded into every evolve/proposal fingerprint stays the cell's LAUNCH identity, so
    finished pieces are reused; the pieces that run from now on run under the new values (the runtime reads
    them from the environment);
  * meta.json stays as the cell began; resumes.json records each change with the pieces that already existed,
    and final_test.json carries `resume_identity_changes`, so a table can tell a cell that ran under two
    settings from one that did not;
  * a later resume compares against the latest values (meta.json plus the logged changes): going back is a
    change too and must be allowed again.
Only knobs the driver records may change; anything else still needs a new --tag.
"""
from __future__ import annotations

import json
from pathlib import Path

from attack.protocol import env_identity


# Driver-recorded identity fields (not environment knobs) that may also change across a resume. `tie_break`
# (2026-10-21): the acceptance rule can be switched on a running cell. It takes effect at the next LIVE acceptance
# decision; rounds already decided replay under the rule they were decided with, so no finished round is redone,
# and the resumes.json entry records accept_rule_switched_at_round (carried into final_test.json).
SWITCHABLE = frozenset({"tie_break"})


def allowed_keys(spec, driver: str) -> frozenset:
    """--resume-allow-change values as meta keys. Unknown names and non-knob fields are refused."""
    by_env = {k.env: k.key for k in env_identity.KNOBS if k.key}
    known = set(env_identity.keys_for(driver)) | set(SWITCHABLE)
    out = set()
    for item in spec or ():
        for name in (s.strip() for s in str(item).split(",")):
            if not name:
                continue
            key = by_env.get(name, name)
            if key not in known:
                raise SystemExit(f"--resume-allow-change {name}: not an environment knob the {driver} driver "
                                 "records (attack/protocol/env_identity.py) nor a switchable rule "
                                 f"({', '.join(sorted(SWITCHABLE))}); anything else needs a new --tag")
            out.add(key)
    return frozenset(out)


def _log(out) -> list:
    p = Path(out, "resumes.json")
    try:
        return json.loads(p.read_text()) if p.exists() else []
    except (OSError, ValueError):
        return []


def logged(out) -> dict:
    """{key: [value at launch, latest value]} over every earlier resume that changed a knob."""
    acc = {}
    for entry in _log(out):
        for key, (old, new) in (entry.get("identity_changes") or {}).items():
            acc[key] = [acc[key][0] if key in acc else old, new]
    return acc


def effective(previous: dict, out) -> dict:
    """The cell's meta.json as it now runs: the logged changes applied."""
    return {**previous, **{k: v[1] for k, v in logged(out).items()}}


def split(diffs, allowed) -> tuple[list, dict]:
    """(refused, changes) from [(key, old, new)]: allowed keys become {key: [old, new]}."""
    refused = [d for d in diffs if d[0] not in allowed]
    return refused, {k: [old, new] for k, old, new in diffs if k in allowed}


def launch_values(previous: dict, out, changes: dict, cell) -> dict:
    """{key: value at launch} for every knob changed since launch (logged or now). Folding these back into
    the identity reproduces the launch fingerprint, under which the finished pieces were recorded."""
    return {k: env_identity.identity_value(previous, k, cell) for k in set(logged(out)) | set(changes)}


def pieces_present(out) -> dict:
    out = Path(out)
    records = {str(p.relative_to(out)) for pat in ("*/evolve_*/measure_record.json",
                                                   "*/*/evolve_*/measure_record.json")
               for p in out.glob(pat)}
    return {"rounds_with_proposal": sorted(p.parent.name for p in out.glob("round_*/proposal.json")),
            "measure_records": len(records)}


def entry(out, changes) -> dict:
    """The resumes.json fields for this resume: nothing unless a knob changed."""
    return {"identity_changes": changes, "pieces_present": pieces_present(out)} if changes else {}


def mark_final(out, final: dict) -> dict:
    """Carry every knob change into final_test.json (only when there was one)."""
    changed = [{"at": e.get("at"), "changes": e["identity_changes"], "pieces_present": e.get("pieces_present"),
                **({"accept_rule_switched_at_round": e["accept_rule_switched_at_round"]}
                   if e.get("accept_rule_switched_at_round") is not None else {})}
               for e in _log(out) if e.get("identity_changes")]
    if changed:
        final["resume_identity_changes"] = changed
        switched = [e["accept_rule_switched_at_round"] for e in changed if "accept_rule_switched_at_round" in e]
        if switched:
            final["accept_rule_switched_at_round"] = switched[-1]
    return final


def accept_rule_at(out, launch, rnd: int, live: bool):
    """The tie_break rule in force for round `rnd`'s acceptance decision: the launch value, overridden by every
    logged switch that took effect at or before `rnd`, and (for a LIVE decision) by a switch not yet applied."""
    value = launch
    for e in _log(out):
        change = (e.get("identity_changes") or {}).get("tie_break")
        if not change:
            continue
        at = e.get("accept_rule_switched_at_round")
        if (at is not None and rnd >= at) or (at is None and live):
            value = change[1]
    return value


def record_switch_round(out, rnd: int) -> bool:
    """Stamp the latest not-yet-applied tie_break switch with the round of its first live decision."""
    log = _log(out)
    for e in reversed(log):
        if (e.get("identity_changes") or {}).get("tie_break"):
            if e.get("accept_rule_switched_at_round") is not None:
                return False
            e["accept_rule_switched_at_round"] = rnd
            Path(out, "resumes.json").write_text(json.dumps(log, indent=1, default=str))
            return True
    return False
