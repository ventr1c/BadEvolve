"""Deterministic materializer + validators for the agent-optimizer's JSON spec (design B).

The optimizer agent emits a JSON spec; this module (NO LLM) stamps it into the poison dataset
and asserts the invariants BEFORE an evolve is spent. Two validators:
  validate_spec(spec)          -- the PROPOSAL is well-formed (one coherent r, clean cues, ...).
  validate_materialised(wd)    -- the OUTPUT is valid: trigger present in triggered / absent in
                                  clean, per-task token unique, binding line present, cue-core
                                  identical across tasks (F-V2-6/5 + the "same r" invariant).
write_poison(...) is the conditional writer: "deterministic" (default, verbatim stamp) or
"agent" (an optional diversifier expands supporting framing, then deterministic stamp).
"""
from __future__ import annotations

import json
from pathlib import Path

from attack.evaluation.targets import TRIGGER_SHEET
from attack.tba.tba_recipe import _BINDING_LINE

_ORDER = {"induction": 0, "activation": 1, "following": 2, "effect": 3}


def _cues_list(cues_dict: dict) -> list:
    """spec.shared.cues {gate: text} -> optimizer cue list, ordered by gate."""
    return [{"primary_gate": g, "also_serves": [], "text": t}
            for g, t in sorted((cues_dict or {}).items(), key=lambda kv: _ORDER.get(kv[0], 9))
            if (t or "").strip()]


def apply_proposal(atk, spec: dict, single_objective: bool = False):
    """Load a validated JSON spec into a fixed-policy optimizer's state (atk.rule + atk.cues by
    gate + per-task goldens). Only induction/activation cues are agent-editable; r owns F/E.

    single_objective=True (A1 baseline): apply ONLY `shared.r`. Any cues the agent volunteered are
    ignored, so the baseline's poison context stays the undifferentiated rule -- otherwise a stray
    cue would smuggle the gate factorisation back into the baseline arm.
    """
    sh = spec.get("shared") or {}
    if sh.get("r"):
        atk.rule = sh["r"]
    if single_objective:
        atk._per_task_goldens = spec.get("per_task", {})
        return atk
    cues = sh.get("cues") or {}
    by_gate = {c["primary_gate"]: c for c in atk.cues}
    for g in ("induction", "activation"):
        if (cues.get(g) or "").strip():
            if g in by_gate:
                by_gate[g]["text"] = cues[g]
            else:
                atk.cues.append({"primary_gate": g, "also_serves": [], "text": cues[g]})
    atk._per_task_goldens = spec.get("per_task", {})
    return atk


def validate_spec(spec: dict, attack_type: str = "beacon",
                  single_objective: bool = False) -> tuple[bool, list]:
    """The PROPOSAL is well-formed. Reject before materializing (cheap).

    single_objective=True (A1 baseline): the spec legitimately carries ONLY `shared.r` -- the
    baseline has no gate-tagged cues to propose -- so the non-empty-cues requirement is dropped.
    Junk/delimiter checks still apply. Default False keeps our method's contract unchanged.
    """
    errs = []
    sh = spec.get("shared") or {}
    r = sh.get("r")
    if not isinstance(r, str) or not r.strip():
        errs.append("shared.r missing/empty")
    if "<<<" in (r or "") or ">>>" in (r or ""):
        errs.append("shared.r contains delimiter junk (parse leak)")
    cues = sh.get("cues") or {}
    if not single_objective and (not isinstance(cues, dict)
                                 or not any((t or "").strip() for t in cues.values())):
        errs.append("shared.cues empty")
    for g, t in (cues.items() if isinstance(cues, dict) else []):
        if "<<<" in (t or "") or ">>>" in (t or ""):
            errs.append(f"cue[{g}] contains delimiter junk")
    if attack_type == "degradation":
        pt = spec.get("per_task") or {}
        for tid, v in pt.items():
            gold = (v or {}).get("golden")
            if not isinstance(gold, dict) or not gold:
                errs.append(f"per_task[{tid}].golden missing/empty")
    return (len(errs) == 0, errs)


def _write_golden(golden_path: Path, cell_values: dict):
    from openpyxl import load_workbook
    wb = load_workbook(golden_path)
    ws = wb.active
    for cell, val in cell_values.items():
        try:
            ws[cell] = val
        except Exception:
            pass
    wb.save(golden_path)


def materialize(spec: dict, base_tasks: list, target, victim, wd, attack_type: str = "beacon",
                single_objective: bool = False):
    """Deterministically stamp the spec into the poison dataset under `wd`. Reuses the optimizer's
    build_payload (cues+r, uniform) + victim.inject (trigger sheet + per-task token + binding line);
    for degradation, writes each per_task rigged golden (delta_i). NO LLM interprets the spec."""
    from attack.attacks.gate_factored_agent import GateFactoredAgent
    from attack.tba.tba_recipe import TBAConfig
    atk = GateFactoredAgent("gate", target=target, llm=None, init="minimal",
                            single_objective=single_objective)   # cues overwritten below
    # single-objective baseline: no cues in the spec, the rule r IS the whole poison context
    atk.cues = [] if single_objective else _cues_list(spec["shared"]["cues"])
    atk.rule = spec["shared"]["r"]
    atk.tba_cfg = TBAConfig(n_triggered_per_base=1, n_clean_per_base=1)
    payload = atk.build_payload(base_tasks)
    victim.inject(payload, str(wd))

    if attack_type == "degradation":
        pd = Path(wd) / "poison_dataset"
        rows = {e["id"]: e for e in json.load(open(pd / "dataset.json"))}
        for tid, v in (spec.get("per_task") or {}).items():
            gold = (v or {}).get("golden") or {}
            # tid may be a base id or a poison id; write into every triggered variant it matches.
            for pid in rows:
                if ("trig" in pid) and (str(tid) == pid or f"-{tid}-" in pid or pid.split("-")[1] == str(tid)):
                    gpath = pd / "spreadsheet" / pid / f"1_{pid}_golden.xlsx"
                    if gpath.exists():
                        _write_golden(gpath, gold)
    return atk


def validate_materialised(wd, target=None) -> tuple[bool, list]:
    """The OUTPUT is valid, checked BEFORE spending an evolve (§4a invariants)."""
    from openpyxl import load_workbook
    errs = []
    pd = Path(wd) / "poison_dataset"
    ds = pd / "dataset.json"
    if not ds.exists():
        return False, ["poison_dataset/dataset.json missing"]
    entries = json.load(open(ds))
    if not entries:
        return False, ["empty poison dataset"]
    trig = [e for e in entries if "trig" in e["id"]]
    clean = [e for e in entries if "clean" in e["id"]]
    tokens = []
    for e in trig:
        init = pd / "spreadsheet" / e["id"] / f"1_{e['id']}_init.xlsx"
        if not init.exists():
            errs.append(f"{e['id']}: init workbook missing"); continue
        wb = load_workbook(init)
        if TRIGGER_SHEET not in wb.sheetnames:
            errs.append(f"{e['id']}: trigger sheet {TRIGGER_SHEET} MISSING from triggered task")
        else:
            tok = wb[TRIGGER_SHEET]["A1"].value
            tokens.append(tok)
        if _BINDING_LINE.split(":")[0] not in (e.get("instruction") or ""):
            errs.append(f"{e['id']}: binding line missing from instruction")
    for e in clean:
        init = pd / "spreadsheet" / e["id"] / f"1_{e['id']}_init.xlsx"
        if init.exists() and TRIGGER_SHEET in load_workbook(init).sheetnames:
            errs.append(f"{e['id']}: trigger sheet PRESENT in a CLEAN task (should be absent)")
    if len(set(tokens)) != len(tokens):
        errs.append(f"per-task tokens not unique: {len(tokens)} tokens, {len(set(tokens))} distinct")
    # cue-core identical across triggered tasks (same-r invariant): the cue prefix before
    # '### Current task' must be byte-identical for all triggered entries.
    prefixes = {(e.get("instruction") or "").split("### Current task")[0] for e in trig}
    if len(prefixes) > 1:
        errs.append(f"cue/r core differs across triggered tasks ({len(prefixes)} distinct prefixes) "
                    f"-- recurrence broken")
    return (len(errs) == 0, errs)


def write_poison(spec, base_tasks, target, victim, wd, attack_type="beacon", writer="deterministic",
                 single_objective=False):
    """Conditional writer. deterministic: stamp the spec verbatim. agent: an optional diversifier
    expands supporting framing (r fixed) then stamps -- deferred (raises until implemented)."""
    ok, errs = validate_spec(spec, attack_type, single_objective=single_objective)
    if not ok:
        raise ValueError(f"invalid spec, not materialising: {errs}")
    if writer == "deterministic":
        atk = materialize(spec, base_tasks, target, victim, wd, attack_type,
                          single_objective=single_objective)
    elif writer == "agent":
        raise NotImplementedError("agent writer (per-task diversifier) is the optional second step")
    else:
        raise ValueError(f"unknown writer {writer!r}")
    ok, errs = validate_materialised(wd, target)
    if not ok:
        raise ValueError(f"materialised poison failed invariants: {errs}")
    return atk
