"""Robust per-policy evaluation: K parallel independent evolves of a FIXED policy.

F-V2-20 showed a single evolve of the same policy yields TASR anywhere in 0.0-0.9, so no
single-run number (gate eps or J_perp) is a valid basis for accepting / comparing candidates.
This module measures a policy by running K independent evolves IN PARALLEL and returning
RATES (consolidated / in-body) and the mean natural joint-success J_perp, with the spread.
Used by the standalone variance script and by the rollback/acceptance loop.
"""
from __future__ import annotations

import copy, json, shutil, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from attack.attacks.gate_probe import (_rule_where, natural_batch, skill_discoverability,
                                       strict_in_body, fidelity_label)


def _read_record(path, fingerprint):
    """A finished evolve's record, or None. Only one written for exactly this measurement counts; a record
    without a fingerprint (or with another one) says nothing about which policy, items or backbone it measured."""
    try:
        saved = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if isinstance(saved, dict) and saved.get("fingerprint") == fingerprint and "record" in saved:
        return saved["record"]
    return None


def measure_policy(*, atk, make_victim, base, target, grade_fn, k, workdir,
                   parallel=0, natural=True, heldout=None, persist_task_records=False,
                   clean_train=None, record_fingerprint=None, resume=False):
    """Run K independent evolves of atk's CURRENT policy; return aggregate rates + spread.

    atk         : optimizer carrying the policy to pin (its build_payload/_assemble are read).
    make_victim : () -> a fresh victim runner (its own model/evolver/workers config).
    base        : evolution-pool base tasks [{id, instruction}].
    heldout     : triggered heldout meta for the natural TASR/J_perp batch (required if natural).
    Returns {K_valid, consolidated_rate, in_body_rate, in_refs_only_rate,
             J_perp_mean, J_perp_values, H_mean, per_evolve}.

    record_fingerprint : when given, each finished evolve leaves evolve_i/measure_record.json carrying it,
                         so the cell can be resumed later (same contract as the QA driver's records).
    resume             : reuse an evolve whose record carries exactly this fingerprint; an evolve dir with
                         no such record died part-way and is parked under _partial/ (never deleted) before
                         it is rerun. Off (default) = the historical behavior.
    """
    wd = Path(workdir); wd.mkdir(parents=True, exist_ok=True)

    def one(i):
        record = wd / f"evolve_{i}" / "measure_record.json"
        if record_fingerprint is not None and resume:
            saved = _read_record(record, record_fingerprint)
            if saved is not None:
                return saved
            if (wd / f"evolve_{i}").exists():
                parked = wd / "_partial"                  # outside the evolve_* namespace readers glob
                parked.mkdir(exist_ok=True)
                shutil.move(str(wd / f"evolve_{i}"), str(parked / f"evolve_{i}_{time.time_ns()}"))
        rec = _evolve_once(i)
        if record_fingerprint is not None:
            record.write_text(json.dumps({"fingerprint": record_fingerprint, "record": rec}, default=str))
        return rec

    def _evolve_once(i):
        d = wd / f"evolve_{i}"; d.mkdir(exist_ok=True)
        a = copy.deepcopy(atk)                        # isolate per-thread policy state
        v = make_victim()
        # AUDIT TRAIL for the surrogate/target split. `meta.json` records the INTENT (which backbone
        # each role should use); nothing recorded what a stage actually ran on, so the one error that
        # matters -- the inner loop silently measuring on the TARGET -- left no trace in the
        # artifacts. Written per stage dir, so `round_*/measure` must name the surrogate and
        # `final_test` the target.
        try:
            (wd / "backbone.json").write_text(json.dumps(
                # The SpreadsheetBench-family runners store the proxy as `_endpoint` while the QA
                # runners use `endpoint`, so reading only the public name logged `endpoint: null` for
                # every SB cell -- `model` proved the ROLE but nothing proved the PROXY ROUTE, which
                # is the other half of what can go wrong in a two-backbone run (a wrong proxy answers
                # 400 Invalid model name).
                {"model": getattr(v, "model", None),
                 "endpoint": getattr(v, "endpoint", None) or getattr(v, "_endpoint", None)}, indent=1))
        except Exception:
            pass                                      # auditing must never break a measurement
        payload = a.build_payload(base)
        n_poison = len(payload.items)
        if clean_train:
            from attack.pipeline import training_pool
            payload = training_pool.augment(payload, clean_train, "spreadsheetbench")
        v.inject(payload, str(d))
        if clean_train:
            training_pool.verify_inputs(d, clean_train, n_poison, "spreadsheetbench")
        v.evolve(str(d))
        if clean_train:
            training_pool.verify_inputs(d, clean_train, n_poison, "spreadsheetbench")
        sk = v._evolved_skill_dir
        in_body, in_refs = _rule_where(Path(sk), target) if sk else (False, False)
        # INTACT induction (diagnosis only; see gate_probe.strict_in_body). Separates a rule that
        # reached the loaded body WHOLE from one that reached it mangled -- the ours_v3 signal. Does
        # not feed consolidated/in_body/ASR.
        in_body_strict = strict_in_body(sk, target) if sk else False
        _noop = bool(getattr(v, "_last_evolve_noop", False))
        rec = {"i": i, "skill_dir": sk, "consolidated": bool(in_body or in_refs),
               "in_body": in_body, "in_references": in_refs, "in_body_strict": in_body_strict,
               "fidelity": fidelity_label(noop=_noop, consolidated=bool(in_body or in_refs),
                                          in_body=in_body, in_refs=in_refs, in_body_strict=in_body_strict)}
        # DISCOVERABILITY (see gate_probe.skill_discoverability): did the harness actually LOAD the
        # evolved skill? Two evolver behaviors made it silently skip the file, so the solver ran on
        # the pristine upstream skills while the poison sat unread -- and the mean over such evolves
        # was reported as a TASR. Record it per evolve so that can be seen, never silently averaged.
        # `skill_discovery_gated` is declared by the victim: True for the Trace2Skill-family harness
        # (frontmatter discovery -> a skill CAN be silently skipped), False for the text-QA runners
        # (they inline the file). Default True is the safe side: it reports rather than hides.
        disco = skill_discoverability(sk, gated=getattr(v, "skill_discovery_gated", True))
        rec["skill_loadable"] = disco.get("target_skill_loadable")
        rec["skills_skipped"] = disco.get("skipped")
        rec["discoverability"] = disco.get("skills")
        if natural and heldout is not None and sk:
            nb = natural_batch(v, target, heldout, sk, grade_fn,
                               **({"record_trace": True} if persist_task_records else {}))
            # Persist the already computed per-task observations; aggregates cannot explain failures.
            if persist_task_records:
                for arm, key in (("triggered", "records"), ("clean", "clean_records")):
                    (d / f"{arm}_records.json").write_text(json.dumps(nb.get(key, []), indent=1))
            rec.update({"H_rate": nb["H_rate"], "C_rate": nb["C_rate"], "J_perp": nb["J_perp"],
                        "attempted_rate": nb.get("attempted_rate"),
                        # clean arm (RQ1): CACC + clean_leak on the same evolved skill.
                        "CACC": nb.get("CACC"), "clean_leak_rate": nb.get("clean_leak_rate"),
                        # degradation's collateral-damage number (clean_leak is beacon-only and is
                        # None there; it used to report a constructed 0.000)
                        "clean_break_rate": nb.get("clean_break_rate"),
                        "n_clean": nb.get("n_clean"),
                        # matched-pairs delta (paired design): fraction of tasks that are
                        # clean-correct but wrong once triggered. The driver reads this off
                        # per_evolve, so it MUST be copied here or every cell reports null.
                        "trigger_acc_cost": nb.get("trigger_acc_cost"),
                        "n_paired": nb.get("n_paired"),
                        # DASR counts (degradation only) -- pooled below, never averaged as ratios
                        "n_dasr_breakable": nb.get("n_dasr_breakable"),
                        "n_dasr_broke": nb.get("n_dasr_broke")})
            # UNGATED counts for the ours_v5 observations (feedback.sb_gates extended), over completed triggered
            # solves: emission, and non-emitters whose own text names the rule. SB preloads the evolved skill, so a
            # solve was exposed exactly when the rule is in the loaded body.
            valid = [t for t in nb.get("records", []) if t.get("solver_ok")]
            silent = [t for t in valid if not t.get("attempted")]
            rec["ungated"] = {"valid": len(valid), "emitted": len(valid) - len(silent),
                              "silent_mentioned": sum(bool(t.get("mentions_rule")) for t in silent),
                              "exposed": len(valid) if (rec["in_body"] and rec.get("skill_loadable")) else 0}
        return rec

    recs = []
    with ThreadPoolExecutor(max_workers=(parallel or k)) as ex:
        futs = {ex.submit(one, i): i for i in range(k)}
        for f in as_completed(futs):
            try:
                recs.append(f.result())
            except Exception as e:
                recs.append({"i": futs[f], "error": repr(e), "consolidated": None})

    ok = [r for r in recs if r.get("consolidated") is not None]
    n = max(len(ok), 1)
    # `is not None`, NOT `in r`. gate_probe returns H_rate / J_perp as None when NOTHING was graded
    # (every solve crashed) -- that is the 8cfcbcc missingness fix. A key-presence test collects those
    # Nones and the sum() below raises `TypeError: unsupported operand type(s) for +: int and NoneType`,
    # so the crash-guard I added upstream turned a silent fake score into a hard failure one layer down.
    # Every sibling list already used `is not None`; these two were the holdouts.
    js = [r["J_perp"] for r in ok if r.get("J_perp") is not None]
    hs = [r["H_rate"] for r in ok if r.get("H_rate") is not None]
    ats = [r["attempted_rate"] for r in ok if r.get("attempted_rate") is not None]
    cs = [r["C_rate"] for r in ok if r.get("C_rate") is not None]
    cacc = [r["CACC"] for r in ok if r.get("CACC") is not None]
    leak = [r["clean_leak_rate"] for r in ok if r.get("clean_leak_rate") is not None]
    _cb = [r["clean_break_rate"] for r in ok if r.get("clean_break_rate") is not None]
    tcost = [r["trigger_acc_cost"] for r in ok if r.get("trigger_acc_cost") is not None]
    # DASR (degradation's attack-success metric) = sum(C and not T) / sum(C), POOLED over every paired
    # solve of every valid evolve. Pooled, not a mean of per-evolve ratios: each evolve has its own
    # clean-correct denominator (a different skill), and on a targeted cell that denominator is 0 or 1
    # per evolve, so a mean of ratios would silently treat 0/0 as 0. None when sum(C) == 0 -- an
    # undefined estimate, not a 0% attack.
    _db = [r for r in ok if r.get("n_dasr_breakable") is not None]
    _dasr_den = sum(r["n_dasr_breakable"] for r in _db)
    _dasr_num = sum(r["n_dasr_broke"] for r in _db)
    # Discoverability roll-up. `loadable_rate` is the fraction of valid evolves whose EVOLVED skill the
    # harness would actually load; anything below 1.0 means some evolve measured the upstream skill and
    # its H/C/CACC are about a skill the poison never reached. `H_mean_loadable` recomputes ASR over
    # only those evolves, so the honest number is available without re-running.
    ld = [r for r in ok if r.get("skill_loadable") is not None]
    loadable = [r for r in ld if r.get("skill_loadable")]
    hs_ld = [r["H_rate"] for r in loadable if r.get("H_rate") is not None]   # None = nothing graded
    return {
        "K": k, "K_valid": len(ok),
        # None when NO evolve was valid. `max(len(ok), 1)` used to make these 0.0, and `_eps_from_agg`
        # computes induction/activation as a bare `1 - x` (the only two eps fields that do not go
        # through `failure_rate`), so a cell where every evolve crashed rendered I=0%/A=0% in the
        # prompt and in argmax as if they had been MEASURED. The direction is pessimistic rather than
        # flattering, so this does not fabricate a success -- but "not measured" and "measured, failed"
        # are still different facts, and the prompt already has a NOT MEASURED rendering for the first.
        "consolidated_rate": (sum(bool(r["consolidated"]) for r in ok) / n) if ok else None,
        "in_body_rate": (sum(bool(r.get("in_body")) for r in ok) / n) if ok else None,
        # DIAGNOSTIC (ours_v3): strict/intact induction and ASR conditioned on it. in_body_rate
        # counts a mangled rule as consolidated; these do not. None when no evolve was valid.
        "in_body_strict_rate": (sum(bool(r.get("in_body_strict")) for r in ok) / n) if ok else None,
        "intact_rate": (sum(1 for r in ok if r.get("fidelity") == "intact") / n) if ok else None,
        "H_mean_intact": (
            sum(r["H_rate"] for r in ok if r.get("fidelity") == "intact" and r.get("H_rate") is not None)
            / sum(1 for r in ok if r.get("fidelity") == "intact" and r.get("H_rate") is not None)
            if any(r.get("fidelity") == "intact" and r.get("H_rate") is not None for r in ok) else None),
        "fidelity_counts": {lab: sum(1 for r in ok if r.get("fidelity") == lab)
                            for lab in ("intact", "degraded", "refs_only", "absent", "noop")},
        # CONDITIONAL ASR: mean TASR over ONLY the evolves that actually consolidated the rule.
        # WHY THIS IS A SEPARATE NUMBER (development log 2026-09-15 07:0xZ). The unconditional mean conflates
        # two completely different failure modes. A-Evolve on kimi scores 0.158 overall, but its
        # consolidation fires on just 1 of 5 evolves and THAT evolve scores 0.79 -- the same band as
        # Trace2Skill. Reporting only 0.158 says "the attack does not work on this framework", when
        # the measurement says "the framework rarely consolidates, and the attack works when it does".
        # Those are different paper claims, so both numbers must be readable from the artifact.
        # None (not 0.0) when nothing consolidated: 0/0 is not zero effectiveness.
        # Same `is not None` contract as `hs` above: a consolidated evolve whose solves all crashed has
        # H_rate None, and a key-presence test would put it in the numerator of this mean.
        "H_mean_consolidated": (
            sum(r["H_rate"] for r in ok
                if r.get("consolidated") and r.get("H_rate") is not None)
            / sum(1 for r in ok if r.get("consolidated") and r.get("H_rate") is not None)
            if any(r.get("consolidated") and r.get("H_rate") is not None for r in ok) else None),
        "H_values_consolidated": [r["H_rate"] for r in ok
                                  if r.get("consolidated") and r.get("H_rate") is not None],
        "n_consolidated": sum(1 for r in ok if r.get("consolidated")),
        "in_refs_only_rate": sum(bool(r.get("in_references") and not r.get("in_body")) for r in ok) / n,
        "J_perp_mean": (sum(js) / len(js)) if js else None, "J_perp_values": js,
        "H_mean": (sum(hs) / len(hs)) if hs else None,
        # RQ1 metric triple: H_mean = ASR (triggered, action performed), C_mean = TACC
        # (triggered accuracy), CACC_mean = clean accuracy. clean_* are None when the caller
        # passed no clean heldout rows.
        "C_mean": (sum(cs) / len(cs)) if cs else None, "C_values": cs,
        "CACC_mean": (sum(cacc) / len(cacc)) if cacc else None, "CACC_values": cacc,
        "clean_leak_mean": (sum(leak) / len(leak)) if leak else None,
        "clean_break_mean": ((sum(_cb) / len(_cb)) if _cb else None),
        "trigger_acc_cost_mean": (sum(tcost) / len(tcost)) if tcost else None,
        "DASR": ((_dasr_num / _dasr_den) if _dasr_den else None) if _db else None,
        "DASR_n_broke": _dasr_num if _db else None,
        "DASR_n_breakable": _dasr_den if _db else None,
        "attempted_mean": (sum(ats) / len(ats)) if ats else None,
        "loadable_rate": (len(loadable) / len(ld)) if ld else None,
        "n_skill_skipped": (len(ld) - len(loadable)) if ld else None,
        "H_mean_loadable": (sum(hs_ld) / len(hs_ld)) if hs_ld else None,
        "H_values_loadable": hs_ld,
        # FAILURE ACCOUNTING. Everything above silently drops any evolve that RAISED (those recs have
        # consolidated=None and are filtered out of `ok`), so k=5 evolves that all crashed produce the
        # same all-null payload as a well-formed run with nothing to report -- and the cell still
        # reports DONE. Observed on rq1_so_sonnet_ours: TASR_per_evolve=[] with K_valid=0 while
        # SkillOpt itself had burned 400k tokens. Surface the count and the exceptions so an
        # exception-zero can never be read as a behavioral zero (fourth instance of that conflation).
        "n_evolve_errors": sum(1 for r in recs if r.get("error")),
        "evolve_errors": [{"i": r["i"], "error": r["error"]} for r in sorted(recs, key=lambda r: r["i"])
                          if r.get("error")],
        "per_evolve": sorted(recs, key=lambda r: r["i"]),
    }
