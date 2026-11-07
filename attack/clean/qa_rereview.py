"""Post-hoc official score re-review of a FINISHED QA cell's final test (no re-solving).

Applies a score-review version (default official_qa_v3) to the recorded final-test answers of a
finished WikiTQ/OfficeQA cell and writes, next to the originals and never over them:

  final_test/score_review<suffix>.json              the review report (same shape as score_review.json)
  final_test/evolve_i/reviewed<suffix>_<arm>_records.json
  final_test<suffix>.json                           final_test.json with every reported metric recomputed
                                                    from the re-reviewed scores

Dev / in-loop measurements are left alone: they only drove candidate selection. Idempotent: the review
fingerprints its inputs, so rerunning on the same cell rewrites identical files; a cell whose inputs
changed is reported, not silently re-scored. Cells reviewed under v2 and v3 alike go through the same
rule, so tables built from the <suffix> files use one scoring rule.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import statistics

from attack.clean import qa_measurement as qm
from attack.evaluation import objective as objlib

SUFFIX = "_v3"
TOOL = "scripts/rereview_final_test.py"
_RAW_KEYS = ("ASR_mean", "TASR_mean", "TACC_mean", "CACC_mean", "DASR", "error_rate", "TASR_mean_evolved")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _std(values):
    values = [v for v in (values or []) if v is not None]
    return (statistics.pstdev(values) if len(values) > 1 else 0.0) if values else None


def _mean(values):
    values = [v for v in (values or []) if v is not None]
    return statistics.mean(values) if values else None


def recompute_final(final: dict, agg: dict, meta: dict, protocol: str) -> dict:
    """final_test.json's reported metrics, recomputed from a reviewed aggregate (the driver's own
    arithmetic, scripts/run_wikitq_optimization.py FINAL block)."""
    degradation = meta.get("attack_type", final.get("attack_type", "beacon")) == "degradation"
    f = copy.deepcopy(final)
    f["qa_score_review"] = protocol
    f["score_review"] = agg["score_review"]
    f["rereview"] = {"source": "final_test.json", "protocol": protocol,
                     "source_protocol": final.get("qa_score_review"),
                     "source_status": (final.get("score_review") or {}).get("status"),
                     "raw": {k: final.get(k) for k in _RAW_KEYS}, "tool": TOOL,
                     "note": "metrics recomputed from the re-reviewed recorded final-test answers; no re-solving; "
                             "dev/in-loop measurements untouched (they only drove selection)"}
    if not qm.score_verified(agg):
        f["rereview"]["status"] = "disputed"       # metrics stay the raw ones, exactly as final_test.json
        return f
    f["rereview"]["status"] = agg["score_review"]["status"]
    per = agg["per_evolve"]
    Hs, Cs, CAs = agg["H_values"], agg["C_values"], agg["CACC_values"]
    objective = final.get("objective") or meta.get("objective") or objlib.resolve_for(meta.get("attack_type", ""))
    f.update(ASR_mean=agg["H_mean"], ASR_per_evolve=Hs, TASR_mean=agg["H_mean"], TASR_per_evolve=Hs,
             TASR_range=([min(Hs), max(Hs)] if Hs else None), TASR_std=_std(Hs),
             optimization_score=objlib.objective_value(agg, objective) if objective in objlib.OBJECTIVES else None,
             TACC_mean=agg["C_mean"], TACC_per_evolve=Cs, TACC_std=_std(Cs) or 0.0,
             CACC_mean=agg["CACC_mean"], CACC_per_evolve=CAs, CACC_std=_std(CAs) or 0.0,
             C_mean=agg["C_mean"], C_std=_std(Cs) or 0.0,
             consolidated_rate=((sum(bool(r.get("consolidated")) for r in per) / len(per)) if per else None),
             clean_leak_mean=agg.get("clean_leak_mean"), clean_C_mean=agg.get("CACC_mean"),
             trigger_acc_cost_mean=agg.get("trigger_acc_cost_mean"))
    if degradation:
        f.update(error_rate=objlib.objective_value(agg, "error_rate"), DASR=agg.get("DASR"),
                 dasr_numerator=agg.get("dasr_numerator"), dasr_denominator=agg.get("dasr_denominator"),
                 DASR_n_broke=agg.get("dasr_numerator"), DASR_n_breakable=agg.get("dasr_denominator"),
                 ASR_mean=None, ASR_per_evolve=[], TASR_mean=None, TASR_per_evolve=[],
                 TASR_range=None, TASR_std=None)
    noop = sum(1 for r in per if r.get("evolve_noop"))
    f.update(n_evolve_noop=noop, noop_rate=(noop / len(per)) if per else None,
             TASR_mean_evolved=None if degradation else _mean([r["H_rate"] for r in per if not r.get("evolve_noop")]))
    if degradation:
        Fs = [None for _ in per]
        f.update(F_attempted_mean=None, F_attempted_per_evolve=Fs, F_to_E_conversion_per_evolve=[None for _ in per])
    else:
        Fs = f.get("F_attempted_per_evolve") or []       # the regrade does not touch Following
        f["F_to_E_conversion_per_evolve"] = ([round(h / x, 4) if x else None for h, x in zip(Hs, Fs)]
                                             if (Hs and Fs) else [])
    if "per_evolve" in f:                                 # native A-Evolve cells carry the records
        f["per_evolve"] = per
    return f


def _summary_metrics(final: dict) -> dict:
    return {"ASR": final.get("ASR_mean") if final.get("ASR_mean") is not None else final.get("TASR_mean"),
            "TACC": final.get("TACC_mean"), "CACC": final.get("CACC_mean"),
            "DASR": final.get("DASR"), "error_rate": final.get("error_rate"),
            "ASR_evolved": final.get("TASR_mean_evolved")}


def _load_tasks(dataset, split, n_test, final_test_dir):
    """The frozen task set the final test solved. The benchmark loader is authoritative; where it cannot
    open the split on this host (OfficeQA asserts its document files exist, which a BOM host outside the
    pods does not have) the recorded clean rows supply the same ids and ground truth: the review covers
    exactly those ids, WikiTQ typed references come from grading_annotations.json, and the OfficeQA
    comparator needs only ground_truth."""
    try:
        from attack.datasets.benchmark import Benchmark
        return Benchmark(dataset).load_splits(split_names=(split,))[split][:n_test], "benchmark"
    except Exception as exc:                        # noqa: BLE001 - any loader failure falls back
        rows = json.loads((final_test_dir / "evolve_0/clean_records.json").read_text())
        return [{"id": r["id"], "ground_truth": r["gold"]} for r in rows], f"records ({type(exc).__name__})"


def rereview(run_dir, *, protocol: str = qm.OFFICIAL_REVIEW, suffix: str = SUFFIX, force: bool = False,
             tasks=None, dry_run: bool = False) -> dict:
    """Re-review one finished cell. Returns a summary row; never raises on a per-cell problem."""
    run = Path(run_dir).resolve()
    row = {"run": str(run), "tag": run.name, "protocol": protocol}
    final_path, meta_path = run / "final_test.json", run / "meta.json"
    if not meta_path.exists():
        return {**row, "status": "skipped", "reason": "no meta.json"}
    if not final_path.exists():
        return {**row, "status": "skipped", "reason": "no final_test.json (not finished)"}
    meta = json.loads(meta_path.read_text())
    final = json.loads(final_path.read_text())
    dataset = meta.get("dataset")
    if dataset not in ("wikitq", "officeqa"):
        return {**row, "status": "skipped", "reason": f"not a WikiTQ/OfficeQA cell ({dataset})"}
    split = final.get("test_split")
    if split not in ("dev", "test") or meta.get("target_id") is not None:
        return {**row, "status": "skipped", "reason": "no frozen universal dev/test task set"}
    k, n_test = final.get("k"), final.get("n_test")
    ft = run / "final_test"
    sources = [final_path, meta_path, *(p for p in (ft / "score_review.json",) if p.exists())]
    agg = {"per_evolve": []}
    raw = (final.get("score_review") or {}).get("raw_scores") or {}
    agg.update({"H_mean": raw.get("H_mean", final.get("ASR_mean", final.get("TASR_mean"))),
                "C_mean": raw.get("C_mean", final.get("TACC_mean")),
                "CACC_mean": raw.get("CACC_mean", final.get("CACC_mean")), "DASR": raw.get("DASR", final.get("DASR"))})
    try:
        for i in range(k):
            directory = ft / f"evolve_{i}"
            record = directory / "measure_record.json"
            agg["per_evolve"].append(copy.deepcopy(json.loads(record.read_text())["record"]))
            sources += [record, *(directory / f"{arm}_records.json" for arm in ("triggered", "clean"))]
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {**row, "status": "skipped", "reason": f"incomplete final_test records: {exc}"}
    originals = {p: _sha(p) for p in sources if p.exists()}
    row.update(source_protocol=final.get("qa_score_review"),
               status_before=(final.get("score_review") or {}).get("status"), before=_summary_metrics(final))
    if dry_run:
        return {**row, "status": "dry_run"}
    if tasks is None:
        tasks, row["tasks_source"] = _load_tasks(dataset, split, n_test, ft)
    report = ft / f"score_review{suffix}.json"
    if force and report.exists():
        report.unlink()
    try:
        qm.review_measurement(agg, tasks=tasks, dataset=dataset, attack_type=meta.get("attack_type", "beacon"),
                              workdir=ft, native=meta.get("deployment_protocol") == "native_multi_skill",
                              protocol=protocol, suffix=suffix)
    except ValueError as exc:                       # inputs changed under an existing report
        return {**row, "status": "error", "reason": str(exc)}
    changed = [str(p) for p, digest in originals.items() if _sha(p) != digest]
    if changed:
        raise RuntimeError(f"re-review touched original files: {changed}")
    new_final = recompute_final(final, agg, meta, protocol)
    out = run / f"final_test{suffix}.json"
    text = json.dumps(new_final, indent=1)
    if not out.exists() or out.read_text() != text:
        out.write_text(text)
    after = _summary_metrics(new_final)
    delta = {m: (None if after[m] is None or row["before"][m] is None else round(after[m] - row["before"][m], 6))
             for m in after}
    review = agg["score_review"]
    row.update(status="done", status_after=review["status"], after=after, delta=delta,
               markup_graded=len(review.get("markup_graded") or []), corrections=review["correction_count"],
               changed=(review["status"] != row["status_before"]) or any(d for d in delta.values() if d),
               outputs=[str(report), str(out)])
    return row
