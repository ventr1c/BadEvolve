#!/usr/bin/env python3
"""Re-review finished QA cells' final tests under official_qa_v3 (or --protocol), writing
final_test/score_review_v3.json and final_test_v3.json next to the originals. Idempotent; never
overwrites originals; dev/in-loop measurements untouched. See attack/clean/qa_rereview.py.

  .venv-attack/bin/python scripts/rereview_final_test.py runs/*__pw_* --summary runs/adhoc/rereview_v3.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack.clean import qa_measurement as qm  # noqa: E402
from attack.clean import qa_rereview  # noqa: E402

METRICS = ("ASR", "TACC", "CACC", "DASR", "error_rate")


def _fmt(v):
    return "   -  " if v is None else f"{v:+.3f}" if isinstance(v, float) and v < 0 else f"{v: .3f}" if v is not None else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run dirs (shell glob, e.g. runs/*)")
    ap.add_argument("--protocol", default=qm.OFFICIAL_REVIEW, choices=list(qm.OFFICIAL_REVIEWS))
    ap.add_argument("--suffix", default=qa_rereview.SUFFIX)
    ap.add_argument("--force", action="store_true", help="drop an existing <suffix> report first")
    ap.add_argument("--dry-run", action="store_true", help="list what would be re-reviewed; write nothing")
    ap.add_argument("--summary", type=Path, help="write the per-cell rows + totals as JSON")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    rows, cache = [], {}
    for run in a.runs:
        try:
            row = qa_rereview.rereview(run, protocol=a.protocol, suffix=a.suffix, force=a.force, dry_run=a.dry_run)
        except Exception as exc:                    # noqa: BLE001 - one cell never stops the sweep
            row = {"run": str(run), "tag": Path(run).name, "protocol": a.protocol, "status": "error",
                   "reason": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if a.quiet:
            continue
        if row["status"] != "done":
            print(f"{row['status']:8s} {row['tag']}: {row.get('reason', '')}", flush=True)
            continue
        deltas = " ".join(f"{m}{_fmt(row['delta'].get(m))}" for m in METRICS if row["before"].get(m) is not None
                          or row["after"].get(m) is not None)
        print(f"{'CHANGED ' if row['changed'] else 'same    '}{row['tag']}: {row['source_protocol']}/"
              f"{row['status_before']} -> {row['protocol']}/{row['status_after']} markup={row['markup_graded']} "
              f"{deltas}", flush=True)
    done = [r for r in rows if r["status"] == "done"]
    changed = [r for r in done if r["changed"]]
    totals = {"cells": len(rows), "done": len(done), "changed": len(changed),
              "skipped": sum(r["status"] == "skipped" for r in rows), "errors": sum(r["status"] == "error" for r in rows),
              "disputed_before": sum(r.get("status_before") == "disputed" for r in done),
              "disputed_after": sum(r.get("status_after") == "disputed" for r in done),
              "markup_graded_rows": sum(r.get("markup_graded", 0) for r in done)}
    for m in METRICS:
        ds = [r["delta"][m] for r in done if r["delta"].get(m) is not None]
        nz = [d for d in ds if d]
        totals[m] = {"n": len(ds), "n_changed": len(nz), "mean_delta": (sum(ds) / len(ds)) if ds else None,
                     "max_abs_delta": max((abs(d) for d in ds), default=None)}
    print(json.dumps(totals, indent=1))
    if a.summary:
        a.summary.parent.mkdir(parents=True, exist_ok=True)
        a.summary.write_text(json.dumps({"protocol": a.protocol, "suffix": a.suffix, "totals": totals, "rows": rows},
                                        indent=1))
    return 1 if totals["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
