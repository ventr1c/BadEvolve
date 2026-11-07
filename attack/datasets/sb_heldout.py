"""SpreadsheetBench held-out task records (dev / test) for the SB driver.

Moved verbatim from the lifecycle-funnel script, which is not part of this release; only the dataset root
is now resolved from this repository (attack/datasets/paths.py) instead of an absolute checkout path.
"""
from __future__ import annotations

import json
from pathlib import Path

from attack.datasets.paths import get_dataset_paths

REPO = Path(__file__).resolve().parents[2]
DATASET = get_dataset_paths("spreadsheetbench", repo=REPO).data_root / "dataset.json"


def resolve_xlsx(task_dir: Path, tid, kind: str) -> Path | None:
    """SpreadsheetBench Verified-400 ships TWO naming conventions: most instances
    use `1_<id>_init.xlsx` / `1_<id>_golden.xlsx`, but 5 of 400 use bare
    `initial.xlsx` / `golden.xlsx`. Hardcoding the first killed a whole run on a
    single task. Returns None if neither exists."""
    names = ([f"1_{tid}_init.xlsx", "initial.xlsx"] if kind == "init"
             else [f"1_{tid}_golden.xlsx", "golden.xlsx"])
    for n in names:
        p = task_dir / n
        if p.exists():
            return p
    return None


def heldout_meta(ids, n_trig, n_clean):
    rows = {r["id"]: r for r in json.load(open(DATASET))}
    root = DATASET.parent
    out = []
    skipped = []
    for i, tid in enumerate(ids[: n_trig + n_clean]):
        r = rows[tid]; sp = root / r["spreadsheet_path"]
        init_p, gold_p = resolve_xlsx(sp, tid, "init"), resolve_xlsx(sp, tid, "golden")
        if init_p is None or gold_p is None:
            skipped.append(tid); continue
        out.append({
            "id": tid, "instruction": r["instruction"],
            "xlsx_path": str(init_p),
            "answer_xlsx": str(gold_p),
            "answer_position": r["answer_position"], "answer_sheet": r.get("answer_sheet"),
            "triggered": i < n_trig, "token": f"AUD-{tid}",
        })
    if skipped:
        print(f"[heldout] WARNING skipped {len(skipped)} task(s) with unresolvable "
              f"workbooks: {skipped}", flush=True)
    return out
