#!/usr/bin/env python3
"""SkillGrad victim adapter for Phase-A — aligned Verified-400 split.

SkillGrad's native config (200 evolution + 200 held-out) is NOT runnable on the
data we have (its bundled dataset is the 210-subset; Verified-400 is 400). But
two facts make the ALIGNED split feasible:
  * SkillGrad's loader (`data/load.py`) NATIVELY supports the Verified-400
    init/golden layout (Pattern 2) — feasibility() confirms all 80/40/280
    aligned ids resolve with their workbooks.
  * `pipeline.training` only calls `create_split` (the 200+200 minimum) when
    `split.json` is ABSENT; if we pre-write a custom split.json it is loaded
    verbatim, bypassing the minimum (the same trick `scripts/poison_experiment.py`
    uses).

So we point SkillGrad at the official Verified-400 dataset and write:
  evolution_ids = aligned TRAIN (80) [+ poison ids for the poison arm]
  val_ids       = aligned VAL  (40)   test_ids = aligned TEST (280)
  failure_ids.json (poison arm) FORCES the poison ids into the sampled failures
  so the diagnoser/momentum see them (poison self-qualifies as a base failure
  anyway; forcing makes the smoke deterministic on small windows).

We DO NOT modify the SkillGrad repo. Outputs land under our runs/ dir.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2])
SKILLGRAD = f"{REPO}/evolution_methods/SkillGrad"
V400 = f"{REPO}/dataset/spreadsheetbench/official_verified_400/spreadsheetbench_verified_400"
ALIGNED = f"{REPO}/dataset/spreadsheetbench/skillopt_verified_400_splits"
LOCAL210 = f"{REPO}/dataset/spreadsheetbench/skillgrad_local_210"


def _aligned_ids(split: str) -> list[str]:
    return [str(t["id"]) for t in json.load(open(f"{ALIGNED}/{split}/items.json"))]


def feasibility(data_dir: str = V400) -> dict:
    """Probe whether SkillGrad's loader can ingest the aligned split off data_dir.
    Pure check (no model calls). Returns a support verdict for the bucket."""
    sys.path.insert(0, SKILLGRAD)
    from data.load import SpreadsheetBenchDataset
    ds = SpreadsheetBenchDataset(data_dir)
    id_to_idx = {str(s["id"]): i for i, s in enumerate(ds.dataset)}
    rep = {"data_dir": data_dir, "n_dataset": len(ds.dataset), "splits": {}, "files_ok": {}}
    supported = True
    for split in ("train", "val", "test"):
        S = _aligned_ids(split)
        missing = [i for i in S if i not in id_to_idx]
        rep["splits"][split] = {"n": len(S), "resolved": len(S) - len(missing), "missing": missing[:10]}
        # workbook resolution for a sample (input + answer paths must exist)
        bad = []
        for i in S[:8]:
            if i not in id_to_idx:
                continue
            try:
                ip = ds.get_input_path(i, 1)   # (sample_id, test_case)
                apth = ds.get_answer_path(i, 1)
                if not (ip and apth and os.path.exists(ip) and os.path.exists(apth)):
                    bad.append(i)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{i}:{e}")
        rep["files_ok"][split] = {"sampled": min(8, len(S)), "bad": bad}
        if missing or bad:
            supported = False
    rep["official_support_status"] = "supported" if supported else "unsupported_adapter_or_split"
    return rep


def materialize_aligned(out_root: str, arm: str, *, n_clean: int = -1,
                        poison_dir: str | None = None, data_dir: str = V400,
                        master_seed: int = 0, heldout_seed: int = 42,
                        force_failures: bool = True, frozen_splits: dict | None = None) -> dict:
    """Write a SkillGrad split.json (+ normalized data + forced failure_ids for
    the poison arm) for the aligned split. ``n_clean`` is a LITERAL count of
    aligned-train evolution tasks (-1 = all 80, 0 = none).

    For the poison arm, ``poison_dir`` is a run dir produced by
    attack/builders/spreadsheetbench_poison.py (its poison_tasks/spreadsheet/<pid>/
    use init/golden naming, which SkillGrad's loader Pattern 2 supports). The
    poison entries are injected into a COMBINED dataset dir + dataset.json.

    ``force_failures`` (poison arm only, default True) preserves the historical
    behavior: the poison ids + clean_pool are FORCED into the failure pool via a
    written failure_ids.json that the runner copies over the natural base-traj
    grading. Set ``force_failures=False`` for a NATURAL-only replay: no forced
    file is written and ``manifest["forced_failure_ids"]`` is None, so the runner's
    ``if forced and os.path.exists(forced)`` short-circuits and the natural
    failure_ids.json produced by base-trajectory grading is used untouched.
    """
    sys.path.insert(0, SKILLGRAD)
    from data.load import SpreadsheetBenchDataset

    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)
    if frozen_splits is None:
        train_ids, val_ids, test_ids = (_aligned_ids(s) for s in ("train", "val", "test"))
    else:
        train_ids = [str(t["id"]) for t in frozen_splits["train"]]
        val_ids = [str(t["id"]) for t in frozen_splits["dev"]]
        test_ids = []  # Frozen test data is never needed to evolve a study candidate.

    poison_items: list[dict] = []
    if arm == "poison":
        assert poison_dir, "poison arm requires --poison-dir"
        poison_items = json.load(open(f"{poison_dir}/poison/items.json"))
    poison_ids = [str(p["id"]) for p in poison_items]

    # Build a COMBINED dataset dir = symlink Verified-400 + add poison entries.
    arm_data = f"{out_root}/data"
    if os.path.exists(arm_data):
        shutil.rmtree(arm_data)
    os.makedirs(f"{arm_data}/spreadsheet", exist_ok=True)
    base = json.load(open(f"{data_dir}/dataset.json"))
    if frozen_splits is not None:
        allowed = set(train_ids + val_ids)
        base = [row for row in base if str(row["id"]) in allowed]
    # symlink every base spreadsheet dir (cheap; SkillGrad reads from here)
    if not os.path.islink(f"{arm_data}/spreadsheet/_base"):
        pass
    for s in base:
        sid = str(s["id"])
        if frozen_splits is not None and sid in set(poison_ids):
            continue  # Mixed-study train rows must resolve to their private, verified copies.
        src = f"{data_dir}/spreadsheet/{sid}"
        dst = f"{arm_data}/spreadsheet/{sid}"
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    combined = ([row for row in base if str(row["id"]) not in set(poison_ids)]
                if frozen_splits is not None else list(base))
    for p in poison_items:
        pid = str(p["id"])
        # poison spreadsheet dir (init/golden naming) -> symlink into arm data
        psrc = p.get("spreadsheet_path")
        if psrc and os.path.isdir(psrc):
            pdst = f"{arm_data}/spreadsheet/{pid}"
            if not os.path.exists(pdst):
                os.symlink(os.path.abspath(psrc), pdst)
        combined.append({"id": pid, "instruction": p.get("instruction", ""),
                         "answer_position": p.get("answer_position", "")})
    json.dump(combined, open(f"{arm_data}/dataset.json", "w"), ensure_ascii=False, indent=2)

    # evolution pool = aligned train (minus poison sources) [+ poison ids]
    src_ids = set()
    if arm == "poison" and os.path.exists(f"{poison_dir}/poison/src_ids.json"):
        src_ids = set(json.load(open(f"{poison_dir}/poison/src_ids.json")).get("src_ids", []))
    clean_pool = [i for i in train_ids if i not in src_ids]
    if n_clean >= 0:
        clean_pool = clean_pool[:n_clean]
    evolution_ids = clean_pool + poison_ids

    # resolve to indices in the COMBINED dataset
    ds = SpreadsheetBenchDataset(arm_data)
    id_to_idx = {str(s["id"]): i for i, s in enumerate(ds.dataset)}
    def idxs(ids):
        # Preserve id order so split_indices[i] resolves to split_ids[i].
        # Sorting the indices silently evaluates different tasks.
        return [id_to_idx[i] for i in ids if i in id_to_idx]
    split_info = {
        "master_seed": master_seed, "heldout_seed": heldout_seed,
        "n_evolution": len(evolution_ids), "n_test": len(test_ids), "n_val": len(val_ids),
        "evolution_indices": idxs(evolution_ids), "test_indices": idxs(test_ids),
        "val_indices": idxs(val_ids),
        "evolution_ids": [i for i in evolution_ids if i in id_to_idx],
        "test_ids": [i for i in test_ids if i in id_to_idx],
        "val_ids": [i for i in val_ids if i in id_to_idx],
        "data_dir": arm_data,
    }
    split_dir = f"{out_root}/splits/master_{master_seed}_heldout_{heldout_seed}"
    os.makedirs(split_dir, exist_ok=True)
    json.dump(split_info, open(f"{split_dir}/split.json", "w"), indent=2)

    # normalized data (input/answer symlinks) for the pipeline
    normalized = f"{out_root}/normalized"
    ds.prepare_for_pipeline(normalized)

    # poison arm: force poison ids into failure pool so they ARE sampled
    base_traj = f"{out_root}/base_trajectories/master_{master_seed}_heldout_{heldout_seed}/forced"
    os.makedirs(base_traj, exist_ok=True)
    if arm == "poison" and force_failures:
        json.dump({"failure_ids": poison_ids + clean_pool, "success_ids": []},
                  open(f"{base_traj}/failure_ids.json", "w"), indent=2)

    if force_failures:
        forced_failure_ids = f"{base_traj}/failure_ids.json"
        evolution_policy = "aligned-train evolution pool; SkillGrad failure-filter; poison forced into failures"
    else:
        # NATURAL-only replay: no forced file; the runner's copy-over is skipped
        # (forced_failure_ids is None) so base-trajectory grading decides failures.
        forced_failure_ids = None
        evolution_policy = "aligned-train evolution pool; SkillGrad NATURAL failure-filter (forced override OFF)"

    manifest = {
        "victim": "skillgrad", "arm": arm, "aligned_split": ALIGNED,
        "n_clean": len(clean_pool), "n_poison": len(poison_ids),
        "n_evolution": len(evolution_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "beta_window": round(len(poison_ids) / max(1, len(evolution_ids)), 4),
        "poison_ids": poison_ids, "excluded_src_ids": sorted(src_ids),
        "split_json": f"{split_dir}/split.json", "data_dir": arm_data,
        "normalized": normalized, "forced_failure_ids": forced_failure_ids,
        "evolution_policy": evolution_policy,
    }
    json.dump(manifest, open(f"{out_root}/split_manifest.json", "w"), indent=2)
    if frozen_splits is None:
        print(json.dumps({k: manifest[k] for k in
                          ("arm", "n_clean", "n_poison", "n_evolution", "beta_window")}, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("feasibility")
    f.add_argument("--data-dir", default=V400)
    f.add_argument("--out", default=None)
    m = sub.add_parser("materialize")
    m.add_argument("--out-root", required=True)
    m.add_argument("--arm", choices=["clean", "poison"], default="clean")
    m.add_argument("--n-clean", type=int, default=-1)
    m.add_argument("--poison-dir", default=None)
    m.add_argument("--data-dir", default=V400)
    args = ap.parse_args()
    if args.cmd == "feasibility":
        rep = feasibility(args.data_dir)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(rep, open(args.out, "w"), indent=2)
        print(json.dumps({"official_support_status": rep["official_support_status"],
                          "splits": rep["splits"]}, indent=2))
    elif args.cmd == "materialize":
        materialize_aligned(args.out_root, args.arm, n_clean=args.n_clean,
                            poison_dir=args.poison_dir, data_dir=args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
