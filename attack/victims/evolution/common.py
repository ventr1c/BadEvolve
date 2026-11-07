"""Shared evolution trajectories and historical OfficeQA method entry points.

The default dataset remains OfficeQA for callers that omit a Benchmark. Native
method implementations live in their named modules; dispatch lives in registry.
"""
from __future__ import annotations

import json
from pathlib import Path

from attack.clean.common import write_json
from attack.victims.officeqa_clean import DATA, REPO, SO, load_splits, run_batch


def evolve(args, run_dir: Path, splits: dict, skills: Path, options: dict, benchmark=None,
           pool=None) -> Path:
    """Compatibility entry point for the canonical method registry."""
    from attack.victims.evolution.registry import evolve as dispatch
    return dispatch(args, run_dir, splits, skills, options, benchmark, pool=pool)


def _skillopt(args, run_dir: Path, splits: dict, skills: Path, options: dict, benchmark=None) -> Path:
    """Preserve the historical patchable SkillOpt entry point."""
    from attack.victims.evolution.skillopt import evolve_skillopt
    return evolve_skillopt(args, run_dir, splits, skills, options, benchmark)


def initial_trajectories(args, benchmark=None, pool=None) -> tuple[Path, list[dict]]:
    """Trajectory pool the native evolution cores consolidate from.

    `pool` is the ATTACK-side seam: `(output_dir, rows)` already materialized from the injected
    poison tasks. The clean controls never pass it, so `None` reproduces today's behavior exactly
    and every clean cell is unaffected.

    It exists because the checks below are correct for the clean arm and WRONG for the attack arm:
    they demand a finished clean seed run whose rows cover the whole `train` split. Under attack the
    pool IS the poison set (poison-in-train-only holds by construction, since no other split is ever
    requested), so those assertions would reject a valid attack pool.
    """
    if pool is not None:
        return pool
    tag = args.seed_run_tag or args.tag
    dataset = benchmark.name if benchmark is not None else "officeqa"
    root = REPO / "runs" / f"clean__seed__{dataset}__cacc__{tag}"
    if not (root / "result.json").exists():
        raise RuntimeError(f"Initial-skill train run must finish first: {root}")
    meta = json.loads((root / "meta.json").read_text())["config"]
    for name in ("model", "endpoint", "seed", "tool_turns", "max_tokens"):
        if meta[name] != getattr(args, name):
            raise ValueError(f"Initial trajectories use a different {name}")
    output = root / "workdir/train"
    rows = json.loads((output / "records.json").read_text())
    splits = benchmark.load_splits() if benchmark is not None else load_splits()
    source_limit = meta.get("limit", 0)
    if source_limit and (not getattr(args, "limit", 0) or getattr(args, "eval_split", "test") != "dev"):
        raise ValueError("A reported run cannot use a partial smoke training pool")
    expected = {str(t["id"]) for t in splits["train"][:source_limit or None]}
    if (len(rows) != len(expected) or {r["id"] for r in rows} != expected
            or any(r["transport_error"] for r in rows)):
        raise ValueError(f"Initial trajectories must cover all {len(expected)} train tasks without transport errors")
    if dataset == "spreadsheetbench":
        from attack.clean.spreadsheet import native_transport_failure
        if any(native_transport_failure(json.loads((output / "predictions" / r["id"] / "native_result.json").read_text()))
               for r in rows):
            raise ValueError("Initial SpreadsheetBench trajectories contain masked transport failures")
    return output, rows
