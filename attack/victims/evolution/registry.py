"""Dispatch the shared single-package evolution contract by method name.

Native A-Evolve uses catalog/read_skill loading and complete skill directories.
Its separate runtime is intentionally not part of this xlsx-package contract.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from attack.victims.evolution import common

# module, entry point, injected trajectory keyword. SkillOpt's compatibility
# entry point keeps historical monkeypatches while calling its sole implementation.
METHODS = {
    "trace2skill": ("trace2skill", "evolve_trace2skill", "traj_pool"),
    "skillgrad": ("skillgrad", "evolve_skillgrad", "pool"),
    "skillopt": ("common", "_skillopt", None),
}


def evolve(args, run_dir: Path, splits: dict, skills: Path, options: dict, benchmark=None,
           pool=None) -> Path:
    done = run_dir / "workdir/evolution_done.json"
    if done.exists():
        return Path(json.loads(done.read_text())["skills"])
    try:
        module_name, entry_point, pool_keyword = METHODS[args.method]
    except KeyError:
        raise ValueError(args.method) from None
    module = importlib.import_module(f"{__package__}.{module_name}")
    extra = {pool_keyword: pool} if pool_keyword else {}
    final = getattr(module, entry_point)(args, run_dir, splits, skills, options, benchmark, **extra)
    if not (final / "xlsx/SKILL.md").read_text().strip():
        raise RuntimeError("Evolution produced an empty skill")
    if benchmark is not None:
        benchmark.finalize_skill(final)
    common.write_json(done, {"method": args.method, "skills": str(final)})
    return final
