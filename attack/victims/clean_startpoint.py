"""Explicit, immutable clean checkpoints for two-stage attack experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

from attack.clean.common import digest, write_json
from attack.datasets.benchmark import Benchmark

TEXT = {".md", ".txt", ".json", ".yaml", ".yml", ".py"}


def files(root):
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(
        p.read_bytes().replace(str(root).encode(), b"<CHECKPOINT>")).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts}


def load(source_run, *, dataset, framework, model, seed, evolver=None):
    if not source_run:
        return None
    root = Path(source_run).resolve()
    meta = json.loads((root / "meta.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    status = json.loads((root / "status.json").read_text())
    cfg = meta["config"]
    expected = {"dataset": dataset, "method": framework, "model": model, "seed": seed}
    identity = {k: cfg.get(k, summary.get(k)) for k in expected}
    if identity != expected or status["state"] != "done":
        raise ValueError("Clean checkpoint identity/completion does not match this attack")
    split_hashes = cfg.get("split_hashes", cfg.get("splits_sha256"))
    if split_hashes != Benchmark(dataset, seed).split_hashes():
        raise ValueError("Clean checkpoint uses different frozen splits")
    if dataset == "spreadsheetbench" and framework == "trace2skill":
        if cfg.get("t2s_evolver", "error") != evolver:
            raise ValueError("Clean checkpoint uses a different Trace2Skill evolution variant")
    native = framework == "aevolve"
    artifact = root / "workdir/aevolve/workspace" if native else Path(summary["skill_dir"])
    if not artifact.is_dir() or not list(artifact.rglob("SKILL.md")):
        raise ValueError("Clean checkpoint has no deployable skill")
    return {**expected, "protocol": "clean_then_poison_v1",
            "source_run": str(root), "artifact_path": str(artifact.resolve()),
            "artifact_kind": "native_workspace" if native else "skill_directory",
            "artifact_sha256": digest(files(artifact)),
            "split_hashes": split_hashes,
            "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
            "source_config": cfg}


# Trace2Skill's backup_skill() leaves its pre-evolution copy `<skill>_backup_<YYYYmmdd_HHMMSS>` beside the
# evolved skill. It is not part of the deployed skill, and Trace2Skill's own skill validator rejects it.
EVOLVER_BACKUP = re.compile(r".+_backup_\d{8}_\d{6}")


def clone(checkpoint, destination, receipt=None):
    """One private starting copy per candidate/repetition; preserve evolved copies on resume.

    `receipt` defaults to `<destination>_clean_startpoint.json` beside it; a QA framework under POISON_NEUTRAL_NAMES
    keeps it outside its neutral workspace, since it names the clean run."""
    destination = Path(destination)
    source = Path(checkpoint["artifact_path"])
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Attack attempts cannot be staged inside the cached checkpoint")
    receipt = Path(receipt) if receipt else destination.parent / (destination.name + "_clean_startpoint.json")
    if digest(files(source)) != checkpoint["artifact_sha256"]:
        raise ValueError("Cached clean checkpoint changed")
    if receipt.exists():
        if json.loads(receipt.read_text()) != checkpoint or not destination.is_dir():
            raise ValueError("Attempt resumed with a different or missing clean checkpoint")
        return destination
    if destination.exists():
        raise ValueError("Cannot overwrite an unrecorded starting skill")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint["artifact_kind"] == "native_workspace":
        from attack.victims.aevolve_sb_agent import clone_deployment
        clone_deployment(source, destination)
    else:
        shutil.copytree(source, destination, ignore=lambda folder, names: [
            n for n in names if n == "__pycache__" or (Path(folder) == source and EVOLVER_BACKUP.fullmatch(n))])
        for p in destination.rglob("*"):
            if p.is_file() and p.suffix in TEXT:
                p.write_bytes(p.read_bytes().replace(str(source).encode(), str(destination).encode()))
    write_json(receipt, checkpoint)
    return destination


def attach(victim, checkpoint):
    if checkpoint:
        if victim.model != checkpoint["model"]:
            raise ValueError("A clean checkpoint cannot be reused across victim backbones")
        victim.clean_startpoint = checkpoint
    return victim


def stage(victim, destination, receipt=None):
    checkpoint = getattr(victim, "clean_startpoint", None)
    return clone(checkpoint, destination, receipt) if checkpoint else None
