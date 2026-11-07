"""Structured experiment log — the single source of truth for every run.

Each run directory is self-describing, so the monitor never greps stage logs or
guesses paths (that does not scale). Three files per run:

  meta.json     written once at launch: the full config + git sha + a reproduce
                command. Answers "what is this run and how do I rerun it".
  status.json   updated at every stage: {stage, state, elapsed, updated_at, error}.
                Answers "where is it now, is it alive, did it fail". `updated_at`
                doubles as a heartbeat (stale => stuck).
  result.json   written at the end: the funnel row (I/R/F/E/CACC/TACC/TASR).

Adding a victim, stage, or attack does not touch the monitor: it reads this schema.
Stage order is fixed so progress is a position along STAGES.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

STAGES = ["inject", "evolve", "heldout", "score"]   # coarse-grained lifecycle stages


def _write(path, text: str) -> None:
    """Atomic replace through a writer-unique temp file: readers see the old or the new file, never a mix. Several
    processes on several hosts rewrite one matrix status.json; a plain write_text interleaved on 2026-10-20 03:22Z
    and every v5 controller died with JSONDecodeError on its next read."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _now() -> float:
    return time.time()


def init_run(run_dir: str, meta: dict) -> None:
    """Write meta.json once at launch. `meta` should carry attack/victim/type/
    target/sizes/seed/model/script/git_sha and a `reproduce` command string."""
    p = Path(run_dir); p.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "created_at": _now()}
    _write(p / "meta.json", json.dumps(meta, indent=1))
    set_status(run_dir, stage="inject", state="running", note="launched")


def set_status(run_dir: str, *, stage: str, state: str, note: str = "",
               error: str = "", extra: dict | None = None) -> None:
    """state in {running, done, failed}. Called once per stage transition."""
    p = Path(run_dir) / "status.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    started = prev.get("started_at", _now())
    st = {
        "stage": stage, "state": state, "note": note, "error": error,
        "stage_idx": STAGES.index(stage) if stage in STAGES else -1,
        "n_stages": len(STAGES),
        "started_at": started, "updated_at": _now(),
        "elapsed_s": round(_now() - started, 1),
        **(extra or {}),
    }
    _write(p, json.dumps(st, indent=1))


def set_progress(run_dir: str, **fields) -> None:
    """Merge progress fields into status.json without touching stage/state.

    Lets a long stage report what it is actually doing (which evolution sub-stage,
    how many held-out tasks are done) so the monitor stays structured-state-only."""
    p = Path(run_dir) / "status.json"
    if not p.exists():
        return
    st = json.loads(p.read_text())
    st.update(fields)
    st["updated_at"] = _now()
    st["elapsed_s"] = round(_now() - st.get("started_at", _now()), 1)
    _write(p, json.dumps(st, indent=1))


def beat(run_dir: str) -> None:
    """Refresh updated_at/elapsed_s without changing stage/state.

    set_status only fires at stage transitions, so during a long evolve the
    heartbeat froze and the monitor could not tell 'slow' from 'stuck'."""
    p = Path(run_dir) / "status.json"
    if not p.exists():
        return
    st = json.loads(p.read_text())
    if st.get("state") != "running":
        return
    st["updated_at"] = _now()
    st["elapsed_s"] = round(_now() - st.get("started_at", _now()), 1)
    _write(p, json.dumps(st, indent=1))


def finish_run(run_dir: str, row: dict) -> None:
    _write(Path(run_dir) / "result.json", json.dumps({"row": row}, indent=1))
    set_status(run_dir, stage="score", state="done", note="complete")


def fail_run(run_dir: str, stage: str, error: str) -> None:
    set_status(run_dir, stage=stage, state="failed", error=error[:500])


def snapshot_sha(repo) -> str | None:
    """git_sha recorded by a code snapshot at `repo` (a frozen code snapshot).

    A snapshot has no .git of its own. A matrix snapshot lives under runs/ INSIDE the main checkout, so
    `git -C <snapshot> rev-parse HEAD` walks up and returns the MAIN tree's HEAD at launch: v5 cells recorded
    67d2c69 although their code is the bae2040 archive. A host-local snapshot has no repository above it and
    recorded "unknown". The snapshot's own record is the truth for both."""
    try:
        import json as _json
        from pathlib import Path as _Path
        return _json.loads((_Path(repo) / "snapshot.json").read_text()).get("git_sha") or None
    except (OSError, ValueError):
        return None


def git_sha(repo: str) -> str:
    snap = snapshot_sha(repo)
    if snap:
        return snap[:7]
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def runtime_provenance(repo: str) -> dict:
    """Which interpreter and which dependency locks ACTUALLY ran this process.

    meta.json records intent; nothing recorded the runtime, and on 2026-10-07 the anaconda
    python3.13 every earlier cell ran on was deleted. The rebuild is /usr/local python3.12 under
    uv locks (scripts/requirements_{clean,attack}.lock), so a reader comparing a cell across that
    boundary needs the interpreter and lock hashes beside the numbers, not inferred from dates."""
    import hashlib
    import sys
    locks = {}
    for name in ("requirements_clean.lock", "requirements_attack.lock"):
        p = Path(repo) / "scripts" / name
        if p.is_file():
            locks[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"python_executable": sys.executable, "python_version": sys.version.split()[0],
            "sys_prefix": sys.prefix, "in_venv": sys.prefix != sys.base_prefix,
            "lock_sha256": locks}


def read(run_dir: str) -> dict:
    """Everything the monitor needs, in one call: meta + status + result."""
    p = Path(run_dir)
    def _load(f):
        fp = p / f
        return json.loads(fp.read_text()) if fp.exists() else {}
    return {"meta": _load("meta.json"), "status": _load("status.json"),
            "result": _load("result.json")}
