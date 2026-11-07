"""Dataset-independent utilities for clean execution and its compatibility adapters."""
from __future__ import annotations

import hashlib
import os
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from attack.evaluation import runlog

# Rate-limit retries for every OpenAI-SDK client WE build against the model gateway.
# Layers stack -- each client retry re-triggers the gateway's own retries (~65 s of backoff) -- so 10 already
# keeps one request trying for ~11 min, longer than the provider throttle episodes we observed (~7 min).
# SDK default is 2. Upstream clients we do not build (SkillOpt qwen_chat, Trace2Skill SB solver) are NOT covered.
RATE_LIMIT_RETRIES = 10

_STATUS_LOCK = threading.Lock()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def skill_bundle(directory: Path | None, text: str | None = None) -> tuple[str, dict[str, str]]:
    if directory is None:
        return text or "", {}
    root = directory / "xlsx"
    text_suffixes = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".csv", ".tsv"}
    files = {str(p.resolve()): p.read_text() for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in text_suffixes}
    return files.get(str((root / "SKILL.md").resolve()), ""), files


def progress(run_dir: Path | None, **fields) -> None:
    if run_dir:
        with _STATUS_LOCK:
            runlog.set_progress(str(run_dir), **fields)


# A task whose model call died of a transport error (429 after every retry, dropped connection) is redone by the batch itself, up
# to TASK_RETRIES more times, instead of failing the whole batch -- and with it a SkillOpt evolve or a clean-arm split. `solve_fn`
# already treats a cached row with transport_error as "not done" (attack/clean/wikitq.py:126, officeqa_clean.py:152), so a redo
# touches only the failed tasks and every task that succeeded is reused as is. 3 of 20 OfficeQA SkillOpt cells lost 27-32 min to this
# on 2026-10-14/15 and 3 more died of it. A task that still fails after the retries raises exactly as before.
TASK_RETRIES = 2


def run_batch(items: list[dict], skills: Path | None, output: Path, *, model: str,
              endpoint: str, workers: int = 4, tool_turns: int = 24,
              max_tokens: int = 16384, skill_text: str | None = None,
              run_dir: Path | None = None, solve_fn, bundle_fn=None) -> list[dict]:
    text, files = (bundle_fn or skill_bundle)(skills, skill_text)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(solve_fn, t, text, files, output / "predictions" / str(t["id"]),
                               model=model, endpoint=endpoint, tool_turns=tool_turns,
                               max_tokens=max_tokens): t for t in items}
        for future in as_completed(futures):
            row = future.result()
            results[row["id"]] = row
            n_ok = sum(r["hard"] for r in results.values())
            progress(run_dir, substage=str(output.name), completed_tasks=len(results),
                     total_tasks=len(items), correct=n_ok)
            if len(results) % 10 == 0 or len(results) == len(items):
                print(f"[{output.name}] {len(results)}/{len(items)} correct={n_ok}", flush=True)
    for redo in range(TASK_RETRIES):
        bad = [t for t in items if results[str(t["id"])]["transport_error"]]
        if not bad:
            break
        print(f"[{output.name}] {len(bad)} task(s) hit a transport error; redoing them ({redo + 1}/{TASK_RETRIES})", flush=True)
        time.sleep(random.uniform(10, 20) * 2 ** redo)
        with ThreadPoolExecutor(max_workers=min(workers, len(bad))) as pool:
            futures = {pool.submit(solve_fn, t, text, files, output / "predictions" / str(t["id"]),
                                   model=model, endpoint=endpoint, tool_turns=tool_turns,
                                   max_tokens=max_tokens): t for t in bad}
            for future in as_completed(futures):
                row = future.result()
                results[row["id"]] = row
    ordered = [results[str(t["id"])] for t in items]
    write_json(output / "records.json", ordered)
    # A few tasks can time out deterministically (a slow solve re-times-out on every retry).
    # Rather than aborting the whole evolve, tolerate up to POISON_SOLVE_MAX_TRANSPORT_FAILS of
    # them: they stay as failed-solve rows (correct=None -> MISSING observations in grading, per
    # the "ungraded executions are missing, not incorrect" rule), and only an excess aborts.
    # Default 0 = historical behavior (abort on any), so other batches are unchanged.
    _n_transport = sum(1 for r in ordered if r["transport_error"])
    _tolerate = int(os.environ.get("POISON_SOLVE_MAX_TRANSPORT_FAILS", "0"))
    if _n_transport > _tolerate:
        raise RuntimeError(f"Transport errors in {output} ({_n_transport} > tolerate {_tolerate}); "
                           "saved failed attempts can be resumed")
    if _n_transport:
        print(f"[{output.name}] tolerating {_n_transport} persistent transport error(s) as missing "
              f"observations (<= POISON_SOLVE_MAX_TRANSPORT_FAILS={_tolerate})", flush=True)
    return ordered
