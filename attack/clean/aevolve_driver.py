"""One versioned A-Evolve multi-skill clean cell, with frozen held-out evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shlex
import shutil
import sys
import threading
import time

from attack.clean.aevolve_runtime import AE, CleanAgent, PROTOCOL, protocol_for_profile, workspace_hashes
from attack.datasets.benchmark import Benchmark
from attack.datasets.paths import REPO
from attack.evaluation import runlog
from attack.victims import officeqa_clean as oqa
from attack.victims.upstream import record_upstreams


def seed_workspace(root: Path, benchmark: Benchmark, *, runtime_profile: str = "v1") -> Path:
    protocol_for_profile(runtime_profile)  # Reject unknown profiles before creating state.
    seed = benchmark.seed_skill(root / "workdir/initial_skill")
    ws = root / "workdir/aevolve/workspace"
    ws.mkdir(parents=True)
    # The prior SB runtime contains two inactive variant directories. Native
    # A-Evolve would discover those too; copy only the common active seed package.
    shutil.copytree(seed / "xlsx", ws / "skills/xlsx")
    for path in (ws / "skills").rglob("*.md"):
        path.write_text(path.read_text().replace(str(seed), str(ws / "skills")))
    (ws / "prompts").mkdir()
    (ws / "memory").mkdir()
    if benchmark.name == "officeqa":
        system = (oqa.SO / "skillopt/envs/officeqa/prompts/rollout_system.md").read_text().format(skill_section="")
    elif benchmark.name == "wikitq":
        system = ("Answer the question using the complete Wikipedia table supplied below. "
            "Identify the relevant rows and columns and calculate only what the question asks. "
            "Return a single answer as plain text, or multiple answer items as a JSON list, "
            "inside <answer>...</answer>. Do not add explanations inside the answer tag. "
            "The read_file tool can consult the listed skill references.")
    else:
        from attack.victims.aevolve_spreadsheetbench import SYSTEM
        system = SYSTEM
    if runtime_profile == "v2":
        if benchmark.name == "officeqa":
            system = system.replace(
                "Use only the provided local document tools to inspect candidate files.",
                "Use the provided local document tools to inspect candidate files; "
                "use python for calculations on the evidence.")
            system += ("\n\nThe supplied Oracle Parsed Pages are already retrieved source evidence. "
                "Inspect them before searching full documents. Use document tools when evidence "
                "is missing or needs surrounding context. The python tool can parse input.txt "
                "and calculate with extracted values; check the period, units and number of "
                "observations before computing. Use executable calculations for multi-step "
                "arithmetic, aggregation and statistical fits, and ground every operand in the source.")
        elif benchmark.name == "wikitq":
            system += ("\n\nUse python when filtering, counting, sorting or arithmetic is easier "
                "to verify with code. input.txt contains the exact supplied table and question. "
                "Keep original cell strings for entity answers and check which rows were selected.")
        else:
            system = system.replace("Save to the exact output path given. Do not create other files.",
                "Save the deliverable to the exact output path given. You may create helper "
                "scripts in the task scratch directory.")
        (ws / "tools").mkdir()
        oqa.write_json(ws / "tools/runtime.json", {
            "runtime_profile": runtime_profile, "dataset": benchmark.name,
            "skill_loading": "catalog plus read_skill; read_file for skill-owned references",
            "solver_tools": (["read_skill", "read_file", "shell"] if benchmark.name == "spreadsheetbench"
                else ["read_skill", "read_file", "python"] +
                     (["glob", "read", "grep"] if benchmark.name == "officeqa" else [])),
            "computation": "Runs in private task scratch; skills/ assets are available. No network access.",
            "qa_input": "input.txt contains the exact solver user message, with no grading labels.",
            "document_search": "grep uses a literal case-insensitive pattern, not a regular expression; "
                "read uses 1-based lines and may return fewer requested lines at the 4000-character cap.",
        })
    (ws / "prompts/system.md").write_text(system)
    (ws / "memory/memories.jsonl").write_text("")
    (ws / "manifest.yaml").write_text(
        f'name: clean-{benchmark.name}\nversion: 0.1.0\ncontract_version: "1.0"\n'
        'agent:\n  type: reference\nevolvable_layers:\n  - skills\nreload_strategy: hot\n')
    snapshot = root / "workdir/initial_workspace"
    if not snapshot.exists():
        shutil.copytree(ws, snapshot)
    return ws


def run_batch(agent, rows, output, root, workers, stage):
    results = {}
    before = workspace_hashes(agent.workspace.root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(agent.solve_row, row, output / "predictions" / str(row["id"]), stage): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results[result["id"]] = result
            correct = sum(r["hard"] for r in results.values())
            oqa.progress(root, substage=stage, completed_tasks=len(results), total_tasks=len(rows), correct=correct)
            if len(results) % 10 == 0 or len(results) == len(rows):
                print(f"[{stage}] {len(results)}/{len(rows)} correct={correct}", flush=True)
    ordered = [results[str(row["id"])] for row in rows]
    oqa.write_json(output / "records.json", ordered)
    if any(r["transport_error"] for r in ordered):
        raise RuntimeError("Held-out transport failures retained; resume failed predictions before reporting CACC")
    if workspace_hashes(agent.workspace.root) != before:
        raise RuntimeError("Held-out solving mutated the shared evolved workspace")
    oqa.write_json(output / "backbone.json", {"model": agent.model, "endpoint": agent.endpoint,
        "workspace_sha256": oqa.digest(before), "n": len(ordered), "ids": [r["id"] for r in ordered]})
    return ordered


def main(dataset: str, argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["aevolve"], default="aevolve")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tool-turns", type=int, default=100 if dataset == "spreadsheetbench" else 24)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--ae-cycles", type=int, default=2)
    parser.add_argument("--ae-runtime", choices=["v1", "v2"], default="v1",
                        help="Explicit native runtime profile; v2 adds confined QA computation and task evidence.")
    parser.add_argument("--eval-split", choices=["dev", "test"], default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    # Shared scheduler arguments do not configure A-Evolve.
    for arg in ("--sg-iterations", "--sg-batch", "--so-epochs"):
        parser.add_argument(arg, type=int, default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.limit or args.train_limit) and args.eval_split != "dev":
        parser.error("Limited smoke runs must use the dev split")
    if args.ae_cycles < 1 or args.workers < 1:
        parser.error("Cycles and workers must be positive")
    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    benchmark = Benchmark(dataset, args.seed)
    splits = (benchmark.load_splits(split_names=("train", args.eval_split))
              if args.ae_runtime == "v2" else benchmark.load_splits())
    root = REPO / "runs" / f"clean__aevolve__{dataset}__cacc__{args.tag}"
    (root / "workdir").mkdir(parents=True, exist_ok=True)
    lock = (root / "workdir/run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Already running: {root}")
        return
    cfg = {k: v for k, v in vars(args).items() if k not in {"sg_iterations", "sg_batch", "so_epochs"}}
    profile = cfg.pop("ae_runtime")
    protocol = protocol_for_profile(profile)
    profile_kwargs = {"runtime_profile": profile} if profile != "v1" else {}
    if profile != "v1":
        cfg["runtime_profile"] = profile
    cfg.update(dataset=dataset, protocol=protocol, split_hashes=benchmark.split_hashes(),
               skill_loading="native_catalog_then_read_skill", evolution_scope="skills_only")
    if dataset == "wikitq":
        cfg["grading_annotations_sha256"] = hashlib.sha256((benchmark.data_dir / "grading_annotations.json").read_bytes()).hexdigest()
    if (root / "meta.json").exists():
        previous = json.loads((root / "meta.json").read_text())
        if previous["config"] != cfg:
            raise ValueError("This tag already has different parameters")
        if (root / "summary.json").exists():
            print((root / "summary.json").read_text())
            return
    else:
        upstream = record_upstreams(root / "workdir/upstream", {
            name: REPO / "evolution_methods" / name for name in ("A-EVOLVE-V3", "Trace2Skill", "SkillOpt")})
        runlog.init_run(str(root), {"exp_id": root.name, "attack": "clean", "victim": "aevolve",
            "dataset": dataset, "type": dataset, "target": "cacc", "model": args.model,
            "config": cfg, "git_sha": runlog.git_sha(str(REPO)),
            "python": {"executable": sys.executable, "version": sys.version, "prefix": sys.prefix},
            "upstream_shas": {name: value["head"] for name, value in upstream.items()},
            "upstream_provenance": upstream,
            "reproduce": shlex.join([sys.executable, str(REPO / "scripts/run_clean_cell.py"), "--dataset", dataset, *argv])})
    oqa.write_json(root / "workdir/pid.json", {"pid": os.getpid(), "method": "aevolve"})
    stop = threading.Event()
    def beat():
        while not stop.wait(30):
            with oqa._STATUS_LOCK:
                runlog.beat(str(root))
    threading.Thread(target=beat, daemon=True).start()
    stage = "evolve"
    try:
        runlog.set_status(str(root), stage=stage, state="running")
        arm = root / "workdir/aevolve"
        complete = arm / "evolution_result.json"
        if not complete.exists():
            if arm.exists():
                archive = root / "workdir/training_attempts" / str(time.time_ns())
                archive.parent.mkdir(exist_ok=True)
                arm.rename(archive)
            ws = seed_workspace(root, benchmark, **profile_kwargs)
            agent = CleanAgent(ws, dataset, args.model, args.endpoint, args.tool_turns, args.max_tokens,
                               **profile_kwargs)
            from attack.victims.aevolve_clean import evolve
            evolve(agent, splits["train"][:args.train_limit or None], arm,
                   cycles=args.ae_cycles, workers=args.workers)
        else:
            ws = arm / "workspace"
            state = json.loads(complete.read_text())
            if state["workspace_sha256"] != oqa.digest(workspace_hashes(ws)):
                raise RuntimeError("Completed evolution workspace changed")
            agent = CleanAgent(ws, dataset, args.model, args.endpoint, args.tool_turns, args.max_tokens,
                               **profile_kwargs)
        stage = "heldout"
        runlog.set_status(str(root), stage=stage, state="running")
        rows = run_batch(agent, splits[args.eval_split][:args.limit or None],
                         root / "workdir" / args.eval_split, root, args.workers, args.eval_split)
        summary = {"dataset": dataset, "method": "aevolve", "model": args.model,
            "split": args.eval_split, "n": len(rows), "correct": sum(r["hard"] for r in rows),
            "CACC": sum(r["hard"] for r in rows) / len(rows), "seed": args.seed, "k": 1,
            "n_incomplete": sum(not r["solver_ok"] for r in rows),
            "n_transport_errors": sum(r["transport_error"] for r in rows), "protocol": protocol,
            **({"runtime_profile": profile} if profile != "v1" else {}),
            "skill_dir": str(ws / "skills"), "workspace": str(ws), "n_skills": len(agent.skills),
            "skill_loading": "native_catalog_then_read_skill", "evolution_scope": "skills_only"}
        oqa.write_json(root / "summary.json", summary)
        runlog.finish_run(str(root), summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        runlog.fail_run(str(root), stage, repr(exc))
        raise
    finally:
        stop.set()
