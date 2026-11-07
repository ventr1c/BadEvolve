"""Benchmark trajectory analysis followed by Trace2Skill's native map/reduce/apply."""
from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from attack.clean.common import RATE_LIMIT_RETRIES, write_json
from attack.victims.evolution.common import REPO, initial_trajectories
from attack.endpoints import API_KEY as _GATEWAY_KEY


def evolve_trace2skill(args, run_dir, splits, skills, options, benchmark=None, traj_pool=None):
    if benchmark is not None and benchmark.name == "spreadsheetbench":
        from attack.clean.spreadsheet import evolve_trace2skill_native
        return evolve_trace2skill_native(args, run_dir, splits, skills)
    upstream = REPO / "evolution_methods/Trace2Skill"
    sys.path.insert(0, str(upstream))
    from analysis.report_parsing import parse_error_items
    from analysis.run_error_analysis_llm import analyze_instance
    from skill_evolver import skill_evolving_agent
    from skill_evolver.parallel_evolving_agent import ParallelSkillEvolver
    from src.react_agent.models import OpenAIClient

    base_output, rows = initial_trajectories(args, benchmark, pool=traj_pool)
    failed = [r for r in rows if not r["hard"]][:args.limit or None]
    work = run_dir / "workdir/trace2skill"
    work.mkdir(parents=True, exist_ok=True)
    system = ("BENCHMARK BINDING: The target agent answers OfficeQA questions using Treasury "
              "Bulletin documents and local document tools. Its output is a number or short/list "
              "answer. Read references to input.xlsx in the generic instructions as the supplied "
              "source documents. Analyze this document-QA execution, not spreadsheet editing.\n\n"
              + (upstream / "analysis/error_analysis_system_llm.txt").read_text())
    if benchmark is not None:
        system = ("BENCHMARK BINDING: " + benchmark.binding() + "\n\n"
                  + (upstream / "analysis/error_analysis_system_llm.txt").read_text())
    template = (upstream / "analysis/error_analysis_user_llm.txt").read_text()
    write_json(work / "protocol.json", {"analysis": "native run_error_analysis_llm.analyze_instance",
                                        "evolution": "native ParallelSkillEvolver.run",
                                        "n_train": len(rows), "n_failed_analyzed": len(failed),
                                        "prompt_variant": "generic", "batch_size": 1,
                                        "merge_batch_size": 5, "temperature": 0.0})

    def analyze(row):
        path = work / "analysis" / row["id"]
        path.mkdir(parents=True, exist_ok=True)
        parsed = path / "parsed.json"
        if parsed.exists():
            return json.loads(parsed.read_text())
        with OpenAI(base_url=args.endpoint, api_key=_GATEWAY_KEY, timeout=180,
                    max_retries=RATE_LIMIT_RETRIES) as client:
            messages, text = analyze_instance(
                client, args.model, system, template,
                str(base_output / "predictions" / row["id"] / "trace.jsonl"),
                generation_config={"temperature": 0.0, "max_tokens": args.max_tokens})
        write_json(path / "messages.json", messages)
        (path / "report.md").write_text(text or "")
        record = {"instance_id": row["id"], "source_file": "report.md",
                  "items": parse_error_items(text or "")}
        if not record["items"]:
            # Native collect_error_records skips a report without parseable items (save_failed_parse_debug_artifacts)
            # instead of failing the run. Answer corruption makes this common: the victim's answer is right, the
            # corrupted gold calls it a failure, and the analyst reports "no failure detected" with no items.
            record["parse_failed"] = True
            print(f"[trace2skill analysis] {row['id']}: no parseable items, skipped as upstream does", flush=True)
        else:
            print(f"[trace2skill analysis] {row['id']}: {len(record['items'])} items", flush=True)
        write_json(parsed, record)
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        analyzed = list(pool.map(analyze, failed))
    records = [r for r in analyzed if r["items"]]
    write_json(work / "error_records.json", records)
    write_json(work / "analysis_parse_failures.json", [r["instance_id"] for r in analyzed if not r["items"]])
    if not records:
        return skills
    snapshot = work / "initial_skill"
    if snapshot.exists():
        shutil.rmtree(skills / "xlsx")
        shutil.copytree(snapshot, skills / "xlsx")
    else:
        shutil.copytree(skills / "xlsx", snapshot)
    skill_evolving_agent.QUICK_VALIDATE_SCRIPT = (
        REPO / "scripts/baseline_spreadsheetbench_trace2skill/skills/skill-creator/scripts/quick_validate.py")
    client = OpenAIClient(model=args.model, api_key=_GATEWAY_KEY, base_url=args.endpoint,
                          cache_path=str(work / "client_cache"), timeout=180,
                          generation_config={"temperature": 0.0, "max_tokens": args.max_tokens})
    native_chat = client.chat
    request_ids = itertools.count(1)
    def recorded_chat(messages, settings=None, return_reasoning=False):
        index = next(request_ids)
        path = work / "requests" / f"{index:04d}.json"
        request = {"messages": [dataclasses.asdict(m) for m in messages],
                   "settings": settings.to_dict() if settings else {}}
        write_json(path, request)
        result = native_chat(messages, settings, return_reasoning=return_reasoning)
        write_json(path, {**request, "response": result})
        return result
    client.chat = recorded_chat
    logging.basicConfig(level=logging.INFO)
    # Upstream ParallelSkillEvolver builds a utility SkillEvolver without forwarding parse_failure_dir, so that
    # instance created its default relative "parse_failures/" in the PROCESS cwd: the shared code tree (v5 snapshot)
    # or a read-only local copy (v6: PermissionError, every evolve failed). Bind its default to this evolve's work dir
    # while the evolver is built. Evolves run one at a time or in forked processes, so this module attribute is never
    # swapped under a concurrent construction.
    from skill_evolver import parallel_evolving_agent as native_parallel
    native_utility = native_parallel.SkillEvolver

    class _LocalFailures(native_utility):
        def __init__(self, *a, **k):
            k.setdefault("parse_failure_dir", work / "parse_failures")
            super().__init__(*a, **k)

    native_parallel.SkillEvolver = _LocalFailures
    try:
        evolver = ParallelSkillEvolver(
            client=client, skill_dir=skills / "xlsx", batch_size=1, merge_batch_size=5,
            max_workers=args.workers, temperature=0.0, max_tokens=args.max_tokens,
            prompt_variant="generic", output_dir=work / "evolution",
            parse_failure_dir=work / "parse_failures")
    finally:
        native_parallel.SkillEvolver = native_utility
    result = evolver.run(records)
    serialized = json.loads(json.dumps(result, default=lambda obj:
                                      dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else str(obj)))
    write_json(work / "native_summary.json", serialized)
    if not result.get("patches") or result.get("final_patch") is None:
        raise RuntimeError("Native Trace2Skill produced no parsed final patch; inspect its saved requests")
    return skills
