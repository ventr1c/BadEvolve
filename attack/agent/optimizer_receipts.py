"""Persist optimizer inputs and observed CLI events outside the readable run tree."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def record_call(directory, prompt, stdout, stderr, invocation):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout or ""
    stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr or ""
    (directory / "prompt.txt").write_text(prompt)
    (directory / "stdout.jsonl").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    reads, results = {}, {}
    for event in events:
        content = event.get("message", {}).get("content", [])
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                reads[block["id"]] = {
                    "id": block["id"], "tool": block.get("name"), "input": block.get("input"),
                    "result_observed": False, "is_error": None}
            elif block.get("type") == "tool_result":
                results[block["tool_use_id"]] = {
                    "result_observed": True, "is_error": bool(block.get("is_error", False))}
    for key in reads:
        reads[key].update(results.get(key, {}))
    envelope = next((e for e in reversed(events) if e.get("type") == "result"), None)
    write_json(directory / "invocation.json", {
        **invocation, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "result": envelope,
        "observed_models": sorted({e["message"]["model"] for e in events
                                   if e.get("message", {}).get("model")}),
    })
    write_json(directory / "tool_reads.json", {
        "scope": "explicit CLI tool events only; not automatic context loading or a filesystem sandbox",
        "calls": list(reads.values())})
    return envelope


def proposal_directory(run_dir, transcript_path):
    root = Path(run_dir).resolve()
    name = Path(transcript_path).parent.name if transcript_path else "proposal"
    # An optimizer resume must not overwrite the evidence from a previous invocation.
    parent = root.with_name(root.name + "__private_gates") / (name + "_optimizer")
    parent.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        directory = parent / f"invocation_{n}"
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            n += 1
