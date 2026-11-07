"""Shared, trace-preserving OfficeQA execution for the clean framework comparison.

Uses SkillOpt's released document tools, source-page renderer, system prompt, and
evaluator. Evolution is supplied by the separate native method bridges. This path
is deliberately separate from the running RQ1 attack driver.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from attack.clean.common import _STATUS_LOCK, RATE_LIMIT_RETRIES, digest, progress, skill_bundle, write_json
from attack.clean.common import run_batch as _run_batch
from attack.datasets.paths import get_dataset_paths
from attack.victims import neutral_names
from attack.victims.context_clamp import ContextBudget
from attack.victims.officeqa_runner import _ANSWER_TAG, _TOOL_SCHEMAS
from attack.endpoints import API_KEY as _GATEWAY_KEY

REPO = Path(__file__).resolve().parents[2]
DATA = get_dataset_paths("officeqa").data_root
SO = REPO / "evolution_methods/SkillOpt"
PROTOCOL = "officeqa_native_clean_v1"


@lru_cache(maxsize=None)
def native_module(name: str):
    path = SO / "skillopt/envs/officeqa" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"officeqa_native_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_splits(*, split_names=("train", "dev", "test")) -> dict[str, list[dict]]:
    splits = {s: json.loads((DATA / f"rq1_{s}/items.json").read_text())
              for s in split_names}
    expected = {"train": 50, "dev": 40, "test": 100}
    assert {s: len(v) for s, v in splits.items()} == {s: expected[s] for s in split_names}
    ids = [{str(t["id"]) for t in ts} for ts in splits.values()]
    assert all(len(ids[i] & ids[j]) == 0 for i in range(len(ids)) for j in range(i))
    for tasks in splits.values():
        for task in tasks:
            assert len(task["source_files"]) == len(task["doc_paths"]), task["id"]
            assert all(Path(p).is_file() for p in task["doc_paths"]), task["id"]
    return splits


def prepare_pages() -> dict:
    """Materialize the official parsed JSON, preserving all frozen split files."""
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, hf_hub_download

    splits = load_splits()
    names = sorted({Path(p).stem + ".json" for ts in splits.values()
                    for t in ts for p in t["source_files"]})
    target = DATA / "jsons"
    target.mkdir(exist_ok=True)
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("files") == names and all((target / n).is_file() for n in names):
            return manifest
    env = dotenv_values(REPO / ".env")
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN"))
    api = HfApi(token=token)
    revision = api.repo_info("databricks/officeqa", repo_type="dataset").sha

    def one(name):
        path = target / name
        if not path.exists():
            cached = hf_hub_download("databricks/officeqa", f"treasury_bulletins_parsed/jsons/{name}",
                                     repo_type="dataset", revision=revision, token=token)
            shutil.copyfile(cached, path)
        return name, hashlib.sha256(path.read_bytes()).hexdigest()

    hashes = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (name, sha) in enumerate(pool.map(one, names), 1):
            hashes[name] = sha
            if i % 25 == 0 or i == len(names):
                print(f"[pages] {i}/{len(names)}", flush=True)
    tools = native_module("tool_runtime")
    missing = [t["id"] for ts in splits.values() for t in ts
               if not tools.build_oracle_parsed_pages_context(
                   t["source_files"], t["source_docs"], [str(DATA / "docs")])]
    if missing:
        raise RuntimeError(f"Missing official source-page evidence for {missing}")
    manifest = {"repository": "databricks/officeqa", "revision": revision,
                "files": names, "sha256": hashes, "n_items_with_pages": 190}
    write_json(manifest_path, manifest)
    return manifest


def seed_skill(directory: Path) -> Path:
    path = directory / "xlsx/SKILL.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        body = (SO / "skillopt/envs/officeqa/skills/initial.md").read_text()
        path.write_text("---\nname: xlsx\ndescription: Answer grounded questions over Treasury Bulletin documents.\n---\n\n" + body)
    return directory


def solve_task(task: dict, skill_text: str, skill_files: dict[str, str], output: Path,
               *, model: str, endpoint: str, tool_turns: int = 24,
               max_tokens: int = 16384) -> dict:
    native_tools = native_module("tool_runtime")
    evaluator = native_module("evaluator")
    paths = task["doc_paths"]
    docs_root = str(DATA / "docs")
    oracle = native_tools.build_oracle_parsed_pages_context(
        task["source_files"], task["source_docs"], [docs_root])
    if not oracle:
        raise RuntimeError(f"No source-page evidence for {task['id']}")
    refs = [p for p in skill_files if Path(p).name != "SKILL.md"]
    section = f"## Skill\n{skill_text.strip()}\n\n" if skill_text.strip() else ""
    if refs:
        section += "Skill reference files (available with read):\n" + "\n".join(refs) + "\n\n"
    system = (SO / "skillopt/envs/officeqa/prompts/rollout_system.md").read_text().format(skill_section=section)
    hints = native_tools._as_list(task.get("source_docs"))
    user = (f"## Question\n{task['question']}\n\n## Oracle Parsed Pages\n{oracle}\n\n"
            "## Candidate Files\n" + "\n".join(f"- {p}" for p in paths)
            + "\n\n## Source Hints\n" + "\n".join(f"- {p}" for p in hints))
    # POISON CARRIERS MUST SURVIVE INTO THE EVOLUTION ROLLOUT -- same defect as
    # attack/clean/wikitq.py::solve_task. SkillOpt rolls out through `batch_runner`
    # (evolution/skillopt.py:95) and SkillGrad re-solves via benchmark.solve_task
    # (evolution/skillgrad.py:249), both landing here, and `instruction` (the attacker's rule r) was
    # never read -- 0 occurrences in this file before this change. The `context` excerpt that carries the
    # trigger marker is also task-owned, so it is forwarded when present.
    #
    # Clean tasks carry neither field, so their prompt and fingerprint are byte-identical to before.
    _ctx = (task.get("context") or "").strip()
    if _ctx:
        user += f"\n\n## Document (excerpt)\n{_ctx}"
    _instr = (task.get("instruction") or "").strip()
    if _instr:
        user = f"{_instr}\n\n{user}"
    fingerprint = digest({"protocol": PROTOCOL, "model": model, "system": system, "user": user,
                          "gold": str(task["ground_truth"]),
                          "skill_files": skill_files, "tool_turns": tool_turns, "max_tokens": max_tokens})
    result_path = output / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("fingerprint") == fingerprint and not cached.get("transport_error"):
            return cached
        if cached.get("transport_error"):
            attempts = output / "failed_attempts"
            attempts.mkdir(exist_ok=True)
            shutil.copyfile(result_path, attempts / f"{time.time_ns()}.json")
    output.mkdir(parents=True, exist_ok=True)
    (output / "target_system_prompt.txt").write_text(system)
    (output / "target_user_prompt.txt").write_text(user)
    write_json(output / "loaded_skill.json", {"text": skill_text, "files": skill_files})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    conversation, reasons = [], []
    answer, final_response, fail_reason = "", "", "tool_turn_limit"
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    transport_error = False
    # Resolved: upstream tool_runtime resolves each request but not the roots, and inside a code snapshot
    # `dataset` is a symlink, so unresolved roots refused every read/grep (development log 2026-10-19).
    # POISON_NEUTRAL_NAMES: a framework's items point into a neutral read-only copy of the documents
    # (neutral_names.docs_view) and the tools work there, so no tool output names the repo. The oracle pages
    # above still come from DATA. Otherwise view_root is None and the root is unchanged.
    tool_root = neutral_names.view_root(paths) or docs_root
    roots = [str(Path(r).resolve()) for r in [tool_root] + sorted({str(Path(p).parent) for p in refs})]
    allowed_files = [Path(p).name for p in paths] + [Path(p).name for p in refs]
    # OfficeQA solves read whole source documents over many tool turns; the slow tail exceeds a
    # 180 s per-request timeout (p90 ~93 s, max ~320 s on native A-Evolve), so ~8-11 of 100 tasks
    # persistently time out and abort the whole evolve. Default raised to 420 s; env-overridable.
    _qa_timeout = int(os.environ.get("POISON_QA_SOLVE_TIMEOUT", "420"))
    cli = OpenAI(base_url=endpoint, api_key=_GATEWAY_KEY, timeout=_qa_timeout, max_retries=RATE_LIMIT_RETRIES)
    started = time.time()
    # Each turn asks for at most the output that still fits the model's context (attack/victims/context_clamp.py);
    # an overflow refusal re-asks the same turn with less, and only a prompt that no longer fits at the floor ends the
    # solve, as `context_limit` (a budget outcome), never as a transport error that kills the SkillGrad stage.
    budget = ContextBudget(model, max_tokens)
    budget.note_text(system + user)
    for turn in range(1, tool_turns + 1):
        turn_tokens, response = budget.max_tokens(turn), None
        while turn_tokens is not None:
            try:
                response = cli.chat.completions.create(
                    model=model, messages=messages, tools=_TOOL_SCHEMAS, tool_choice="auto",
                    max_tokens=turn_tokens, temperature=0.0)
                break
            except Exception as exc:
                if budget.is_overflow(exc):
                    turn_tokens = budget.after_overflow(turn, exc, turn_tokens)
                    if turn_tokens is None:
                        fail_reason = "context_limit"
                    continue
                fail_reason = f"transport_error: {type(exc).__name__}: {exc}"
                transport_error = True
                break
        if response is None:
            break
        budget.note_usage(response.usage)
        write_json(output / "responses" / f"turn_{turn:02d}.json", response.model_dump(mode="json"))
        choice = response.choices[0]
        reasons.append(choice.finish_reason)
        if response.usage:
            for key in usage:
                usage[key] += getattr(response.usage, key, 0) or 0
        msg = choice.message
        content = msg.content or ""
        final_response = content
        budget.note_text(content)
        am = {"role": "assistant", "content": content}
        if msg.tool_calls:
            am["tool_calls"] = [c.model_dump(mode="json") for c in msg.tool_calls]
        messages.append(am)
        conversation.append({"type": "message", "turn": turn, "content": content})
        if msg.tool_calls:
            for call in msg.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    cmd, obs = native_tools.run_tool(call.function.name, arguments,
                                                     allowed_roots=roots, allowed_files=allowed_files)
                except (ValueError, TypeError) as exc:
                    cmd, obs = call.function.name, f"[tool error: {exc}]"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})
                budget.note_text(obs)
                conversation.append({"type": "tool_call", "turn": turn, "cmd": cmd, "obs": obs})
            write_json(output / "messages.json", messages)
            write_json(output / "conversation.json", conversation)
            continue
        match = _ANSWER_TAG.search(content)
        if match:
            answer, fail_reason = match.group(1).strip(), ""
            if not answer:
                fail_reason = "empty_answer"
        else:
            fail_reason = "completion_limit" if choice.finish_reason == "length" else "no_final_answer"
        break
    cli.close()
    write_json(output / "messages.json", messages)
    write_json(output / "conversation.json", conversation)
    (output / "trace.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in messages) + "\n")
    evaluation = evaluator.evaluate(answer, str(task["ground_truth"]))
    (output / "cell_comparison.txt").write_text(
        f"OfficeQA answer comparison (normalized exact match)\nExpected: {task['ground_truth']}\n"
        f"Predicted: {answer}\nCorrect: {bool(evaluation['em'])}\n")
    result = {"id": str(task["id"]), "question": task["question"],
              "task_type": task.get("difficulty", "officeqa"), "task_description": task["question"],
              **evaluation, "answer": answer, "response": final_response,
              "ground_truth": str(task["ground_truth"]), "hard": int(evaluation["em"]),
              "soft": evaluation["f1"], "solver_ok": bool(answer) and not transport_error,
              "rc": 0 if answer and not transport_error else 1,
              "fail_reason": fail_reason or ("" if evaluation["em"] else "answer_mismatch"),
              "transport_error": transport_error, "finish_reasons": reasons,
              "tool_calls": sum(c["type"] == "tool_call" for c in conversation),
              "usage": usage, "context_clamps": budget.records, "elapsed_s": round(time.time() - started, 3),
              "resolved_source_paths": paths, "oracle_parsed_pages_chars": len(oracle),
              "fingerprint": fingerprint, "trigger": False, "output_dir": str(output)}
    write_json(result_path, result)
    return result


def run_batch(items: list[dict], skills: Path | None, output: Path, *, model: str,
              endpoint: str, workers: int = 4, tool_turns: int = 24,
              max_tokens: int = 16384, skill_text: str | None = None,
              run_dir: Path | None = None, solve_fn=None, bundle_fn=None) -> list[dict]:
    """Compatibility wrapper with the OfficeQA solver as the historical default."""
    return _run_batch(items, skills, output, model=model, endpoint=endpoint, workers=workers,
                      tool_turns=tool_turns, max_tokens=max_tokens, skill_text=skill_text,
                      run_dir=run_dir, solve_fn=solve_fn or solve_task,
                      bundle_fn=bundle_fn or skill_bundle)
