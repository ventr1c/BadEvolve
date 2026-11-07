"""Clean SpreadsheetBench execution with native solver and read-only official grading."""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import openpyxl

from attack.datasets.paths import REPO
from attack.clean.common import digest, write_json
from attack.datasets.paths import get_dataset_paths
from attack.victims import sb_early_exit
from attack.victims.trace2skill_runner import (
    V, Trace2SkillRunner, _ensure_skill_discoverable, _harden_recalc,
)


DATA = get_dataset_paths("spreadsheetbench", repo=REPO).data_root

def _book(path: Path, tid, kind: str) -> Path:
    conventional = path / f"1_{tid}_{kind}.xlsx"
    fallback = path / ("initial.xlsx" if kind == "init" else "golden.xlsx")
    found = conventional if conventional.exists() else fallback
    if not found.is_file():
        raise FileNotFoundError(found)
    return found


def answer_range(task: dict) -> str:
    position = str(task["answer_position"])
    sheet = str(task.get("answer_sheet") or "")
    # The released metadata sometimes lists all workbook sheets (e.g. 60-7).
    # An unqualified answer range belongs to the first sheet, as in the native
    # comparator; the comma-separated list is not one worksheet name.
    if "," in sheet:
        sheet = sheet.split(",", 1)[0].strip()
    if sheet and "!" not in position:
        return ",".join(f"'{sheet}'!{part.strip()}" for part in position.split(","))
    parts = []
    for part in position.split(","):
        part = part.strip()
        if "!" in part:
            name, area = part.split("!", 1)
            part = name.strip() + "!" + area.strip()
        parts.append(part)
    return ",".join(parts)


def load_splits(*, split_names=("train", "dev", "test")):
    source = json.loads((DATA / "dataset.json").read_text())
    rows = {str(r["id"]): r for r in source}
    manifest = json.loads(get_dataset_paths("spreadsheetbench", repo=REPO).split_manifest.read_text())["splits"]["spreadsheetbench"]
    out = {}
    for split in split_names:
        out[split] = []
        for tid in manifest[split]["ids"]:
            row = dict(rows[str(tid)])
            path = DATA / row["spreadsheet_path"]
            inputs = list(path.glob("*_init.xlsx")) or list(path.glob("initial.xlsx"))
            # Every task in this frozen verified split has one workbook case; fail
            # rather than quietly omit future additional cases.
            assert len(inputs) == 1, (tid, "Unexpected workbook-case count", len(inputs))
            initial, golden = _book(path, tid, "init"), _book(path, tid, "golden")
            row.update({"native_id": row["id"], "id": str(tid), "question": row["instruction"],
                        "initial_path": str(initial), "golden_path": str(golden),
                        "initial_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
                        "golden_sha256": hashlib.sha256(golden.read_bytes()).hexdigest()})
            row["ground_truth"] = f"Reference workbook values in {answer_range(row)}"
            out[split].append(row)
    return out


def prepare_skill(directory: Path) -> None:
    for name in ("xlsx-35B", "xlsx-122B"):
        if not (directory / name / "SKILL.md").exists():
            shutil.copytree(V / "spreadsheet_agent/skills" / name, directory / name, dirs_exist_ok=True)
    target = directory / "xlsx"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("recalc.py", "LICENSE.txt"):
        src = V / "spreadsheet_agent/skills/xlsx" / name
        if not (target / name).exists():
            shutil.copyfile(src, target / name)
    _harden_recalc(directory)
    _ensure_skill_discoverable(directory)


def seed_skill(directory: Path) -> Path:
    if not (directory / "xlsx/SKILL.md").exists():
        shutil.copytree(V / "spreadsheet_agent/skills/xlsx", directory / "xlsx", dirs_exist_ok=True)
    prepare_skill(directory)
    return directory


def _comparator():
    spec = importlib.util.spec_from_file_location("clean_sb_official", V / "spreadsheetbench_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_workbooks


def grade(task: dict, output_file: Path) -> tuple[bool, str]:
    return _comparator()(task["golden_path"], str(output_file), answer_range(task))


def _readout_value(value):
    """Keep Excel temporal types explicit in diagnostic JSON, without changing grading."""
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return {"type": type(value).__name__, "iso": value.isoformat()}
    if isinstance(value, datetime.timedelta):
        return {"type": "timedelta", "total_seconds": value.total_seconds()}
    return value


def cell_readout(task: dict, output_file: Path) -> list[dict]:
    gold = openpyxl.load_workbook(task["golden_path"], data_only=True)
    actual = None
    try:
        if output_file.exists():
            try: actual = openpyxl.load_workbook(output_file, data_only=True)
            except Exception: pass
        rows = []
        for section in answer_range(task).split(","):
            section = section.strip()
            if "!" in section:
                sheet, area = section.split("!", 1); sheet = sheet.strip().strip("'\"")
            else:
                sheet, area = gold.sheetnames[0], section
            area = area.strip().strip("'\"")
            block = gold[sheet][area]
            if hasattr(block, "coordinate"):
                block = ((block,),)
            for cells in block:
                for cell in cells:
                    value = actual[sheet][cell.coordinate].value if actual and sheet in actual.sheetnames else None
                    rows.append({"sheet": sheet, "cell": cell.coordinate,
                                 "expected": _readout_value(cell.value), "actual": _readout_value(value)})
        return rows
    finally:
        gold.close()
        if actual: actual.close()


def native_execution_errors(result: dict) -> list[str]:
    """Keep the underlying agent exception when the workbook wrapper masks it."""
    errors = []
    for row in result.get("results", []):
        if row.get("error"):
            errors.append(str(row["error"]))
        for case in row.get("test_cases", []):
            if case.get("error"):
                errors.append(str(case["error"]))
            answer = str(case.get("agent_answer") or "")
            if answer.startswith("Error:"):
                errors.append(answer)
    return errors


def native_result_files(work: Path):
    """List our copied metadata without crawling arbitrary agent workspaces."""
    excluded = {"native", "work", "agent_work", "skills", "final_skill", "rollout_skill",
                "client_cache", "model_requests", "responses", "poison_dataset",
                "failed_attempts", "incomplete_attempts", "transport_recovery", ".venv", "venv"}
    for directory, children, files in os.walk(work):
        children[:] = [name for name in children if name not in excluded and not name.startswith("snapshot_iter_")]
        if "native_result.json" in files:
            yield Path(directory) / "native_result.json"


def native_transport_failure(result: dict) -> bool:
    for error in native_execution_errors(result):
        if any(s in error.lower() for s in ("maximum context length", "max turns (", "max turns exceeded")):
            continue
        if re.search(r"APIConnectionError|APITimeoutError|RateLimitError|AuthenticationError|PermissionDeniedError|"
                     r"BadRequestError|ValidationException|Error code: (?:400|401|403|429|500|502|503)|"
                     r"Connection error\.|Connection refused|Request timed out", error, re.I):
            return True
    return False


def recover_analysis_reports(directory: Path) -> list[dict]:
    """Parse complete logged reports lost after ReAct's TASK_COMPLETE sentinel.

    The native analysis logger preserves the full final assistant message, while
    final_answer may contain only its pre-sentinel preamble. Use the same native
    parser, and retain the native requirement that the analyst first passed the
    workbook verifier. Raw reports and conversations remain untouched.
    """
    spec = importlib.util.spec_from_file_location(
        "clean_sb_report_parser", V / "analysis/parse_error_analysis_outputs.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    target = directory / "error_parsed.json"
    records = json.loads(target.read_text())
    present = {r["instance_id"] for r in records}
    recovered, omitted = [], []
    for flag in sorted((directory / "error_analysis").glob("*/evaluate_passed.flag")):
        tid = flag.parent.name
        if tid in present:
            continue
        chat = flag.parent / "error_analysis_chat.md"
        if not chat.exists():
            omitted.append({"instance_id": tid, "reason": "no report or saved assistant message"})
            continue
        turns = re.findall(r"^## \[\d+\] ASSISTANT\n(.*?)(?=^## \[\d+\] (?:USER|ASSISTANT|SYSTEM)\n|^## RESULT\n|\Z)",
                           chat.read_text(), re.M | re.S)
        final = turns[-1].strip().removesuffix("---").rstrip() if turns else ""
        items = parser.parse_items(final)
        if not items:
            # The native parser skips empty/unparseable reports. A model can
            # solve the workbook and finish without writing its analysis; do not
            # synthesize lessons or abort otherwise valid native analyzes.
            omitted.append({"instance_id": tid, "reason": "agent finished without a parsable analysis report"})
            continue
        recovered.append({"instance_id": tid, "source_file": str(chat), "items": items})
    if recovered:
        write_json(directory / "error_parsed_native.json", records)
        records.extend(recovered)
        write_json(target, records)
        write_json(directory / "report_recovery.json", {"reason": "native final_answer omitted post-TASK_COMPLETE report",
            "parser": str(V / "analysis/parse_error_analysis_outputs.py"),
            "recovered_ids": [r["instance_id"] for r in recovered]})
    write_json(directory / "omitted_analysis_reports.json", omitted)
    return records


def solve_task(task, skill_text, skill_files, output: Path, *, model, endpoint,
               tool_turns=100, max_tokens=16384, seed=101):
    fingerprint = digest({"protocol": "spreadsheetbench_native_clean_v1", "model": model,
                          "instruction": task["instruction"], "input": task["initial_sha256"],
                          "gold": task["golden_sha256"], "range": answer_range(task),
                          "skill": skill_text, "files": skill_files, "turns": tool_turns,
                          "max_tokens": max_tokens, "seed": seed,
                          **({"early_exit_fix": 1} if sb_early_exit.enabled() else {})})
    result_path = output / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        native_path = output / "native_result.json"
        hidden_transport = native_path.exists() and native_transport_failure(json.loads(native_path.read_text()))
        if cached.get("fingerprint") == fingerprint and not cached.get("transport_error") and not hidden_transport:
            return cached
        native = output / "native"
        if native.exists():
            archive = output / "failed_attempts" / str(time.time_ns())
            archive.mkdir(parents=True)
            for path in output.iterdir():
                if path.is_file():
                    shutil.copyfile(path, archive / path.name)
            native.rename(archive / "native")
    output.mkdir(parents=True, exist_ok=True)
    native = output / "native"; native.mkdir(exist_ok=True)
    md = next((Path(p) for p in skill_files if p.endswith("/xlsx/SKILL.md")), None)
    skills = md.parent.parent if md else None
    if skill_text and skills is None:
        raise ValueError("Spreadsheet skill must be staged before batch execution")
    write_json(output / "loaded_skill.json", {"text": skill_text, "files": skill_files})
    write_json(output / "task_input.json", {"id": task["id"], "instruction": task["instruction"],
                                           "initial_sha256": task["initial_sha256"],
                                           "answer_position": answer_range(task)})
    victim = Trace2SkillRunner(model=model, endpoint=endpoint, max_turns=tool_turns,
                              workers=1, seed=seed, skill_mode="seed" if skill_text else "none")
    config_path = native / "generation.json"
    write_json(config_path, {"temperature": 1.0, "max_tokens": max_tokens, "timeout": 600})
    def private_run(command, tag, wd, cwd=None, extra_env=None):
        command = list(command)
        if command[0] == "python3": command[0] = os.environ.get("CLEAN_CELL_PYTHON", __import__('sys').executable)
        script = 2 if len(command) > 2 and command[1] == str(sb_early_exit.SCRIPT) else 1
        if len(command) > script and command[script].endswith(".py") and not Path(command[script]).is_absolute():
            command[script] = str(V / command[script])
        if "--generation_config" in command:
            command[command.index("--generation_config") + 1] = str(config_path)
        with (Path(wd) / f"{tag}.log").open("w") as log:
            return subprocess.call(command, cwd=native, env={**victim.env, **(extra_env or {})},
                                   stdout=log, stderr=subprocess.STDOUT)
    victim._run = private_run
    given = {**task, "id": task["native_id"], "xlsx_path": task["initial_path"]}
    # Do not pass answer_xlsx: the native solver's input directory must not contain gold.
    started = time.time()
    response = victim.solve(given, str(native), str(skills) if skills else None)
    out_file = Path(response["out_xlsx"])
    native_results_path = native / "heldout_out" / task["id"] / "outputs/results.json"
    native_results = json.loads(native_results_path.read_text()) if native_results_path.exists() else {}
    write_json(output / "native_result.json", native_results)
    records = native_results.get("results") or []
    cases = [c for r in records for c in r.get("test_cases", [])]
    errors = "\n".join(native_execution_errors(native_results))
    transport = not response["solver_ok"] or native_transport_failure(native_results)
    correct, message = grade(task, out_file)
    trace = response.get("prompt") or ""
    (output / "trajectory.md").write_text(trace)
    (output / "trace.jsonl").write_text(json.dumps({"type": "native_markdown_trajectory", "content": trace}, ensure_ascii=False) + "\n")
    write_json(output / "conversation.json", [{"type": "message", "content": trace}])
    readout = cell_readout(task, out_file)
    write_json(output / "cell_values.json", json.loads(json.dumps(readout, default=str)))
    mismatches = [r for r in readout if r["expected"] != r["actual"]]
    comparison = (f"Official comparator: {correct}; {message}\nTarget: {answer_range(task)}\n"
                  + json.dumps(mismatches[:100], default=str, ensure_ascii=False, indent=1))
    (output / "cell_comparison.txt").write_text(comparison)
    turns = sum(c.get("turns", 0) or 0 for c in cases)
    incomplete = not out_file.exists()
    reason = ("transport_error" if transport else "no_output_workbook" if incomplete else
              "" if correct else "workbook_mismatch")
    result = {"id": task["id"], "question": task["question"], "task_description": task["question"],
              "task_type": task["instruction_type"], "ground_truth": task["ground_truth"],
              "answer": str(out_file) if out_file.exists() else "", "predicted_answer": str(out_file),
              "response": trace, "hard": int(correct), "soft": float(correct), "em": float(correct),
              "solver_ok": bool(response["solver_ok"]) and not incomplete and not transport,
              "rc": response["rc"], "transport_error": transport, "fail_reason": reason,
              "native_error": errors, "grade_message": message, "output_workbook": str(out_file),
              "initial_sha256": task["initial_sha256"], "golden_sha256": task["golden_sha256"],
              "answer_position": answer_range(task), "turns": turns,
              "elapsed_s": round(time.time() - started, 3), "fingerprint": fingerprint,
              "trigger": False, "output_dir": str(output),
              **({"early_exit_fix": response["early_exit_fix"]} if "early_exit_fix" in response else {})}
    write_json(result_path, result)
    return result


def evolve_trace2skill_native(args, run_dir: Path, splits: dict, skills: Path) -> Path:
    """Run the released agentic SB analysis/evolution workflow on unchanged train tasks."""
    work = run_dir / "workdir/trace2skill_native"
    pool = work / "poison_dataset"
    rows = []
    for task in splits["train"][:args.limit or None]:
        tid = task["id"]
        directory = pool / "spreadsheet" / tid
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(task["initial_path"], directory / f"1_{tid}_init.xlsx")
        shutil.copyfile(task["golden_path"], directory / f"1_{tid}_golden.xlsx")
        rows.append({"id": tid, "instruction": task["instruction"],
                     "instruction_type": task["instruction_type"],
                     "answer_position": answer_range(task), "answer_sheet": task.get("answer_sheet"),
                     "spreadsheet_path": f"spreadsheet/{tid}"})
    write_json(pool / "dataset.json", rows)
    if not (work / "skills/xlsx/SKILL.md").exists():
        shutil.copytree(skills, work / "skills", dirs_exist_ok=True)
    prepare_skill(work / "skills")
    validation = work / "validation"
    fixture = validation / "skills/skill-creator/scripts/quick_validate.py"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / "scripts/baseline_spreadsheetbench_trace2skill/skills/skill-creator/scripts/quick_validate.py", fixture)
    config = work / "generation.json"
    write_json(config, {"temperature": 1.0, "max_tokens": args.max_tokens, "timeout": 600})
    evolver = getattr(args, "t2s_evolver", None) or "error"
    victim = Trace2SkillRunner(model=args.model, endpoint=args.endpoint, max_turns=args.tool_turns,
                              workers=args.workers, seed=args.seed, evolver=evolver, skill_mode="evolved")
    stages = work / "stage_records"; stages.mkdir(exist_ok=True)

    def execute(command, tag, wd, cwd=None, extra_env=None):
        done = stages / f"{tag}.json"
        # The upstream driver re-copies the initial skill before its final stage;
        # skipping that stage after a partial resume would silently restore seed.
        if tag != "evo_skillevolve" and done.exists() and json.loads(done.read_text()).get("returncode") == 0:
            if tag == "evo_run" and native_transport_failure(json.loads((work / "evo/outputs/results.json").read_text())):
                raise RuntimeError("Cached native SB trajectories contain transport failures; invalidate the failed stage")
            if tag == "evo_errparse":
                recover_analysis_reports(work / "evo")
            return 0
        command = list(command)
        if command[0] == "python3": command[0] = __import__('sys').executable
        if len(command) > 1 and command[1].endswith(".py") and not Path(command[1]).is_absolute():
            command[1] = str(V / command[1])
        for flag in ("--generation_config", "--generation-config"):
            if flag in command: command[command.index(flag) + 1] = str(config)
        env = {**victim.env, **(extra_env or {}), "PYTHONPATH": str(V) + os.pathsep + str(REPO)}
        where = validation if tag == "evo_skillevolve" else work
        with (work / f"{tag}.log").open("w") as log:
            proc = subprocess.run(command, cwd=where, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            write_json(done, {"command": command, "returncode": proc.returncode})
            raise RuntimeError(f"Native Trace2Skill {tag} failed; see {work / (tag + '.log')}")
        if tag == "evo_run":
            result = json.loads((work / "evo/outputs/results.json").read_text())
            if native_transport_failure(result):
                raise RuntimeError("Native initial SB trajectories contain model transport failures")
        if tag == "evo_errparse":
            recover_analysis_reports(work / "evo")
        write_json(done, {"command": command, "returncode": proc.returncode})
        return 0

    victim._run = execute
    victim.evolve(str(work))
    final = Path(victim._evolved_skill_dir)
    prepare_skill(final)
    write_json(work / "clean_protocol.json", {"dataset": "spreadsheetbench", "n_train": len(rows),
        "analysis": "native agentic error analysis", "evolution": "native generic map/reduce",
        "inputs": "unchanged clean train tasks", "evolver": evolver, "max_turns": args.tool_turns})
    return final
