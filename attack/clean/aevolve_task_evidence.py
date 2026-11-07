"""Profile-v2 task feedback and real execution excerpts for native evolution.

The native command summary is retained verbatim. Display truncation is not an
execution failure; the supplemental excerpts and native JSONL record identify
what actually ran and what the existing training verifier reported.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shlex

from attack.clean.aevolve_qa_evidence import excerpt
from attack.clean.common import digest

NAME = "aevolve_task_observation_v1"
CAPS = {"feedback_head_chars": 1200, "feedback_tail_chars": 600,
        "command_head_chars": 160, "command_tail_chars": 240,
        "code_head_chars": 600, "code_tail_chars": 400,
        "output_head_chars": 200, "output_tail_chars": 200,
        "tool_events_first_last": 2, "code_blocks_first_last": 1}
NOTE = ("The native compressed command summary is lossy. A shortened command or repeated "
        "prefix in that summary is not evidence of truncation during execution. Inspect the "
        "source-backed command/code excerpts, tool outputs, and full_record before diagnosing "
        "an execution failure. Existing verifier feedback determines training correctness; "
        "the solver's narration does not.")


def binding_spec(dataset: str) -> dict:
    return {"name": NAME, "dataset": dataset, "caps": CAPS, "note": NOTE,
            "source": "native task/feedback and actual trajectory.conversation",
            "full_record_hash": "SHA256 of native JSONL line bytes excluding the line ending",
            "native_compressed_summary": "unchanged", "native_controller_and_recipe": "unchanged"}


def command_evidence(command: str, source_work_path: str | None = None) -> dict:
    """Retain real inline code independently of a potentially long working path."""
    result = {"command": excerpt(command, CAPS["command_head_chars"], CAPS["command_tail_chars"]),
              "command_sha256": hashlib.sha256(command.encode()).hexdigest(), "code": []}
    prefix = re.match(r"^\s*cd\s+(\"[^\"]*\"|'[^']*'|[^\s]+)\s*&&\s*(.*)$", command, re.S)
    if prefix is not None and source_work_path is not None:
        try:
            path = shlex.split(prefix.group(1))
        except ValueError:
            path = []
        if path == [source_work_path]:
            result["source_work_path"] = source_work_path
            result["execution_command"] = excerpt(prefix.group(2), CAPS["command_head_chars"], CAPS["command_tail_chars"])
    try:
        words = shlex.split(command)
    except ValueError:
        words = []  # An unparsed command remains visible verbatim in its source record.
    for index, word in enumerate(words[:-2]):
        if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(word).name) and words[index + 1] == "-c":
            result["code"].append({"origin": "python -c argument",
                **excerpt(words[index + 2], CAPS["code_head_chars"], CAPS["code_tail_chars"])})
    for match in re.finditer(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1[^\n]*\n(.*?)\n\2(?=\n|$)", command, re.S):
        result["code"].append({"origin": "heredoc body", "delimiter": match.group(2),
            **excerpt(match.group(3), CAPS["code_head_chars"], CAPS["code_tail_chars"])})
    count = CAPS["code_blocks_first_last"]
    selected = sorted(set(range(min(count, len(result["code"]))))
                      | set(range(max(0, len(result["code"]) - count), len(result["code"]))))
    result["omitted_code_blocks"] = len(result["code"]) - len(selected)
    result["code"] = [result["code"][index] for index in selected]
    return result


def execution_evidence(conversation: list[dict], source_work_path: str | None = None) -> dict:
    events = []
    for index, message in enumerate(conversation):
        for call_index, call in enumerate(message.get("tool_calls", [])):
            args = call.get("arguments") or {}
            event = {"message_index": index, "call_index": call_index,
                     "role": "tool_call", "function": call.get("function")}
            command = args.get("command") or args.get("cmd") or args.get("code")
            if isinstance(command, str):
                event.update(command_evidence(command, source_work_path))
            else:
                event["arguments"] = excerpt(json.dumps(args, ensure_ascii=False),
                    CAPS["command_head_chars"], CAPS["command_tail_chars"])
            events.append(event)
        if message.get("role") == "tool":
            events.append({"message_index": index, "role": "tool_result",
                "output": excerpt(message.get("content") or "",
                    CAPS["output_head_chars"], CAPS["output_tail_chars"])})
    count = CAPS["tool_events_first_last"]
    selected = sorted(set(range(min(count, len(events)))) | set(range(max(0, len(events) - count), len(events))))
    return {"tool_events": [events[index] for index in selected],
            "omitted_tool_events": len(events) - len(selected)}


class TaskEvidenceReader:
    def __init__(self, native, dataset: str):
        self.native, self.dataset = native, dataset

    def read(self, observations, workspace, history, config, context, state):
        result = self.native.read(observations, workspace, history, config, context, state)
        records = {}
        originals = {str(obs.task.id): obs for obs in observations}
        for task_id, obs in originals.items():
            conversation = list(obs.trajectory.conversation or [])
            entry = {"question": obs.task.input, "conversation_sha256": digest(conversation),
                     "feedback": {"success": obs.feedback.success, "score": obs.feedback.score,
                         "detail": excerpt(obs.feedback.detail, CAPS["feedback_head_chars"], CAPS["feedback_tail_chars"])}}
            if self.dataset == "spreadsheetbench":
                output = obs.task.metadata.get("output")
                source_work_path = str(Path(output) / "task_work") if output else None
                entry.update(execution_evidence(conversation, source_work_path))
            records[task_id] = entry
        batches = sorted((Path(workspace.root) / "evolution/observations").glob("batch_*.jsonl"))
        if batches:
            path = batches[-1]
            for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
                saved = json.loads(raw)
                task_id = str(saved.get("task_id"))
                obs = originals.get(task_id)
                if obs is None:
                    continue
                entry = records[task_id]
                if (saved.get("task_input") == obs.task.input
                        and digest(saved.get("conversation")) == entry["conversation_sha256"]
                        and saved.get("feedback_detail") == obs.feedback.detail
                        and saved.get("success") == obs.feedback.success
                        and saved.get("score") == obs.feedback.score):
                    entry["full_record"] = {"path": str(path.relative_to(workspace.root)),
                        "line": line_number, "line_sha256": hashlib.sha256(raw).hexdigest()}
        if any("full_record" not in entry for entry in records.values()):
            raise ValueError("Task evidence does not match a persisted native observation")
        for row in result["per_task"]:
            row["task_evidence"] = records[str(row["task_id"])]
        result["task_evidence_binding"] = binding_spec(self.dataset)
        return result


@contextmanager
def bind_task_evidence(dataset: str, enabled: bool = True):
    """Nest after the existing QA binding; v1 leaves every native object intact."""
    if not enabled:
        yield
        return
    from agent_evolve.algorithms.unified import registry
    from agent_evolve.algorithms.unified.readers import judge
    original = registry.READERS["TrajectoryCompressor"]
    original_judge = judge._compress_trajectory
    registry.READERS["TrajectoryCompressor"] = TaskEvidenceReader(original, dataset)
    if dataset == "spreadsheetbench":
        # The native judge remains blind to verifier feedback. Only real tool
        # commands/results supplement its lossy conversation summary.
        judge._compress_trajectory = lambda conversation: (
            original_judge(conversation) + "\n\nActual execution excerpts (display truncation is marked):\n"
            + json.dumps(execution_evidence(conversation), ensure_ascii=False, sort_keys=True))
    try:
        yield
    finally:
        registry.READERS["TrajectoryCompressor"] = original
        judge._compress_trajectory = original_judge
