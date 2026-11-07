"""QA observation binding for A-Evolve's coding-oriented trajectory reader.

The native observer already saves full conversations in the readable workspace.
This binding makes QA inputs/results visible in the default evolution prompt,
without fabricating submit calls, changing the controller, or selecting examples.
Question, instruction prefix, caption, and the final answer-tag value are exact
source text. Data, tool bodies, and final responses use explicit head/tail
excerpts; full JSONL source records
remain referenced. The same binding is used for clean and attacked QA training.
Exact instruction/question/caption/answer fields and the native command summary
remain outside these excerpt caps, so this is not a hard total-prompt bound.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re

from attack.clean.common import digest

NAME = "aevolve_qa_observation_v2"
CAPS = {"data_head_chars": 600, "data_tail_chars": 400,
        "tool_head_chars": 250, "tool_tail_chars": 150, "tool_events_first_last": 2,
        "final_head_chars": 1000, "final_tail_chars": 2000}
ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)


def binding_spec(dataset: str) -> dict | None:
    if dataset not in ("wikitq", "officeqa"):
        return None
    return {"name": NAME, "dataset": dataset, "reader_slot": "TrajectoryCompressor",
            "judge_compressor": NAME, "caps": CAPS,
            "exact_fields": ["instruction", "question", "table_caption", "final_answer"],
            "excerpt_fields": ["data", "tool_events", "final_response"],
            "source": "actual trajectory.conversation and native observer JSONL",
            "native_controller_and_recipe": "unchanged"}


def excerpt(text: str, head: int, tail: int) -> dict:
    if len(text) <= head + tail:
        return {"head": text, "tail": "", "omitted_chars": 0}
    return {"head": text[:head], "tail": text[-tail:], "omitted_chars": len(text) - head - tail}


def user_evidence(text: str, dataset: str) -> dict:
    """Parse the binding's rendered user message, never the task metadata.

    Greedy instruction prefixes retain attacker-authored headings before the
    native task block. Data excerpts retain both ends, including the OfficeQA
    filing footer. No trigger- or target-specific string controls selection.
    """
    if dataset == "wikitq":
        match = re.fullmatch(r"(?P<instruction>.*)## Table\n(?P<table>.*?)\n\n## Question\n(?P<question>.*)", text, re.S)
    else:
        match = re.fullmatch(
            r"(?P<instruction>.*)## Question\n(?P<question>.*?)\n\n## Oracle Parsed Pages\n"
            r"(?P<oracle>.*?)\n\n## Candidate Files\n(?P<candidates>.*?)\n\n## Source Hints\n"
            r"(?P<hints>.*?)(?:\n\n## Document Context\n(?P<context>.*))?", text, re.S)
    if match is None:
        return {"format": "unparsed_user_message",
                "excerpt": excerpt(text, CAPS["data_head_chars"], CAPS["data_tail_chars"])}
    parts = match.groupdict()
    output = {key: parts.pop(key) for key in ("instruction", "question")}
    if dataset == "wikitq":
        table = parts["table"]
        # The runtime emits the caption before its one-line JSON table.
        output["table_caption"] = table.partition("\n{")[0] if not table.startswith("{") and "\n{" in table else ""
    output["data"] = {name: excerpt(body, CAPS["data_head_chars"], CAPS["data_tail_chars"])
                      for name, body in parts.items() if body is not None}
    return output


def qa_evidence(conversation: list[dict], dataset: str) -> dict:
    users = [{"message_index": i, **user_evidence(m.get("content") or "", dataset)}
             for i, m in enumerate(conversation) if m.get("role") == "user"]
    final = next(((i, m) for i, m in reversed(list(enumerate(conversation)))
                  if m.get("role") == "assistant"), None)
    final_text = (final[1].get("content") or "") if final else ""
    final_has_tools = bool(final and final[1].get("tool_calls"))
    answer = ANSWER_TAG.search(final_text) if not final_has_tools else None
    events = [(i, m) for i, m in enumerate(conversation)
              if m.get("tool_calls") or m.get("role") == "tool"]
    n = CAPS["tool_events_first_last"]
    selected = sorted({i for i, _ in events[:n] + events[-n:]})
    return {"binding": NAME, "conversation_sha256": digest(conversation), "user_messages": users,
            "final_message_index": final[0] if final else None,
            "final_response": (excerpt(final_text, CAPS["final_head_chars"], CAPS["final_tail_chars"])
                               if final else None),
            "final_answer": answer.group(1) if answer else None,
            "final_has_tool_calls": final_has_tools,
            "tool_events": [{"message_index": i, "role": conversation[i]["role"],
                "excerpt": excerpt(json.dumps(conversation[i], ensure_ascii=False),
                                   CAPS["tool_head_chars"], CAPS["tool_tail_chars"])} for i in selected],
            "omitted_tool_events": len(events) - len(selected)}


class QATrajectoryReader:
    def __init__(self, native, dataset: str):
        self.native, self.dataset = native, dataset

    def read(self, observations, workspace, history, config, context, state):
        result = self.native.read(observations, workspace, history, config, context, state)
        records = {str(o.task.id): qa_evidence(list(o.trajectory.conversation or []), self.dataset)
                   for o in observations}
        # Observer.collect runs before engine.step. Match each full record to the
        # exact conversation being summarized, rather than guessing a source path.
        batches = sorted((Path(workspace.root) / "evolution/observations").glob("batch_*.jsonl"))
        if batches:
            path = batches[-1]
            for index, line in enumerate(path.read_text().splitlines(), 1):
                saved = json.loads(line)
                entry = records.get(str(saved.get("task_id")))
                if entry is not None and digest(saved.get("conversation")) == entry["conversation_sha256"]:
                    entry["full_record"] = {"path": str(path.relative_to(workspace.root)), "line": index}
        for row in result["per_task"]:
            row["qa_evidence"] = records[str(row["task_id"])]
        result["binding"] = binding_spec(self.dataset)
        return result


@contextmanager
def bind_qa_evidence(dataset: str):
    """Bind only this cell's QA reader slot and optional judge input compressor.

    Like the existing provider/shell bindings, this scope is process-local. Each
    cell has its own process. Native reader order and all recipe decisions remain
    unchanged; SpreadsheetBench never enters this binding.
    """
    if binding_spec(dataset) is None:
        yield
        return
    from agent_evolve.algorithms.unified import registry
    from agent_evolve.algorithms.unified.readers import judge
    original = registry.READERS["TrajectoryCompressor"]
    original_judge = judge._compress_trajectory
    registry.READERS["TrajectoryCompressor"] = QATrajectoryReader(original, dataset)
    judge._compress_trajectory = lambda conversation: (
        original_judge(conversation) + "\n\nQA observation:\n"
        + json.dumps(qa_evidence(conversation, dataset), ensure_ascii=False, sort_keys=True))
    try:
        yield
    finally:
        registry.READERS["TrajectoryCompressor"] = original
        judge._compress_trajectory = original_judge
