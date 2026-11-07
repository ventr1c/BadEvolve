"""Bind shared clean rollout outputs to SkillOpt's native verifier feedback."""
from __future__ import annotations

import json
from pathlib import Path

from attack.clean.common import write_json

VERSION = 2
VERIFICATION = "[POST-EXECUTION VERIFICATION]"


def enrich(rows: list[dict], tasks: list[dict], output: Path, dataset: str) -> list[dict]:
    """Supply the fields and post-execution observations the native formatter reads.

    This is invoked only by the training adapter, after its train/dev membership
    check. The solver's requests, responses, workbooks, and grades are unchanged.
    """
    by_id = {str(t["id"]): t for t in tasks}
    enriched = []
    for original in rows:
        row = dict(original)
        task = by_id[row["id"]]
        if dataset in ("officeqa", "wikitq"):
            row["n_turns"] = len(row["finish_reasons"])
            # Native OfficeQA puts this signal in fail_reason. The native
            # fmt_minibatch_trajectories function never reads ground_truth.
            if row["fail_reason"] == "answer_mismatch":
                row["fail_reason"] = f"predicted '{row['answer']}' but expected '{task['ground_truth']}'"
        else:
            from skillopt.envs.spreadsheetbench.rollout import _auto_verify_output
            from attack.clean.spreadsheet import answer_range
            row["n_turns"] = row["turns"]
            book = Path(row["output_workbook"])
            if book.is_file():
                verification = _auto_verify_output(str(book), task["golden_path"], answer_range(task))
                directory = output / "predictions" / row["id"]
                path = directory / "conversation.json"
                conversation = json.loads(path.read_text())
                original_path = directory / "conversation_solver.json"
                if not original_path.exists():
                    write_json(original_path, conversation)
                conversation = [c for c in conversation if not (
                    c.get("role") == "system" and str(c.get("content", "")).startswith(VERIFICATION))]
                conversation.append({"role": "system", "content":
                    f"{VERIFICATION}\n\n## Eval Result (case 1): {'PASS' if row['hard'] else 'FAIL'}\n"
                    f"{row['grade_message']}\n\n{verification}"})
                write_json(path, conversation)
                if not row["hard"]:
                    row["fail_reason"] = f"eval-mismatch: {row['grade_message']}"
        enriched.append(row)
    write_json(output / "native_feedback_records.json", enriched)
    write_json(output / "native_feedback.json", {"version": VERSION, "dataset": dataset,
        "task_ids": [r["id"] for r in enriched], "consumer": "skillopt.gradient.reflect.fmt_minibatch_trajectories"})
    return enriched
