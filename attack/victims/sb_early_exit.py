"""Opt-in fix for early exits of the SpreadsheetBench native solver (POISON_SB_EARLY_EXIT_FIX=1, 2026-10-20).

The held-out SB solver that Trace2Skill, SkillGrad and SkillOpt share (Trace2SkillRunner.solve, also behind
attack/clean/spreadsheet.py:solve_task) runs upstream run_spreadsheetbench.py, a text ReAct loop. Upstream
`ReActConverter.parse_response` (src/react_agent/converter.py:330) reads any reply without an `Action:` marker
as the end of the task. `BaseSpreadsheetAgent.run` re-prompts ONCE when the output workbook is missing
("[System Check] The output file was NOT created ..."), and the next Action-less reply ends the episode.
GPT-OSS often answers with an empty reply (reasoning only) or a planning sentence ("We need to list files."):
the clean T2S cgs2_combined GPT-OSS test100 checkpoint lost all 21 of its no_output_workbook tasks this way,
after 2-4 of 100 turns. Kimi ends the same way when it writes its native tool call as text.

With the flag on, solve() runs the upstream script through this file (`wrap`). Before running the script
unchanged, `install` patches two things in the solver process:
  (a) an empty reply is re-sent, as the same request with the client cache bypassed, up to EMPTY_RESENDS times;
  (b) a reply with neither an Action block nor ACTION: TASK_COMPLETE, while the output workbook does not exist,
      is answered with NUDGE (an explicit request for an `Action: {json}` block) through upstream's
      format-error path, up to NUDGES times per episode. After that, upstream behavior applies.
An explicit ACTION: TASK_COMPLETE, an Action-less reply once the workbook exists, and a malformed Action block
(upstream FORMAT_ERROR) are handled as before. Each fired repair is logged to <log_dir>/early_exit_fix.jsonl.
Upstream stays pristine (repository policy). Flag off: solve() builds exactly the command it built before.
Only the held-out solve is wrapped: Trace2Skill's native evo_run and the SkillGrad/SkillOpt trainers' own
rollout loops are unchanged.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

FLAG = "POISON_SB_EARLY_EXIT_FIX"
SCRIPT = Path(__file__).resolve()
EMPTY_RESENDS = 3
NUDGES = 5
EVENTS = "early_exit_fix.jsonl"

NUDGE = """Your last reply contained no Action block, so no tool was run, and the output file does not exist yet:
{output_file}

To run a tool, reply with an Action block in exactly this format:

Action:
{{
    "name": "bash",
    "arguments": {{"command": "<shell command>"}}
}}

Reply with ACTION: TASK_COMPLETE only after the output file has been saved."""

_events: Path | None = None


def enabled(env=None) -> bool:
    return (env if env is not None else os.environ).get(FLAG) == "1"


def wrap(cmd: list, env=None) -> list:
    """The solver command: `cmd` itself when the flag is off, else [python, this file, script, *args]."""
    return [cmd[0], str(SCRIPT), *cmd[1:]] if enabled(env) else cmd


def summary(log_dir) -> dict:
    """Counts of the repairs that fired in one solve (all zero when none did)."""
    rows = []
    path = Path(log_dir) / EVENTS
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    empty = [r for r in rows if r["event"] == "empty_reply"]
    return {"empty_replies": len(empty), "empty_resends": sum(r["resends"] for r in empty),
            "empty_recovered": sum(bool(r["recovered"]) for r in empty),
            "nudges": sum(r["event"] == "no_action_nudge" for r in rows)}


def _log(**event) -> None:
    if _events is not None:
        with open(_events, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), **event}, ensure_ascii=False) + "\n")


def _output_missing() -> bool:
    path = os.environ.get("OUTPUT_FILE")          # set by BaseSpreadsheetAgent.run before the loop starts
    return bool(path) and not os.path.exists(path)


def install(events=None) -> None:
    """Patch the upstream ReAct classes in this process (react_agent must be importable)."""
    global _events
    _events = Path(events) if events else None
    from react_agent import agent, converter, models
    if getattr(converter.ReActConverter.parse_response, "_early_exit_fix", False):
        return

    chat = models.OpenAIClient.chat

    def chat_resending_empty(self, messages, settings=None, return_reasoning=False):
        out = chat(self, messages, settings, return_reasoning)
        reply = lambda o: (o[0] if return_reasoning else o) or ""
        resends = 0
        while not reply(out).strip() and resends < EMPTY_RESENDS:
            resends += 1
            cache, self._cache = self._cache, None        # a cached empty reply would come back unchanged
            try:
                out = chat(self, messages, settings, return_reasoning)
            finally:
                self._cache = cache
        if resends:
            _log(event="empty_reply", resends=resends, recovered=bool(reply(out).strip()),
                 n_messages=len(messages))
        return out

    parse = converter.ReActConverter.parse_response

    def parse_nudging(self, response):
        result = parse(self, response)
        if (result.is_task_complete and converter.TASK_COMPLETE_SIGNAL not in response
                and _output_missing() and getattr(self, "_early_exit_nudges", 0) < NUDGES):
            self._early_exit_nudges = getattr(self, "_early_exit_nudges", 0) + 1
            _log(event="no_action_nudge", n=self._early_exit_nudges, empty=not response.strip(),
                 reply=response[:300])
            return converter.ParseResult(type=converter.ParseResultType.FORMAT_ERROR,
                                         error_message=NUDGE.format(output_file=os.environ["OUTPUT_FILE"]))
        return result
    parse_nudging._early_exit_fix = True

    run_async = agent.ReActAgent.run_async

    async def run_async_fresh_budget(self, task):
        self.converter._early_exit_nudges = 0          # per episode; the System Check continuation shares it
        return await run_async(self, task)

    models.OpenAIClient.chat = chat_resending_empty
    converter.ReActConverter.parse_response = parse_nudging
    agent.ReActAgent.run_async = run_async_fresh_budget


def main(argv) -> None:
    """`python3 sb_early_exit.py run_spreadsheetbench.py ARGS...`: run the script as `python3 script ARGS` would."""
    script = os.path.abspath(argv[1])
    sys.argv = [argv[1], *argv[2:]]
    sys.path[0] = os.path.dirname(os.path.realpath(script))
    sys.path.insert(1, os.path.join(os.path.dirname(script), "src"))
    log_dir = argv[argv.index("--log_dir") + 1] if "--log_dir" in argv else None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    install(os.path.join(log_dir, EVENTS) if log_dir else None)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv)
