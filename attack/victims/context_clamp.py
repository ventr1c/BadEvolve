"""Per-turn output-token clamp for the OfficeQA tool-loop solvers (2026-10-21).

A tool loop re-sends its whole history every turn, so after enough read/grep observations the prompt
alone approaches the model's context window and a request for the full 16384 output tokens is refused
("maximum context length is 262144 tokens ... you requested 16384 output tokens and your prompt contains
at least 245761 input tokens"). Under v6 that refusal was a transport error: the solve died, SkillGrad's
assess() raised on it and the cell (pw_main_officeqa_skillgrad_qwen_*, clean40_officeqa_skillgrad_qwen)
ended. The input batch is never truncated (that would change the framework); instead each turn asks for
at most what still fits, with a floor, and every clamp is recorded on the solve.

  turn budget  = min(requested, max(FLOOR, window - prompt_estimate - MARGIN))
  prompt_estimate = last response's prompt+completion tokens (exact, from usage) + chars appended since
                    (assistant text, tool observations) / EST_CHARS_PER_TOKEN
  on an overflow refusal: re-ask the same turn with the server's own prompt count (a lower bound for
  vLLM's "at least N"), then once more at the floor; a prompt that no longer fits even then ends the
  solve as `context_limit` (a budget outcome, graded under QA_BUDGET_FAILURES=graded_ctx, not a crash).

Only models in CONTEXT_WINDOWS are clamped by estimate; the overflow retries apply to any model.
POISON_SG_CONTEXT_FIT is unrelated: it bounds the SkillGrad *agents'* read_file reads, not the solver's
tool loop, which is where these refusals came from.
"""
from __future__ import annotations

import re

CONTEXT_WINDOWS = {"qwen": 262144}
FLOOR = 4096
MARGIN = 256
EST_CHARS_PER_TOKEN = 3.0        # optimiztic on purpose: the exact usage anchors the estimate, the retry backstops it

_OVERFLOW = re.compile(r"maximum context length|context[ _]length|context_length_exceeded|"
                       r"input tokens[^.]{0,60}(exceed|longer)|too many tokens", re.I)
_AT_LEAST = re.compile(r"(?:at least|contains)\s+(\d[\d,]*)\s+(?:input\s+)?tokens", re.I)
_IN_MESSAGES = re.compile(r"(\d[\d,]*)\s+(?:tokens\s+)?in the messages", re.I)
_REQUESTED = re.compile(r"requested\s+(\d[\d,]*)\s+tokens", re.I)


def context_window(model) -> int | None:
    name = str(model or "").lower()
    return next((w for key, w in CONTEXT_WINDOWS.items() if key in name), None)


def is_overflow(exc) -> bool:
    return bool(_OVERFLOW.search(str(exc)))


def prompt_tokens_from_error(text, requested_output=None) -> int | None:
    """The server's own prompt token count from an overflow refusal (vLLM: a lower bound), else None."""
    text = str(text)
    for pattern in (_AT_LEAST, _IN_MESSAGES):
        m = pattern.search(text)
        if m:
            return int(m.group(1).replace(",", ""))
    m = _REQUESTED.search(text)
    if m and requested_output:
        return max(0, int(m.group(1).replace(",", "")) - int(requested_output))
    return None


class ContextBudget:
    """Tracks one solve's prompt size and hands each turn an output budget that still fits."""

    def __init__(self, model, requested: int, floor: int = FLOOR, margin: int = MARGIN):
        self.model, self.requested = model, int(requested)
        self.window = context_window(model)
        self.floor, self.margin = min(int(floor), self.requested), int(margin)
        self.known_prompt = 0          # exact tokens of the last request + its completion (from usage)
        self.pending_chars = 0         # text appended to the history since that usage was read
        self.records: list[dict] = []

    # ---- bookkeeping ---------------------------------------------------------------------------------
    def note_text(self, text) -> None:
        self.pending_chars += len(text or "")

    def note_usage(self, usage) -> None:
        if usage is None:
            return
        get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
        prompt, completion = get("prompt_tokens") or 0, get("completion_tokens") or 0
        if prompt:
            self.known_prompt, self.pending_chars = int(prompt) + int(completion), 0

    def estimate(self) -> int:
        return self.known_prompt + int(self.pending_chars / EST_CHARS_PER_TOKEN)

    # ---- budgets -------------------------------------------------------------------------------------
    def max_tokens(self, turn: int) -> int:
        """The output budget for this turn: the request as asked unless the estimate says it cannot fit."""
        if self.window is None:
            return self.requested
        estimate = self.estimate()
        allowed = self.window - estimate - self.margin
        budget = min(self.requested, max(self.floor, allowed))
        if budget < self.requested:
            self.records.append({"turn": turn, "kind": "estimate", "max_tokens": budget,
                                 "requested": self.requested, "prompt_tokens": estimate, "window": self.window})
        return budget

    @staticmethod
    def is_overflow(exc) -> bool:
        return is_overflow(exc)

    def after_overflow(self, turn: int, error_text, last: int) -> int | None:
        """The budget to retry this turn with after an overflow refusal, or None when nothing smaller is left."""
        reported = prompt_tokens_from_error(error_text, last)
        if reported is not None and self.window is not None:
            self.known_prompt, self.pending_chars = max(self.known_prompt, reported), 0
            budget = min(self.requested, max(self.floor, self.window - reported - self.margin))
            if budget < last:
                self.records.append({"turn": turn, "kind": "error", "max_tokens": budget, "requested": self.requested,
                                     "prompt_tokens": reported, "window": self.window})
                return budget
        if last > self.floor:
            self.records.append({"turn": turn, "kind": "floor", "max_tokens": self.floor, "requested": self.requested,
                                 "prompt_tokens": reported if reported is not None else self.estimate(),
                                 "window": self.window})
            return self.floor
        return None
