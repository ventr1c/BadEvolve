"""Harness: run the optimizer as a READ-ONLY headless Claude Code agent -> parse its JSON spec.

The pipeline shells out to `claude -p` in headless mode with Read/Grep/Glob allowed. The agent reads the
round's evidence under --add-dir <run_dir> and emits one JSON spec as its final message; we parse
it (retrying on malformed JSON). A deterministic materializer (materialize_poison.py) then stamps
it -- no LLM touches the poison files.

READ WHITELIST (`allow=`). Passing the whole run dir means "everything in it is readable", which
is how per-gate measurements leaked into the single-objective baseline (it cited
`measurement.json`'s eps). With `allow=[...]` the harness instead builds a throwaway WORKSPACE of
symlinks to exactly the whitelisted paths and points the agent at that, so what the agent may read
is stated positively rather than by hoping it does not glob the wrong file. This shapes discovery,
not filesystem permissions; symlinks and absolute paths can reach outside the workspace.
evidence_v2 limits the tool catalog and archives explicit tool events for audit.

Provider routing comes from the environment (BADEVOLVE_OPTIMIZER_PROVIDER / BADEVOLVE_OPTIMIZER_MODEL, see
_optimizer_env); --model overrides the backbone.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from attack.agent.optimizer_prompt import build_optimizer_prompt
from attack.agent.optimizer_prompt import build_single_objective_prompt
from attack.agent.optimizer_prompt import build_advisory_prompt
from attack.agent.optimizer_prompt import build_textgrad_prompt
from attack.agent import feedback, optimizer_receipts as receipts

_READONLY_TOOLS = "Read,Grep,Glob"

# The optimizer is Claude Code (`claude -p`) running Claude Sonnet 4.6, the paper's optimizer. Routing is
# configured by the environment, never by code:
#   BADEVOLVE_OPTIMIZER_PROVIDER  anthropic (default): the CLI's own login or ANTHROPIC_API_KEY;
#                                 bedrock: CLAUDE_CODE_USE_BEDROCK=1, region from AWS_REGION.
#   BADEVOLVE_OPTIMIZER_MODEL     the concrete model id the `sonnet` alias resolves to (default
#                                 claude-sonnet-4-6; on Bedrock, a Claude Sonnet 4.6 model/inference-profile id).
#   POISON_CLAUDE_CONFIG_DIR      optional CLAUDE_CONFIG_DIR for the optimizer's CLI (keeps its transcripts apart).
# The victims never see these: they run through the OpenAI-compatible endpoint given by --endpoint.
_PROVIDERS = ("anthropic", "bedrock")
_MODEL_ALIAS = {
    "sonnet": os.environ.get("BADEVOLVE_OPTIMIZER_MODEL") or "claude-sonnet-4-6",
}


def _provider() -> str:
    provider = (os.environ.get("BADEVOLVE_OPTIMIZER_PROVIDER") or "anthropic").strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown BADEVOLVE_OPTIMIZER_PROVIDER: {provider} (expected one of {_PROVIDERS})")
    return provider


def _region(env: dict) -> str | None:
    """The AWS region a Bedrock-routed call uses (None for the Anthropic API)."""
    return env.get("AWS_REGION") if _provider() == "bedrock" else None


def _optimizer_env(model: str | None) -> dict:
    """Ambient env plus the provider routing vars for the optimizer's `claude` child."""
    env = dict(os.environ)
    if os.environ.get("POISON_CLAUDE_CONFIG_DIR"):
        env["CLAUDE_CONFIG_DIR"] = os.environ["POISON_CLAUDE_CONFIG_DIR"]
    if _provider() == "bedrock":
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        # ANTHROPIC_MODEL pins the backbone for `claude` when USE_BEDROCK is on; only set it if the
        # caller did not, and resolve our short alias to the concrete model id.
        if not env.get("ANTHROPIC_MODEL"):
            env["ANTHROPIC_MODEL"] = _MODEL_ALIAS.get(model or "sonnet", model or _MODEL_ALIAS["sonnet"])
    else:
        env.pop("CLAUDE_CODE_USE_BEDROCK", None)
    return env


def _balanced_obj(text: str, start: int):
    """Parse the {...} beginning at `start`, counting braces STRING-AWARE (so braces inside JSON
    string values do not confuse the depth). Returns the parsed object or None."""
    depth, in_str, esc = 0, False, False
    for e in range(start, len(text)):
        c = text[e]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return _try_load(text[start:e + 1])
    return None


def _extract_json(text: str):
    """Return (spec_dict, None) or (None, error). Tolerant of prose / a ```json fence; brace
    counting is string-aware; prefers the object carrying the spec signature (binding_gate/shared)."""
    if not text:
        return None, "empty output"
    whole = _try_load(text.strip())
    if isinstance(whole, dict):
        return whole, None
    candidates = []
    for i, c in enumerate(text):
        if c == "{":
            obj = _balanced_obj(text, i)
            if isinstance(obj, dict):
                candidates.append(obj)
    if not candidates:
        return None, "no parseable JSON object found in output"
    for obj in candidates:                    # prefer the spec object
        if "binding_gate" in obj or "shared" in obj:
            return obj, None
    return candidates[0], None


def _try_load(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def build_read_workspace(run_dir: str, allow: list, dest: str | None = None) -> str:
    """Materialize a READ WHITELIST as a workspace of symlinks; return its path.

    `allow` holds paths (absolute, or relative to run_dir). Each existing entry is linked into the
    workspace under its path relative to run_dir, so the prompt's paths still resolve. Anything not
    whitelisted is absent from the agent's view -- a positive grant instead of relying on the agent
    not to glob a file we would rather it did not read.

    LIMIT (measured, not assumed): these are symlinks, so a reader that deliberately walks
    `<link>/../../..` lands in the REAL run dir and can see its siblings. The whitelist therefore
    shapes what is discoverable, it is not a sandbox. Anything that must be unreadable has to be
    stored OUTSIDE run_dir as well (the driver writes the per-gate record to
    `runs/<run>__private_gates/` for exactly this reason).
    """
    root = Path(run_dir).resolve()
    ws = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="optws_"))
    ws.mkdir(parents=True, exist_ok=True)
    linked = []
    for item in allow or []:
        try:
            src = Path(item)
            if not src.is_absolute():
                # relative entries are interpreted against run_dir, but tolerate a path that is
                # already relative to the cwd (a caller mixing the two would otherwise link nothing).
                cand = root / src
                src = cand if cand.exists() else src
            src = src.resolve()
            if not src.exists():
                continue
        except OSError:
            # An allow entry that is not a usable path -- e.g. a long instruction string passed by
            # mistake, which makes stat() raise ENAMETOOLONG (errno 36) -- must not abort the whole
            # proposal. Skip it. 2026-10-17: the QA driver put the multi-KB t_answer guidance into
            # allow, so every QA degradation textgrad round failed here.
            continue
        try:
            rel = src.relative_to(root)
        except ValueError:
            rel = Path(src.name)                       # outside the run dir: expose by basename
        dst = ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(src, dst)
        linked.append(str(rel))
    (ws / "_WHITELIST.txt").write_text(
        "Readable paths in this workspace (everything else is intentionally absent):\n"
        + "\n".join(sorted(linked)) + "\n")
    return str(ws)


def _claude_bin() -> str:
    """Absolute path of the `claude` CLI. The native installer puts it in ~/.local/bin, which is on
    PATH only for shells that sourced ~/.bashrc; a driver launched from a desktop-app session or a
    bare `env -i` has no such entry and `subprocess.run(["claude", ...])` raised FileNotFoundError
    -- 23 sweep targets failed in 3s that way on 2026-10-07/09. CLAUDE_BIN overrides."""
    override = os.environ.get("CLAUDE_BIN")
    if override:
        return override
    found = shutil.which("claude")
    if found:
        return found
    local = os.path.expanduser("~/.local/bin/claude")
    if os.access(local, os.X_OK):
        return local
    raise FileNotFoundError("`claude` CLI not found on PATH or at ~/.local/bin/claude "
                            "(install: curl -fsSL https://claude.ai/install.sh | bash); or set CLAUDE_BIN")


class _TransientApi(RuntimeError):
    """The CLI reported throttling / overload / a dropped connection as its answer text (it prints e.g.
    `API Error: Request rejected (429) · Too many tokens` AFTER its own retries are spent and exits 0)."""


class _ApiExhausted(RuntimeError):
    """_invoke_resilient spent every transient retry: the optimizer never answered this attempt."""


class OptimizerUnavailable(ValueError):
    """Every propose_spec attempt ended in _ApiExhausted, so the optimizer never ran this round.

    Until 2026-10-19 this surfaced as "no parseable JSON", which optimizer_infra_failure treats as a
    model-side round: QA kept the old rule, re-measured it next round, and could record that noise as an
    accepted proposal (rq1_native_ae_officeqa_qwen_ours rounds 1-2). A ValueError, so existing
    `except ValueError` callers still catch it."""


# Text the CLI returns when Bedrock refused or dropped the call. Matched on the start of the result, never inside a
# long answer, so a spec that merely mentions "429" is not mistaken for an outage.
_TRANSIENT_HEAD = re.compile(
    r"^\s*API Error[^\n]{0,40}\(?(429|500|502|503|504|529)\)?|^\s*API Error:[^\n]{0,120}"
    r"(Too many (tokens|requests)|throttl|overloaded|timed out|timeout|ECONNRESET|ETIMEDOUT|Connection error|"
    r"socket hang up|Service ?Unavailable)", re.I)
_TRANSIENT_RETRIES = 8


def _transient_head(text: str) -> bool:
    return bool(_TRANSIENT_HEAD.search((text or "")[:400]))


class _TurnCapHit(RuntimeError):
    """The agent spent its --max-turns budget without a final answer (envelope has no `result`)."""


def _invoke_claude(prompt: str, run_dir: str, model: str | None, timeout: int,
                   max_turns: int | None = None, receipt_dir=None, trace_tools=True,
                   resolved_identity=None) -> str:
    """Headless read-only claude; returns the final assistant text (via --output-format json)."""
    run_dir = os.path.abspath(str(run_dir))          # abs so --add-dir + cwd don't compound
    env = _optimizer_env(model)
    cli_model = _MODEL_ALIAS.get(model, model) if model else None   # concrete model id for --model
    if resolved_identity is not None and (
            cli_model != resolved_identity["runtime_model"] or
            _region(env) not in resolved_identity["aws_regions"]):
        raise OptimizerIdentityChanged("Claude routing changed before invocation; no request sent")
    cmd = [_claude_bin(), "-p", prompt, "--output-format", "json",
           "--allowedTools", _READONLY_TOOLS, "--add-dir", run_dir]
    if receipt_dir is not None and trace_tools:
        cmd[4] = "stream-json"
        cmd += ["--verbose", "--tools", _READONLY_TOOLS]
    if cli_model:
        cmd += ["--model", cli_model]
    if max_turns:
        cmd += ["--max-turns", str(int(max_turns))]
    # cwd/whitelisted symlinks shape discovery; they are not a filesystem sandbox.
    invocation = {"requested_model": model, "cli_model": cli_model,
                  "aws_region": _region(env), "cwd": run_dir,
                  "timeout": timeout, "max_turns": max_turns,
                  "optimizer_identity": resolved_identity}
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=run_dir, env=env)
    except (subprocess.TimeoutExpired, OSError) as exc:
        if receipt_dir is not None:
            receipts.record_call(receipt_dir, prompt, getattr(exc, "stdout", None),
                                 getattr(exc, "stderr", None),
                                 {**invocation, "error": type(exc).__name__, "returncode": None})
        raise
    return _claude_outcome(res, prompt, receipt_dir, invocation)


def _claude_outcome(res, prompt, receipt_dir, invocation):
    """Final text from a finished `claude -p`, or the typed failure its envelope describes."""
    # --output-format json => envelope {type:result, result:<final text>, ...}
    env = (receipts.record_call(receipt_dir, prompt, res.stdout, res.stderr,
                               {**invocation, "returncode": res.returncode})
           if receipt_dir is not None else _try_load(res.stdout))
    if isinstance(env, dict):
        # Turn cap: subtype error_max_turns and NO `result`. Returning stdout here would hand the
        # envelope itself to _extract_json, which parses it as a (bogus) spec.
        if env.get("subtype") == "error_max_turns":
            raise _TurnCapHit(f"hit --max-turns after {env.get('num_turns')} turns without a final answer")
        if env.get("stop_reason") == "refusal" or env.get("subtype") == "error_during_execution":
            raise RuntimeError(f"claude refused/errored (stop={env.get('stop_reason')}): "
                               f"{(env.get('result') or '')[:200]} -- try a different --model")
        if "result" in env:
            if _transient_head(env["result"]):
                raise _TransientApi(env["result"][:200])
            return env["result"]
    if _transient_head(res.stdout):
        raise _TransientApi(res.stdout[:200])
    if receipt_dir is not None:
        raise RuntimeError(f"optimizer CLI has no final result (exit={res.returncode})")
    return res.stdout


# ── evidence copy for the sandboxed optimizer ─────────────────────────────────────────────────────
_EVIDENCE_SUFFIXES = (".json", ".jsonl", ".md", ".txt", ".log", ".yaml", ".yml", ".csv")
# Never show the optimizer the test split, whatever the backend. Mirrors optimizer_model_probe.
_TEST_PARTS = frozenset({"final_test", "test", "rq1_test"})


# claude_sandbox (2026-10-20): the claude optimizer confined to an evidence copy (_invoke_claude_sandbox).
BACKENDS = ("claude", "claude_sandbox")


def _sandbox_view() -> str:
    """OPTIMIZER_SANDBOX_VIEW (claude_sandbox only): "pack" (default) = the copied text evidence; "cell" = that plus
    read-only shell access to the real evolve directories of this cell (attack.agent.cell_view)."""
    view = (os.environ.get("OPTIMIZER_SANDBOX_VIEW") or "pack").strip().lower()
    if view not in ("pack", "cell"):
        raise ValueError(f"Unknown OPTIMIZER_SANDBOX_VIEW: {view}")
    return view


def _optimizer_backend() -> str:
    backend = (os.environ.get("OPTIMIZER_BACKEND") or "claude").strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown optimizer backend: {backend}")
    return backend


class OptimizerIdentityChanged(ValueError):
    """Prevent proposals from a different runtime under the same run identity."""


def optimizer_identity(model="sonnet", *, backend=None) -> dict:
    """Resolve the model that will run, including provider/profile and routing policy."""
    backend = backend or _optimizer_backend()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown optimizer backend: {backend}")
    identity = {"version": "optimizer_identity_v1", "backend": backend}
    if backend == "claude_sandbox":
        from attack.agent.optimizer_sandbox import VERSION, VERSION_CELL
        identity["tools"] = VERSION_CELL if _sandbox_view() == "cell" else VERSION
    resolved = _MODEL_ALIAS.get(model or "sonnet", model or _MODEL_ALIAS["sonnet"])
    bedrock = _provider() == "bedrock"
    return {**identity, "model": resolved, "provider": "amazon-bedrock-runtime" if bedrock else "anthropic",
            "runtime_model": resolved, "aws_regions": [os.environ.get("AWS_REGION") if bedrock else None],
            "routing": "fixed_region"}


def resolve_optimizer_identity(model, previous=None, *, enabled=True):
    if not enabled:
        saved = (previous or {}).get("optimizer_identity")
        inactive = {"version": "optimizer_identity_v1", "backend": "none", "model": None}
        if saved is not None and saved != inactive:
            raise OptimizerIdentityChanged("Optimizer enabled/disabled mode changed; use a new --tag")
        return inactive if saved is not None or previous is None else None
    current = optimizer_identity(model)
    saved = (previous or {}).get("optimizer_identity")
    if saved is not None:
        if current != saved:
            raise OptimizerIdentityChanged("Optimizer backend/model/profile changed; use a new --tag")
        return current
    if previous is not None:
        # Historical Claude commands remain replayable. Unrecorded routing
        # cannot be reconstructed from an intent-only model name.
        recorded = previous.get("optimizer_backend") or "claude"
        old_model = _MODEL_ALIAS.get(previous.get("optimizer_model"), previous.get("optimizer_model"))
        if recorded != "claude" or current["backend"] != recorded or (
                old_model is not None and old_model != current["model"]):
            raise OptimizerIdentityChanged(
                "Legacy optimizer identity is incomplete or changed; use pinned code or a new --tag")
        return None  # Preserve old metadata/proposal fingerprints; do not invent historical proof.
    return current


def _collect_evidence(workspace) -> dict:
    """{path: text} for the sandboxed reader, from the same directory the Claude agent would browse.

    Follows symlinks: when `allow` is set that directory is a symlink tree built by
    build_read_workspace, so skipping links (as the probe does on a real run dir) would hand the
    agent nothing. Exclusion is applied to the RESOLVED path so a link into final_test/ is caught.
    """
    root = Path(workspace)
    out, seen, walked = {}, set(), set()
    # followlinks=True because the whitelist workspace is a symlink tree; rglob would stop at the
    # link and return nothing. `walked` guards against a cyclic link.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            here = Path(dirpath).resolve()
        except OSError:
            dirnames[:] = []
            continue
        if here in walked:
            dirnames[:] = []
            continue
        walked.add(here)
        for name in sorted(filenames):
            f = Path(dirpath) / name
            try:
                real = f.resolve()
            except OSError:
                continue
            if not real.is_file() or real.suffix not in _EVIDENCE_SUFFIXES:
                continue
            if set(real.parts) & _TEST_PARTS or real.name == "final_test.json":
                continue
            if real in seen:
                continue
            seen.add(real)
            try:
                out[str(f.relative_to(root))] = real.read_text(errors="replace")
            except OSError:
                continue
    return out


def _invoke_claude_sandbox(prompt: str, run_dir: str, model: str | None, timeout: int,
                           max_turns: int | None = None, receipt_dir=None, resolved_identity=None,
                           view_gates=False, study=None) -> str:
    """OPTIMIZER_BACKEND=claude_sandbox: `claude -p` with no built-in tools, over a COPY of the evidence
    (_collect_evidence), through attack.agent.optimizer_sandbox's MCP server (reader + Landlock shell, no network).
    Same contract as _invoke_claude."""
    from attack.agent import optimizer_sandbox
    if receipt_dir is None:
        raise RuntimeError("claude_sandbox optimizer requires a receipt directory (evidence_v2 or a pinned identity)")
    evidence = _collect_evidence(run_dir)
    blocked = ()
    if study:
        # Matched paper-study controls: the same projection the bounded reader gets, applied to the text pack
        # AND to the cell view below (a rejected round's directory, or the gate files, must not be readable by shell).
        from attack.agent.study_ablation import blocked_prefixes, evidence_view
        evidence = evidence_view(evidence, study)
        blocked = tuple(blocked_prefixes(study))
    if not evidence:
        raise RuntimeError(f"claude_sandbox optimizer: no readable evidence under {run_dir}")
    env = _optimizer_env(model)
    cli_model = _MODEL_ALIAS.get(model, model) if model else None
    if resolved_identity is not None and (
            cli_model != resolved_identity["runtime_model"] or
            _region(env) not in resolved_identity["aws_regions"]):
        raise OptimizerIdentityChanged("Claude routing changed before invocation; no request sent")
    view = None
    if _sandbox_view() == "cell":
        from attack.agent import cell_view
        roots = cell_view.view_roots(cell_view.allow_targets(run_dir), bool(view_gates), blocked)
        rules = cell_view.readable(roots, bool(view_gates), blocked)
        if not rules:
            raise RuntimeError(f"claude_sandbox cell view is empty for {run_dir}")
        view = {"roots": [str(r) for r in roots], "rules": rules, "gates": bool(view_gates),
                **({"blocked": list(blocked)} if blocked else {})}
    try:
        res, sent, invocation = optimizer_sandbox.run(
            _claude_bin(), prompt, evidence, env, cli_model=cli_model, timeout=timeout, max_turns=max_turns,
            receipt_dir=receipt_dir, view=view)
    except subprocess.TimeoutExpired as exc:
        if receipt_dir is not None:
            receipts.record_call(receipt_dir, prompt, getattr(exc, "stdout", None), getattr(exc, "stderr", None),
                                 {"backend": "claude_sandbox", "error": "TimeoutExpired", "returncode": None})
        raise
    invocation = {**invocation, "requested_model": model, "optimizer_identity": resolved_identity}
    foreign = optimizer_sandbox.foreign_tools(res.stdout)
    if foreign:
        if receipt_dir is not None:
            receipts.record_call(receipt_dir, sent, res.stdout, res.stderr,
                                 {**invocation, "returncode": res.returncode, "foreign_tools": foreign})
        raise OptimizerIdentityChanged(f"claude_sandbox optimizer used tools outside its sandbox: {foreign}")
    return _claude_outcome(res, sent, receipt_dir, {**invocation})


def _invoke_resilient(prompt, run_dir, model, timeout, max_turns=None, receipt_dir=None, study=None,
                      expected_identity=None, trace_tools=True, view_gates=False):
    """`_invoke_claude`, but a throttled / overloaded / timed-out CLI call is waited out and re-sent UNCHANGED.

    Until 2026-10-15 such a call came back as an answer, failed JSON parsing, and was re-sent after ~2 s with a note
    claiming the previous output was malformed, burning one of the 5 attempts each time (3 of 90 Trace2Skill/SkillGrad
    rounds and 8 of 90 native A-Evolve rounds ended with no proposal). A subprocess timeout used to escape propose_spec
    altogether. Backoff is exponential with jitter so the 12 concurrent optimizers do not retry in lockstep. Parse
    failures and refusals are NOT handled here: they keep their own note-and-retry path in propose_spec."""
    for n in range(_TRANSIENT_RETRIES + 1):
        try:
            current = optimizer_identity(model) if expected_identity is not None else None
            if expected_identity is not None and current != expected_identity:
                raise OptimizerIdentityChanged("Optimizer identity changed before invocation; no request sent")
            backend = current["backend"] if current is not None else _optimizer_backend()
            invoke = {"claude_sandbox": _invoke_claude_sandbox}.get(backend, _invoke_claude)
            return invoke(prompt, run_dir, model, timeout, max_turns=max_turns,
                          **({"resolved_identity": expected_identity} if expected_identity is not None else {}),
                          **({"trace_tools": False} if invoke is _invoke_claude and not trace_tools else {}),
                          **({"study": study} if study else {}),
                          **({"view_gates": view_gates} if invoke is _invoke_claude_sandbox else {}),
                          **({"receipt_dir": Path(receipt_dir) / f"retry_{n}"}
                             if receipt_dir is not None else {}))
        except (_TransientApi, subprocess.TimeoutExpired) as e:
            if n == _TRANSIENT_RETRIES:
                raise _ApiExhausted(f"optimizer CLI still failing after {n + 1} tries: {str(e)[:160]}") from e
            time.sleep(min(120.0, 8.0 * 2 ** n) * random.uniform(0.6, 1.4) if n else random.uniform(3, 8))


def optimizer_infra_failure(exc):
    """Name a propose_spec exception that is an INFRASTRUCTURE failure (plumbing, not the model),
    else None. A plumbing failure means the optimizer never actually ran, so the round is NOT a
    completed optimization round and the cell must not be reported as a clean N-round optimization.
    A model-side outcome (refusal, unparseable JSON) is a real if unproductive round -> None."""
    import errno as _errno
    import subprocess as _sp
    if isinstance(exc, OptimizerIdentityChanged):
        return "optimizer_identity_changed"
    if isinstance(exc, OptimizerUnavailable):
        return "optimizer_api_unavailable"
    if isinstance(exc, FileNotFoundError):
        return "claude_cli_missing"
    if isinstance(exc, _sp.TimeoutExpired):
        return "optimizer_timeout"
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == _errno.ENAMETOOLONG:
        return "path_name_too_long"
    return None


def propose_spec(*, goal, binding_gate, measurement, evidence_paths, run_dir,
                 history_index=None, variant="minimal", attack_type="beacon",
                 writer="deterministic", model="sonnet", timeout=1200, max_retries=4,
                 transcript_path=None, single_objective=False, allow=None,
                 advisory=False, editable_cues=True, show_gates=True, textgrad=False,
                 prompt_version="v1", domain="spreadsheet", evolver=None, max_turns=None,
                 fidelity_feedback=False, feedback_protocol=feedback.LEGACY, study=None,
                 expected_identity=None):
    """Build the prompt, run the read-only agent, parse its JSON spec. Retries on malformed JSON
    (feeding the parse error back). Returns (spec_dict, raw_final_text).

    single_objective=True selects the BASELINE prompt (one aggregate J, no gate decomposition,
    no routing); `binding_gate` and `variant` are then unused. Default False = our method.

    advisory=True selects the GATE-ADVISORY prompt: the full per-gate diagnosis is shown but no
    binding gate and no component restriction -- the agent picks the target. Takes precedence over
    `single_objective`. `editable_cues` says whether it may edit the cues as well as `r`.

    allow=[paths] enables the READ WHITELIST: the agent sees a symlink workspace containing only
    those paths (relative to run_dir), instead of the whole run dir. Use it when some artifact in
    the run dir must stay invisible to the optimizer. Default None = the historical behavior
    (whole run dir readable).

    max_turns=N passes `--max-turns N` to claude, bounding exploration so a round fails fast instead
    of wandering a large run dir past `timeout` (2026-10-05: 9 degradation targets died at 1200 s).
    A capped attempt is retried with a note to read less. Default None = no cap (historical).

    domain selects the prompt's DOMAIN VOCABULARY ("spreadsheet" | "wikitq" | "officeqa"). The
    prompts used to be SpreadsheetBench-literal ("workbook", "{poison_dir}/spreadsheet/*/"), which
    is factually wrong on the QA datasets. "spreadsheet" (the default) renders byte-identically to
    the pre-change prompts, so finished SB cells stay valid.
    """
    if feedback_protocol not in feedback.VERSIONS:
        raise ValueError(f"Unknown optimizer feedback protocol: {feedback_protocol}")
    if study:
        from attack.agent.study_ablation import BOUNDED_BACKENDS
        if (_optimizer_backend() not in BOUNDED_BACKENDS or feedback_protocol != feedback.VERSION
                or not advisory or textgrad):
            raise ValueError("Matched study controls require Ours evidence_v2 and a bounded reader (claude_sandbox)")
        from attack.agent.study_ablation import prompt_inputs
        measurement, history_index = prompt_inputs(measurement, study["history"], study["variant"])
        show_gates = study["variant"] != "no_stage_feedback"
        evidence_paths = {**evidence_paths, "evidence_index": "study_evidence_index.json"}
    if feedback_protocol == feedback.VERSION and (not advisory or (not show_gates and not study) or textgrad):
        raise ValueError("evidence_v2 is scoped to Ours/gate-advisory; baselines are unchanged")
    if textgrad:
        prompt = build_textgrad_prompt(
            goal=goal, measurement=measurement, evidence_paths=evidence_paths,
            history_index=history_index, run_dir=str(run_dir), attack_type=attack_type,
            prompt_version=prompt_version, domain=domain)
    elif advisory:
        prompt = build_advisory_prompt(
            goal=goal, measurement=measurement, evidence_paths=evidence_paths,
            history_index=history_index, run_dir=str(run_dir), attack_type=attack_type,
            editable_cues=editable_cues, show_gates=show_gates, prompt_version=prompt_version,
            domain=domain, evolver=evolver, fidelity_feedback=fidelity_feedback)
    elif single_objective:
        prompt = build_single_objective_prompt(
            goal=goal, measurement=measurement, evidence_paths=evidence_paths,
            history_index=history_index, run_dir=str(run_dir), attack_type=attack_type,
            domain=domain)
    else:
        prompt = build_optimizer_prompt(
            goal=goal, binding_gate=binding_gate, measurement=measurement,
            evidence_paths=evidence_paths, history_index=history_index, run_dir=str(run_dir),
            variant=variant, attack_type=attack_type, writer=writer, domain=domain,
            evolver=evolver)
    if attack_type == "degradation" and domain in ("wikitq", "officeqa"):
        prompt = prompt.replace('"golden": {"<cell>": "<value>"}',
                                '"golden": "<corrupted QA answer string>"')
        prompt = prompt.replace('"golden": {"<cell>": <value>, ...}',
                                '"golden": "<corrupted QA answer string>"')
    if domain in ("wikitq", "officeqa"):
        from attack.victims import neutral_names
        if neutral_names.enabled():
            # POISON_NEUTRAL_NAMES: the victim sees the rule text verbatim, so it must not name the attack either.
            # Every arm (Ours, TextGrad, baselines) gets the same note.
            prompt += neutral_names.OPTIMIZER_NOTE
    workspace, ws_tmp = str(run_dir), None
    if allow or (feedback_protocol == feedback.VERSION and allow is not None):
        workspace = build_read_workspace(str(run_dir), allow)
        ws_tmp = workspace                              # ours to clean up
    receipt_dir = None
    if feedback_protocol == feedback.VERSION or expected_identity is not None:
        receipt_dir = receipts.proposal_directory(run_dir, transcript_path)
        receipts.write_json(receipt_dir / "request.json", {
            "protocol": feedback_protocol, "measurement": measurement,
            "evidence_paths": evidence_paths, "allow": allow,
            "history_index": history_index, "domain": domain, "evolver": evolver,
            "textgrad": textgrad, "advisory": advisory, "show_gates": show_gates,
            "prompt_version": prompt_version, "optimizer_identity": expected_identity})
        if study:
            receipts.write_json(receipt_dir / "study.json", study)
    last_err, refused = None, False
    api_down = 0                                       # attempts lost to API/transport, not to the model
    # ours_v4: a proposal byte-identical to an earlier round's policy is sent back instead of being
    # measured again (25 of 169 Ours cells re-proposed an earlier policy verbatim; each repeat cost a
    # full k-evolve measurement of something already known).
    repeat_note = None
    prior_policies = {}
    if prompt_version == "v4":
        from attack.agent.optimizer_prompt import lever_history, policy_key
        prior_policies = {pol: rnd for rnd, pol in lever_history(run_dir, history_index)["policies"].items()}
    try:
        for attempt in range(max_retries + 1):
            # after a refusal re-send the prompt unchanged (a parse-error note would only make it look
            # stranger); after a parse failure append the note; after an exact repeat, say which round.
            p = prompt if (attempt == 0 or refused) else (
                prompt + "\n\n# RETRY\n" + (repeat_note or
                f"Your previous output could not be parsed as JSON ({last_err}). "
                f"Emit ONLY the JSON object specified above, nothing else."))
            repeat_note = None
            # A refusal is TRANSIENT for this task (sonnet sits near the safeguard boundary: the same
            # prompt is accepted on one call and flagged on the next), so retry instead of losing the
            # round -- clean1 lost all of round 0 to a single refusal.
            try:
                # The sandboxed cell view shows gate files only to an arm shown gates (Ours), never to TextGrad.
                raw = _invoke_resilient(p, workspace, model, timeout, max_turns=max_turns,
                                        view_gates=bool(show_gates and not textgrad),
                                       **({"expected_identity": expected_identity,
                                           "trace_tools": feedback_protocol == feedback.VERSION}
                                          if expected_identity is not None else {}),
                                       **({"study": study} if study else {}),
                                       **({"receipt_dir": receipt_dir / f"attempt_{attempt}"}
                                          if receipt_dir is not None else {}))
            except _TurnCapHit as e:
                # Not a refusal: re-send WITH a note, so the next attempt reads less and answers.
                last_err, refused = (f"{e}; read only the few files you need, then emit the JSON"
                                     ), False
                continue
            except RuntimeError as e:
                last_err, refused = str(e)[:200], True
                api_down += isinstance(e, _ApiExhausted)
                continue
            refused = False
            if transcript_path:
                open(transcript_path, "a").write(f"\n===== attempt {attempt} =====\n{raw}\n")
            spec, err = _extract_json(raw)
            if spec is not None and prior_policies:
                earlier = prior_policies.get(policy_key(spec))
                if earlier is not None:
                    verdict = next((h.get("gate_delta") or h.get("j_delta") for h in (history_index or [])
                                    if h.get("round") == earlier), None)
                    last_err = (f"your proposal is byte-identical to round {earlier}'s policy, which was "
                                f"already measured ({verdict or 'see history'})")
                    repeat_note = (f"Your previous output repeated round {earlier}'s policy VERBATIM (same r, "
                                   f"same cues); it was already measured ({verdict or 'see history'}) and will "
                                   f"not be measured again. Propose a DIFFERENT edit: change the rule text, its "
                                   f"form, or a different cue, and say which failure signal it targets.")
                    if transcript_path:
                        open(transcript_path, "a").write(f"\n===== attempt {attempt} (exact repeat of round {earlier}) =====\n{raw}\n")
                    continue
            if spec is not None:
                if receipt_dir is not None:
                    receipts.write_json(receipt_dir / "proposal.json", spec)
                return spec, raw
            last_err = err
    finally:
        if ws_tmp:
            shutil.rmtree(ws_tmp, ignore_errors=True)
    if api_down == max_retries + 1:
        raise OptimizerUnavailable(f"optimizer API unavailable on all {max_retries+1} attempts: {last_err}")
    raise ValueError(f"optimizer agent produced no parseable JSON after {max_retries+1} attempts: {last_err}")
