"""Environment knobs that change what a cell measures, and how each driver records them (2026-10-20).

Thirty-nine environment variables are read under attack/ and scripts/. Some decide what a cell measures
(the solver's answer budget, a request timeout that turns a slow solve into a MISSING observation, the
confinement of a victim shell, a switch in the optimizer prompt) and used to be recorded nowhere: a cell
launched with WIKITQ_ANSWER_TOKENS=16384 wrote the same meta.json as one launched at the default 4000, and
the legacy RQ1 table already mixes several such invisible protocol signatures. This module is the one
list. Each driver records every knob its process reads (lowercase key, value exactly as the runtime
parses it, defaults included) and folds it into the resume identity, so `--resume` with a different
value is refused instead of splicing two protocols into one cell.

Two compatibility rules keep every cell written before this file existed readable and resumable:
  * a key missing from an old meta.json means the legacy default (`legacy_default`);
  * a knob at its legacy default stays OUT of the measurement fingerprint (`non_default`), so the
    records of a cell that ran at the defaults keep their fingerprint under the new driver.

Operational variables (paths, proxies, credentials, scheduling, logging) are listed too, as `kind ==
"operational"`, so the inventory test can tell "classified as operational" from "forgotten".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

QA, SB = "qa", "sb"


@dataclass(frozen=True)
class Cell:
    """What a knob's applicability depends on. Built by the drivers at launch and from an old meta.json
    by `context_of`."""
    driver: str                   # "qa" | "sb"
    dataset: str | None = None    # wikitq | officeqa | spreadsheetbench
    framework: str | None = None  # trace2skill | skillgrad | skillopt | aevolve | selfevolve
    native: bool = False          # native multi-skill deployment (A-Evolve)
    gate_mode: str | None = None  # the arm; the SB driver turns rollout isolation on for fidelity arms


@dataclass(frozen=True)
class Knob:
    env: str                                   # variable name, as the runtime reads it
    kind: str                                  # "protocol" | "operational" | "proxy"
    key: str | None = None                     # meta.json key (protocol knobs only)
    read: Callable[[Cell], object] | None = None   # the runtime's own parse
    default: object = None                     # legacy value, or a Callable[[Cell], object]
    applies: Callable[[Cell], bool] | None = None  # None = every cell of the listed drivers
    drivers: tuple = ()                        # which driver processes read it
    reads: str = ""                            # where the runtime reads it
    effect: str = ""                           # what it changes
    recorded_by: str = "env_identity"          # "env_identity" | "driver" (an older mechanism) | "-"

    def applicable(self, cell: Cell) -> bool:
        return cell.driver in self.drivers and (self.applies is None or bool(self.applies(cell)))

    def legacy(self, cell: Cell):
        return self.default(cell) if callable(self.default) else self.default


def _flag(name):
    return lambda cell: os.environ.get(name) == "1"


def _int(name, default):
    return lambda cell: int(os.environ.get(name, default))


def _float(name, default):
    return lambda cell: float(os.environ.get(name, default))


def _budget_failures(cell):
    from attack.clean.qa_measurement import budget_policy
    return budget_policy()


def _rate_limit_retries(cell):
    from attack.clean.common import RATE_LIMIT_RETRIES
    return RATE_LIMIT_RETRIES


















def _neutral_names(cell):
    from attack.victims import neutral_names
    return neutral_names.enabled()






def _rollout_isolate(cell):
    from attack.victims.trace2skill_runner import _rollout_isolate
    return _rollout_isolate()


def _caption_cite(cell):
    from attack.victims.wikitq_runner import caption_cite
    return caption_cite()


def _rollout_isolate_default(cell):
    """What an SB cell ran with when nothing was set: the driver `setdefault`s the flag to 1 for the
    fidelity-feedback arm (ours_v3), so for that arm the legacy value is True."""
    if cell.driver != SB or not cell.gate_mode:
        return False
    from attack.protocol.arms import resolve_arm
    try:
        return bool(resolve_arm(cell.gate_mode, SB).fidelity_feedback)
    except ValueError:
        return False


def _warm_start_skill(cell):
    v = os.environ.get("WARM_START_SKILL")
    return os.path.abspath(v) if v else None


def _solver_http_timeout(cell):
    return float(os.environ.get("SOLVER_HTTP_TIMEOUT", "300" if cell.dataset == "wikitq" else "480"))


def _solver_http_timeout_default(cell):
    return {"wikitq": 300.0, "officeqa": 480.0}.get(cell.dataset)


_wikitq = lambda c: c.dataset == "wikitq"
_officeqa = lambda c: c.dataset == "officeqa"
_aevolve = lambda c: c.framework == "aevolve"
_skillgrad = lambda c: c.framework == "skillgrad"
_skillopt = lambda c: c.framework == "skillopt"
_officeqa_or_native = lambda c: c.dataset == "officeqa" or c.framework == "aevolve"
_sb_native_ae = lambda c: c.native or c.framework == "aevolve"

KNOBS: tuple[Knob, ...] = (
    # ---- protocol-defining, recorded here --------------------------------------------------------
    Knob("WIKITQ_ANSWER_TOKENS", "protocol", "wikitq_answer_tokens", _int("WIKITQ_ANSWER_TOKENS", "4000"),
         4000, _wikitq, (QA,), "attack/victims/wikitq_runner.py:153,171",
         "solver max_tokens on WikiTQ: decides whether an answer (and the beacon) fits"),
    Knob("OFFICEQA_ANSWER_TOKENS", "protocol", "officeqa_answer_tokens", _int("OFFICEQA_ANSWER_TOKENS", "4000"),
         4000, _officeqa, (QA,), "attack/victims/officeqa_runner.py:280",
         "solver max_tokens on OfficeQA (was recorded only for clean-startpoint cells)"),
    Knob("SOLVER_HTTP_TIMEOUT", "protocol", "solver_http_timeout", _solver_http_timeout,
         _solver_http_timeout_default, None, (QA,),
         "attack/victims/wikitq_runner.py:48 (300), officeqa_runner.py:76 (480)",
         "per-request read timeout of the QA solver client; a timeout is a MISSING observation"),
    Knob("POISON_QA_SOLVE_TIMEOUT", "protocol", "poison_qa_solve_timeout", _int("POISON_QA_SOLVE_TIMEOUT", "420"),
         420, _officeqa_or_native, (QA,),
         "attack/victims/officeqa_clean.py:174, attack/clean/aevolve_runtime.py:99",
         "per-request timeout of the OfficeQA tool loop and the native A-Evolve runtime"),
    Knob("POISON_SOLVE_MISSING_MAX", "protocol", "poison_solve_missing_max",
         _float("POISON_SOLVE_MISSING_MAX", "0.05"), 0.05, None, (QA,),
         "scripts/run_wikitq_optimization.py:377",
         "max MISSING fraction of a pass before the evolve is redone instead of reported"),
    Knob("QA_BUDGET_FAILURES", "protocol", "qa_budget_failures", _budget_failures, "missing", None, (QA,),
         "attack/clean/qa_measurement.py (budget_policy)",
         "turn/length-limit and no-answer solves: MISSING (paired_v2 default) or graded as wrong answers"),
    Knob("POISON_SOLVE_MAX_TRANSPORT_FAILS", "protocol", "poison_solve_max_transport_fails",
         _int("POISON_SOLVE_MAX_TRANSPORT_FAILS", "0"), 0, None, (QA,),
         "attack/clean/common.py:98 (run_batch: benchmark.run_batch, officeqa_clean, evolution/skillopt)",
         "how many persistent transport failures a batch tolerates as missing rows before it aborts"),
    Knob("POISON_QA_GATE_CONDITIONED", "protocol", "poison_qa_gate_conditioned",
         _flag("POISON_QA_GATE_CONDITIONED"), False, None, (QA,),
         "scripts/run_wikitq_optimization.py:223",
         "Following gate conditioned on induced-and-finished vs unconditioned: changes gate routing"),
    Knob("AE_INPUT_REVIEW", "protocol", "ae_input_review", _flag("AE_INPUT_REVIEW"), False, _aevolve, (QA,),
         "attack/victims/aevolve_qa_runner.py:68",
         "native A-Evolve QA input review on/off (unless the runner is told explicitly)"),
    Knob("SG_REQUEST_TIMEOUT_S", "protocol", "sg_request_timeout_s", _float("SG_REQUEST_TIMEOUT_S", "360"),
         360.0, _skillgrad, (QA,), "attack/victims/evolution/skillgrad.py:248",
         "per-request timeout inside the SkillGrad evolve bridge (QA path; the SB runner uses stages)"),
    Knob("SG_STAGE_TIMEOUT_S", "protocol", "sg_stage_timeout_s", _int("SG_STAGE_TIMEOUT_S", "5400"),
         5400, _skillgrad, (SB,), "attack/victims/skillgrad_runner.py:253",
         "upper bound on one SkillGrad stage; a timed-out stage fails the evolve"),
    Knob("SG_BASE_GRACE_S", "protocol", "sg_base_grace_s", _int("SG_BASE_GRACE_S", "90"),
         90, _skillgrad, (SB,), "attack/victims/skillgrad_runner.py:259",
         "how long a finished-but-hung sg_base is left before it is ended"),
    Knob("POISON_T2S_ROLLOUT_ISOLATE", "protocol", "poison_t2s_rollout_isolate", _rollout_isolate,
         _rollout_isolate_default, None, (SB,), "attack/victims/trace2skill_runner.py:561 (_rollout_isolate)",
         "poison rollout solves on a throwaway skills copy (no self-install); the SB driver sets it for fidelity arms"),
    # Every QA cell: the optimizer NAMING note (optimizer_harness.propose_spec, e36621a) reaches A-Evolve QA arms too.
    Knob("POISON_NEUTRAL_NAMES", "protocol", "poison_neutral_names", _neutral_names, False, None, (QA, SB),
         "attack/victims/neutral_names.py:enabled (qa_framework_runner, aevolve_qa_runner, the SB runners, "
         "optimizer_harness.propose_spec)",
         "trace2skill/skillgrad/skillopt see poison pool items under dataset-native ids (not poison-*) and run their fw/ "
         "tree in a neutral /tmp/ws-* workspace copied back afterwards; every QA optimizer prompt gets a NAMING note "
         "(the rule must not name the attack); native A-Evolve QA/SB run in a neutral mirror; SB poison tasks are "
         "<base>-a/-b in train_data/"),
    Knob("POISON_AE_GPTOSS_TOOLFIX", "protocol", "poison_ae_gptoss_toolfix",
         lambda c: os.environ.get("POISON_AE_GPTOSS_TOOLFIX") == "1", False, lambda c: c.native, (QA,),
         "attack/clean/aevolve_runtime.py:gptoss_toolfix (A-Evolve QA solver; GPT-OSS backbones only)",
         "GPT-OSS sees its calculator as run_python instead of python (its built-in tool name), and a turn whose "
         "output Bedrock dropped is re-sent up to 3 times"),
    Knob("POISON_SB_EARLY_EXIT_FIX", "protocol", "poison_sb_early_exit_fix", _flag("POISON_SB_EARLY_EXIT_FIX"), False,
         lambda c: c.framework in ("trace2skill", "skillgrad", "skillopt") and not c.native, (SB,),
         "attack/victims/sb_early_exit.py:enabled (Trace2SkillRunner.solve; attack/clean/spreadsheet.py:solve_task)",
         "the shared SB held-out solver re-sends an empty reply up to 3 times, and answers a reply with no Action block "
         "(while the output workbook is missing) with a request for an `Action: {json}` block, up to 5 times per "
         "episode, instead of ending the task"),
    Knob("AE_SOLVER_WORKSPACE", "protocol", "ae_solver_workspace",
         lambda c: os.environ.get("AE_SOLVER_WORKSPACE", "off"), "off", _sb_native_ae, (SB,),
         "attack/victims/aevolve_spreadsheetbench.py:294",
         "native SB A-Evolve workspace guard off|detect|protect"),
    Knob("WARM_START_SKILL", "protocol", "warm_start_skill", _warm_start_skill, None, _skillopt, (SB,),
         "attack/victims/skillopt_runner.py:183",
         "SkillOpt trainer starts from this evolved skill instead of the package seed"),
    Knob("TEXTGRAD_CONTEXT", "protocol", "textgrad_context", _flag("TEXTGRAD_CONTEXT"), False, None, (QA, SB),
         "attack/agent/optimizer_harness.py:558",
         "textgrad baseline prompt carries the authorization header: changes the proposed poison text"),
    Knob("POISON_FIX_PROMPT_JSON", "protocol", "poison_fix_prompt_json", _flag("POISON_FIX_PROMPT_JSON"),
         False, None, (QA, SB), "attack/agent/optimizer_prompt.py:1180",
         "optimizer prompt renders the valid JSON skeleton: changes the proposal format"),
    Knob("QA_CAPTION_CITE", "protocol", "qa_caption_cite", _caption_cite, "off", _wikitq, (QA,),
         "attack/victims/wikitq_runner.py (caption_cite / _caption_for: inject and solve)",
         "the WikiTQ caption carries the requirement itself ('-- cite as PROVENANCE=<tok>'): off | inject "
         "(poison rows only; eval captions frozen) | all"),
    Knob("RATE_LIMIT_RETRIES", "protocol", "rate_limit_retries", _rate_limit_retries, 10, None, (QA, SB),
         "attack/clean/common.py:19 (a constant, not an env var; every OpenAI client's max_retries)",
         "how many 429/5xx a request survives before it is graded as a failed solve"),
    # ---- protocol-defining, recorded by an older driver mechanism (kept as is) ----------------------
    Knob("QA_CARRIER_POS", "protocol", "qa_carrier_pos", drivers=(QA,), default="tail", recorded_by="driver",
         reads="attack/evaluation/qa_targets.py:61", effect="where the beacon carrier sits in the poison"),
    Knob("QA_BEACON_PLACEMENT", "protocol", "qa_beacon_placement", drivers=(QA,), default="after",
         recorded_by="driver", reads="attack/evaluation/qa_targets.py:68", effect="beacon placement"),
    Knob("QA_POISON_CLEAN", "protocol", "qa_poison_clean", drivers=(QA,), default="0", recorded_by="driver",
         reads="scripts/run_wikitq_optimization.py:325", effect="clean half of the poison batch"),
    Knob("POISON_SG_CONTEXT_FIT", "protocol", "poison_sg_context_fit",
         lambda c: os.environ.get("POISON_SG_CONTEXT_FIT") == "1", False, lambda c: c.framework == "skillgrad",
         (QA,), "attack/victims/evolution/skillgrad.py:context_fit_budget (models in CONTEXT_WINDOWS: Qwen)",
         "SkillGrad diagnoser/momentum/patcher reads share a budget fitted to the model's context window; an overflowing "
         "read keeps head and tail around a marked omission"),
    Knob("POISON_SG_TASK_FRAME", "protocol", "poison_sg_task_frame",
         lambda c: (os.environ.get("POISON_SG_TASK_FRAME") or "question").strip().lower(), "question",
         lambda c: c.framework == "skillgrad", (QA,), "attack/victims/evolution/skillgrad.py:task_frame_mode",
         "SkillGrad diagnoser's '## Task' is the bare question (question) or the victim's first user message from the "
         "attempt's trace, standing instructions included (prompt)"),
    Knob("SG_TRAIN_POOL", "protocol", "sg_train_pool", drivers=(QA,), default="failures", recorded_by="driver",
         reads="attack/victims/evolution/skillgrad.py:106", effect="SkillGrad distils failures|contrast"),
    Knob("QA_EVOLVE_POOL", "protocol", "qa_evolve_pool", drivers=(QA,), default="poison_only",
         recorded_by="driver", reads="attack/victims/qa_framework_runner.py:145",
         effect="evolve pool poison_only|clean_plus_poison"),
    Knob("QA_EVOLVE_CLEAN_N", "protocol", "qa_evolve_clean_n", drivers=(QA,), default="0", recorded_by="driver",
         reads="attack/victims/qa_framework_runner.py:152", effect="cap on the clean side of the evolve pool"),
    Knob("QA_SCORE_REVIEW", "protocol", "qa_score_review", drivers=(QA,), default="official_qa_v2",
         recorded_by="driver", reads="attack/clean/qa_measurement.py:review_measurement (the driver records the "
         "version chosen by resolve_review; there is no env var)",
         effect="official regrade of both QA arms: official_qa_v3 (new cells since 2026-10-21) grades a completed "
         "solve whose answer is unresolved <answer> markup as wrong; official_qa_v2 disputed the whole measurement "
         "and ended the cell (nt-5308); a recorded version never changes under its tag"),
    Knob("SKILLOPT_GATE", "protocol", "skillopt_gate", drivers=(QA, SB), default="strict", recorded_by="driver",
         reads="attack/victims/evolution/skillopt_gate.py:25", effect="the gate the SkillOpt victim obeys"),
    Knob("OPTIMIZER_BACKEND", "protocol", "optimizer_backend", drivers=(QA, SB), default="claude",
         recorded_by="driver", reads="attack/agent/optimizer_harness.py:335 (+ both drivers)",
         effect="which optimizer CLI proposes (claude|codex); inside optimizer_identity"),
    Knob("OPTIMIZER_SANDBOX_VIEW", "operational", reads="attack/agent/optimizer_harness.py (_sandbox_view)",
         effect="claude_sandbox view: pack (text evidence copy) | cell (+ read-only shell over this cell's evolve "
                "dirs); recorded as optimizer_identity.tools, so a change refuses --resume"),
    # ---- protocol-defining but read inside the LiteLLM proxy, not by the driver process ------------
    # ---- operational ----------------------------------------------------------------------------
    Knob("SG_TRAIN_STALL_S", "operational", reads="attack/victims/skillgrad_runner.py (_wait_stage)",
         effect="end a SkillGrad sg_train whose outputs stopped changing and retry it once (hang recovery)"),
    Knob("OPTIMIZER_BEDROCK_SPREAD", "operational", reads="attack/agent/optimizer_harness.py:86,358",
         effect="spread Sonnet optimizer calls over regions (same model)"),
    Knob("OPTIMIZER_AWS_REGION", "operational", reads="attack/agent/optimizer_harness.py:88,359",
         effect="pin the optimizer's Bedrock region"),
    Knob("CLAUDE_BIN", "operational", reads="attack/agent/optimizer_harness.py:220", effect="claude CLI path"),
    Knob("POISON_CLAUDE_CONFIG_DIR", "operational", reads="attack/agent/optimizer_harness.py:54",
         effect="CLAUDE_CONFIG_DIR handed to the optimizer CLI"),
    Knob("CODEX_HOME", "operational", reads="attack/agent/gate_judge_agent.py:15", effect="codex config dir"),
    Knob("OPENAI_API_KEY", "operational", reads="attack/victims/skillopt_runner.py:51", effect="dummy proxy key"),
    Knob("PATH", "operational", reads="attack/victims/trace2skill_runner.py:606",
         effect="child PATH"),
    Knob("SE_PIPELINE_SOFFICE_LOCK", "operational",
         reads="attack/clean/private_tmp.py:reexec_in_private_tmp (set by skillgrad_runner._sg_env)",
         effect="SkillGrad's host-wide soffice lock file; bound back into a private /tmp"),
    Knob("HF_TOKEN", "operational", reads="attack/victims/officeqa_clean.py:72", effect="dataset download token"),
    Knob("HUGGING_FACE_HUB_TOKEN", "operational", reads="attack/victims/officeqa_clean.py:72",
         effect="dataset download token"),
    Knob("CLEAN_CELL_PYTHON", "operational", reads="attack/clean/spreadsheet.py:272", effect="interpreter path"),
    Knob("MAXJOBS", "operational", reads="attack/clean/workload.py:31", effect="clean-grid concurrency"),
    Knob("PER_PROXY_HARD", "operational", reads="attack/clean/workload.py:35", effect="clean-grid per-proxy cap"),
    Knob("RQ1_RUNS_DIR", "operational", reads="scripts/rq1_*.py", effect="where the readers look for runs"),
    Knob("POISON_NEUTRAL_ROOT", "operational", reads="attack/victims/neutral_names.py:open_workspace",
         effect="parent of the neutral framework workspace under POISON_NEUTRAL_NAMES=1 (default the temp dir)"),
    Knob("OUTPUT_FILE", "operational", reads="attack/victims/sb_early_exit.py:_output_missing",
         effect="internal: the output workbook path upstream BaseSpreadsheetAgent.run sets inside the SB solver process"),
)

_BY_ENV = {k.env: k for k in KNOBS}
_BY_KEY = {k.key: k for k in KNOBS if k.key}
RECORDED = tuple(k for k in KNOBS if k.recorded_by == "env_identity")


def context_of(meta: dict, driver: str) -> Cell:
    """The applicability context of an existing meta.json (either driver's)."""
    return Cell(driver, dataset=meta.get("dataset") or (meta.get("victim") if driver == QA else None),
                framework=meta.get("framework"),
                native=meta.get("deployment_protocol") == "native_multi_skill",
                gate_mode=meta.get("gate_mode"))


def keys_for(driver: str) -> tuple[str, ...]:
    """Every meta.json key this module may record for a driver (whether or not a given cell gets it)."""
    return tuple(k.key for k in RECORDED if driver in k.drivers)


def env_names(driver: str | None = None, kind: str = "protocol") -> set[str]:
    return {k.env for k in KNOBS if k.kind == kind and (driver is None or driver in k.drivers)}


def record(cell: Cell) -> dict:
    """What the driver writes into meta.json: every applicable knob, as the runtime reads it right now."""
    return {k.key: k.read(cell) for k in RECORDED if k.applicable(cell)}


def legacy_default(key: str, cell: Cell):
    """The value a cell whose meta.json lacks `key` ran with (None when the knob does not apply)."""
    k = _BY_KEY.get(key)
    return k.legacy(cell) if k is not None and k.applicable(cell) else None


def identity_value(meta: dict, key: str, cell: Cell):
    """`meta[key]`, or its legacy default when the cell predates the key."""
    return meta[key] if key in meta else legacy_default(key, cell)


def non_default(rec: dict, cell: Cell) -> dict:
    """The knobs that differ from their legacy default: the only ones that may enter a measurement
    fingerprint, so records written at the defaults keep theirs."""
    return {key: v for key, v in rec.items()
            if key in _BY_KEY and _BY_KEY[key].recorded_by == "env_identity" and v != legacy_default(key, cell)}


def strip_defaults(ident: dict, cell: Cell) -> dict:
    """Drop the recorded knobs that sit at their legacy default (the SB resume identity: an old meta
    without the keys and a new one at the defaults describe the same cell)."""
    return {key: v for key, v in ident.items()
            if not (key in _BY_KEY and _BY_KEY[key].recorded_by == "env_identity"
                    and v == legacy_default(key, cell))}


def inventory_markdown() -> str:
    """The knob table."""
    rows = ["| variable | kind | meta key | legacy default | driver(s) | read at | effect |",
            "|---|---|---|---|---|---|---|"]
    for k in KNOBS:
        default = ({"solver_http_timeout": "300.0 wikitq / 480.0 officeqa",
                    "poison_t2s_rollout_isolate": "False; True for the ours_v3 arm (driver setdefault)"}
                   .get(k.key, repr(k.default)) if k.kind == "protocol" else "")
        rows.append(f"| `{k.env}` | {k.kind}{'' if k.recorded_by != 'driver' else ' (recorded by the driver before this change)'} "
                    f"| {('`' + k.key + '`') if k.key else ''} | {default} | {', '.join(k.drivers)} | {k.reads} | {k.effect} |")
    return "\n".join(rows)
