"""ONE definition of what each optimizer arm (`--gate-mode`) is allowed to see and do.

WHY
  Both drivers used to recompute the arm's identity from `a.gate_mode` string lists at every use
  site (12 of them). The lists drifted: the read whitelist was once applied to `ours` as well as to
  the blind baselines (169/513 QA rounds were affected), and `mechanism_binding` landed in the note
  of the wrong arm. Every such bug was one list that did not match another. Resolving the arm once,
  here, makes that class of drift impossible: a driver reads `arm.whitelist`, it never re-derives it.

WHAT IS AN ARM (the three arms of the paper)
  ours_v5       BadEvolve: the gate-ADVISORY prompt (all stage indicators shown, no binding gate),
                prompt version v5 (stage-wise feedback + levers), rollback with the failure-aware gate.
  textgrad      TextGrad-Poison: the gate-blind single-objective baseline prompt; restricted reads
                (READ WHITELIST), aggregate-only feedback, never rolls back.
  tba           TBA: the fixed hand-engineered recipe, no optimization rounds.

Fields: advisory / show_gates (advisory prompt, gates visible), textgrad (gate-blind baseline prompt),
single (r-only seed + aggregate feedback), blind (restricted reads), leak_note (always False here),
tba (fixed recipe), forces_no_rollback (rollback is part of OUR method only).

The values below are exactly the booleans the two drivers computed for these three gate modes.
"""
from __future__ import annotations

from dataclasses import dataclass

_TEXTGRAD = ("textgrad",)


@dataclass(frozen=True)
class ArmPolicy:
    gate_mode: str
    advisory: bool
    show_gates: bool
    textgrad: bool
    single: bool
    blind: bool
    leak_note: bool
    tba: bool
    prompt_version: str
    forces_no_rollback: bool
    run_prefix: str
    fidelity_feedback: bool = False

    @property
    def gate_blind(self) -> bool:
        """The arm must not see a per-gate decomposition anywhere (prompt, notes, measurements)."""
        return self.blind or (self.advisory and not self.show_gates)


_PREFIX = {"textgrad": "agentopt_textgrad", "ours_v5": "agentopt_oursv5", "tba": "agentopt_tba"}

SB_GATE_MODES = ("ours_v5", "textgrad", "tba")
QA_GATE_MODES = ("textgrad", "ours_v5", "tba")


def resolve_arm(gate_mode: str, driver: str) -> ArmPolicy:
    """Resolve `--gate-mode` for `driver` in {"sb", "qa"}. Raises on a mode the driver does not
    support, so an arm can never silently fall through to another arm's defaults."""
    allowed = {"sb": SB_GATE_MODES, "qa": QA_GATE_MODES}[driver]
    if gate_mode not in allowed:
        raise ValueError(f"gate mode {gate_mode!r} is not supported by the {driver} driver; "
                         f"expected one of {allowed}")
    textgrad = gate_mode in _TEXTGRAD
    if driver == "sb":
        advisory = gate_mode == "ours_v5"
        show_gates = gate_mode == "ours_v5"
        single = gate_mode in _TEXTGRAD
        blind = gate_mode in _TEXTGRAD
    else:
        # The QA driver has only the advisory arm and the textgrad baselines (tba never proposes):
        # everything that is not textgrad renders the advisory prompt with gates shown, and the
        # builder dispatch never reaches the single-objective prompt.
        advisory = not textgrad
        show_gates = not textgrad
        single = textgrad
        blind = textgrad
    return ArmPolicy(
        gate_mode=gate_mode, advisory=advisory, show_gates=show_gates, textgrad=textgrad,
        single=single, blind=blind, leak_note=False,
        tba=(gate_mode == "tba"),
        # ours_v5: the advisory prompt + the v2 concreteness constraint + the fidelity block, lever-aware history,
        # exact-repeat rejection, the "defended" outcome, the ungated observations (F_obs, forgot vs mentioned) and
        # a LEVERS block tying each lever to the measured signal (optimizer_prompt.build_advisory_prompt).
        prompt_version=("v5" if gate_mode == "ours_v5" else "v1"),
        forces_no_rollback=(textgrad or gate_mode == "tba"),
        run_prefix=_PREFIX[gate_mode],
        fidelity_feedback=(gate_mode == "ours_v5"))
