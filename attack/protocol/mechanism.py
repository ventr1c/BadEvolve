"""The optimizer prompt's mechanism binding must be the one the victim actually runs.

`optimizer_prompt.mechanism_key(...)` resolves a binding from CLI-level names (framework, evolver
mode, protocol, dataset). Each victim runner separately DECLARES `mechanism_key` next to the code
that makes it true (forced failure pool, disabled selection gate, workspace concatenation, ...).
Two independent derivations of the same fact: if they disagree, the prompt would describe a
different victim than the one being attacked -- which is exactly how QA SkillOpt came to be called
"SkillOpt (SpreadsheetBench)". So the driver refuses to start.
"""
from __future__ import annotations


class MechanismMismatch(RuntimeError):
    pass


def assert_mechanism_matches(prompt_key: str | None, victim) -> None:
    declared = getattr(victim, "mechanism_key", None)
    if declared in ("skillopt_qa", "skillopt_sb"):
        # Adapters declare the deployment family; their acceptance variant is selected
        # by the canonical gate resolver used inside evolve(). Check both facts.
        from attack.victims.evolution.skillopt_gate import gate_variant
        gate = gate_variant()
        if gate != "strict":
            declared = f"{declared}_{gate}"
    if declared is None:
        raise MechanismMismatch(
            f"{type(victim).__name__} declares no mechanism_key, so the prompt binding "
            f"{prompt_key!r} cannot be verified against it")
    if declared != prompt_key:
        raise MechanismMismatch(
            f"optimizer prompt would describe {prompt_key!r} but {type(victim).__name__} runs "
            f"{declared!r}")
