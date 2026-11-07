"""Optimizer objectives and reporting roles for the current experiment policy.

Beacon maximizes action-only ASR. Degradation minimizes triggered task accuracy
by maximizing 1 - TACC, irrespective of clean correctness. CACC and conditional
DASR remain diagnostics. Historical names stay readable without changing old files.

paired_drop (opt-in, user decision 2026-10-21): degradation may instead select on the
paired net drop CACC - TACC, the accuracy the trigger removes on the same tasks, so a
proposal is not rewarded for the victim's natural errors. error_rate stays the default.
"""
from __future__ import annotations

POLICY = "asr_and_task_error_v1"
OBJECTIVES: dict[str, tuple[str, str]] = {
    "asr": ("H_mean", "target action performed on triggered tasks, independent of correctness"),
    "tasr": ("H_mean", "legacy alias for action-only ASR"),
    "error_rate": ("C_mean", "1 - TACC: triggered task error rate, independent of clean correctness"),
    "tacc_drop": ("C_mean", "compatibility alias for error_rate (1 - TACC)"),
    "jperp": ("J_perp_mean", "joint action and task correctness (explicit legacy/control objective)"),
    "tacc": ("C_mean", "triggered task accuracy (explicit accuracy-control objective)"),
    "paired_break": ("DASR", "historical conditional DASR objective; auxiliary only in new runs"),
    "paired_drop": ("CACC_mean", "CACC - TACC: paired net accuracy drop on the same tasks (degradation, opt-in)"),
}
# Explicit directions keep the positive-accuracy tacc control unchanged.
_MINIMISED = frozenset({"error_rate", "tacc_drop"})
# Keep historical objective decoding, but do not offer DASR to new optimizer runs.
OPTIMIZER_CHOICES = tuple(name for name in OBJECTIVES if name != "paired_break")
DEFAULT = "asr"
DEFAULT_BY_ATTACK_TYPE = {"degradation": "error_rate"}
REPORTED_BY_ATTACK_TYPE = {"degradation": ("TACC", "CACC", "DASR")}
REPORTED_DEFAULT = ("ASR", "TACC", "CACC", "clean_leak")
REPORTED_METRIC = {"asr": "ASR", "tasr": "ASR", "error_rate": "task_error_rate",
                   "tacc_drop": "TACC_drop", "jperp": "J_perp", "tacc": "TACC", "paired_break": "DASR",
                   "paired_drop": "TACC_drop_paired"}


def default_for(attack_type: str = "") -> str:
    return DEFAULT_BY_ATTACK_TYPE.get(attack_type or "", DEFAULT)


def resolve_for(attack_type: str, requested: str | None = None) -> str:
    """Resolve current optimization policy; historical paired objectives are read-only."""
    name = requested or default_for(attack_type)
    if name not in OPTIMIZER_CHOICES:
        raise ValueError(f"Objective {name!r} is not available for new optimizer runs; DASR is auxiliary")
    if attack_type == "degradation":
        # Historical SB --objective tasr used raw triggered error, not beacon ASR.
        if name in {"tasr", "tacc_drop"}:
            return "error_rate"
        if name not in ("error_rate", "paired_drop"):
            raise ValueError("Degradation optimizes error_rate (1 - TACC) or, opt-in, paired_drop (CACC - TACC); "
                             "DASR does not select candidates")
    elif name == "paired_drop":
        raise ValueError("paired_drop (CACC - TACC) is defined for the degradation attack only")
    return name


def reported_metrics(attack_type: str = "") -> tuple[str, ...]:
    return REPORTED_BY_ATTACK_TYPE.get(attack_type or "", REPORTED_DEFAULT)


def policy_metadata(attack_type: str) -> dict:
    return {"metric_policy": POLICY,
            "primary_metric": "TACC" if attack_type == "degradation" else "ASR",
            "primary_metric_direction": "minimize" if attack_type == "degradation" else "maximize",
            "auxiliary_metrics": ["CACC", "DASR"] if attack_type == "degradation" else ["TACC", "CACC", "clean_leak"]}


def metric_label(name: str = DEFAULT) -> str:
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}; known: {sorted(OBJECTIVES)}")
    return REPORTED_METRIC[name]


def objective_value(agg: dict, name: str = DEFAULT):
    """Return a utility to maximize, preserving unavailable measurements as None."""
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}; known: {sorted(OBJECTIVES)}")
    field, _ = OBJECTIVES[name]
    if name == "paired_drop":
        cacc, tacc = agg.get("CACC_mean"), _accuracy(agg)
        return None if cacc is None or tacc is None else cacc - tacc
    value = agg.get(field)
    if value is None and field == "C_mean":
        value = agg.get("ACC_mean")
        if value is None:
            value = agg.get("TACC_mean")
    if value is None and field == "H_mean":
        value = agg.get("ASR_mean")
        if value is None:
            value = agg.get("TASR_mean")
    return (None if value is None else 1.0 - value) if name in _MINIMISED else value


def _accuracy(agg: dict):
    for key in ("C_mean", "ACC_mean", "TACC_mean"):
        if agg.get(key) is not None:
            return agg[key]
    return None


def describe(name: str = DEFAULT) -> str:
    return f"{name}: {OBJECTIVES[name][1]}"
