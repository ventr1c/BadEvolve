"""Unified attack interface: an Attack turns a set of base tasks into a poison
payload that the victim's evolution pipeline will consume.

Two families implement it:
  TBAAttack   — fixed hand-engineered recipe (arXiv:2608.08303), trajectory-level.
  OptimAttack — our feedback-driven optimizer (wraps attack/orchestrate + contrastive).

The evaluation layer never imports either directly; it only sees this interface,
so attack and evaluation stay decoupled (attacks/ vs evaluation/).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from attack.evaluation.targets import TargetAction


@dataclass
class PoisonPayload:
    """What gets injected into the victim's evolution pool for one batch."""
    kind: str                          # "trajectory" | "task"
    items: list = field(default_factory=list)     # trajectories or tasks
    triggered_ids: list = field(default_factory=list)
    clean_ids: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        n = len(self.triggered_ids) + len(self.clean_ids)
        return len(self.triggered_ids) / n if n else 0.0


class Attack(Protocol):
    name: str
    target: TargetAction

    def build_payload(self, base_tasks: list[dict]) -> PoisonPayload:
        """base_tasks: [{id, instruction, xlsx_path, ...}] -> PoisonPayload."""
        ...
