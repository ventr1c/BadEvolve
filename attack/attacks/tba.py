"""TBA as an Attack (faithful; arXiv:2608.08303). Trajectory-level, fixed recipe.
Wraps attack/tba/tba_recipe and binds a pluggable TargetAction for B*."""
from __future__ import annotations

from attack.attacks.base import Attack, PoisonPayload
from attack.evaluation.targets import TargetAction, make_target
from attack.tba import tba_recipe as R


class TBAAttack:
    name = "tba"

    def __init__(self, target: TargetAction | str = "beacon", cfg: R.TBAConfig | None = None):
        self.target = make_target(target) if isinstance(target, str) else target
        self.cfg = cfg or R.TBAConfig()

    def build_payload(self, base_tasks: list[dict]) -> PoisonPayload:
        def token_fn(bid, k):
            return f"AUD-{bid}-{k}"
        overrides = {}
        if not self.target.accuracy_invisible:
            action = self.target.describe()
            overrides = {
                "counterfactual": R.degradation_context("ARCHIVAL_AUDIT sheet", action),
                "externalization": "",
            }
        batch = R.build_batch(base_tasks, token_fn, self.cfg, **overrides)
        # embed the pluggable target-action description into each attack query
        desc = self.target.describe()
        for q in batch.triggered + batch.clean:
            q["query"] = q["query"].replace("<policy>", desc)
        return PoisonPayload(
            kind="trajectory",
            items=batch.triggered + batch.clean,
            triggered_ids=[q["base_id"] for q in batch.triggered],
            clean_ids=[q["base_id"] for q in batch.clean],
            meta={"recipe": "counterfactual+externalization+cross-task",
                  "target": self.target.key, "ratio": batch.ratio},
        )
