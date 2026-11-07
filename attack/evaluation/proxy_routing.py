"""Which OpenAI-compatible endpoint serves a given backbone.

A surrogate-transfer run touches TWO backbones in one process (surrogate for the optimizer's inner
loop, target for the final test). When both are served from the endpoint given by `--endpoint`, nothing
needs configuring. When they live behind different gateways, set

    BADEVOLVE_ENDPOINTS='{"claude-sonnet-4.6": "http://host-a:8000/v1", "qwen3-235b": "http://host-b:8000/v1"}'

and each model name is routed to its own base URL; a model that is not listed uses the default.
"""
from __future__ import annotations
import json
import os


def _routes() -> dict:
    raw = os.environ.get("BADEVOLVE_ENDPOINTS") or ""
    if not raw.strip():
        return {}
    routes = json.loads(raw)
    if not isinstance(routes, dict):
        raise ValueError("BADEVOLVE_ENDPOINTS must be a JSON object {model: base_url}")
    return routes


def endpoint_for(model: str, default: str) -> str:
    """Base URL for `model`, or `default` when BADEVOLVE_ENDPOINTS does not list it."""
    return _routes().get(model or "", default)


def endpoint_for_surrogate(model: str, default: str) -> str:
    """Base URL for `model` when it is the SURROGATE (same table as endpoint_for)."""
    return endpoint_for(model, default)
