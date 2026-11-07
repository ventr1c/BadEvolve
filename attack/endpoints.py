"""Model gateway configuration.

Every victim backbone (solver and evolver) is reached through an OpenAI-compatible chat-completions
endpoint, e.g. a LiteLLM proxy or a vLLM server, that serves the model names used on the command line
(`kimi-k2.5`, `gptoss-120b`, `claude-sonnet-4.6`, `qwen3-235b`; see configs/models.example.yaml).

  BADEVOLVE_BASE_URL   default --endpoint of both drivers (default http://localhost:8000/v1)
  BADEVOLVE_API_KEY    API key sent to that endpoint (default "EMPTY", for gateways without auth)
  BADEVOLVE_ENDPOINTS  optional JSON {model: base_url} when backbones live behind different gateways
"""
from __future__ import annotations

import os

DEFAULT_BASE_URL = os.environ.get("BADEVOLVE_BASE_URL") or "http://localhost:8000/v1"
API_KEY = os.environ.get("BADEVOLVE_API_KEY") or "EMPTY"
