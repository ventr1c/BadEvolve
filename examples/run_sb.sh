#!/usr/bin/env bash
# One SpreadsheetBench run with the paper's settings (the driver's defaults).
#   bash examples/run_sb.sh <victim> <victim-model> [extra driver flags...]
#   e.g. bash examples/run_sb.sh skillgrad kimi-k2.5
# Knobs (environment): GATE_MODE=ours_v5|textgrad|tba  ATTACK_TYPE=beacon|degradation  N_BASE=20  ROUNDS=5
#                      SURROGATE_MODEL=  ABLATION=  TAG=
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VICTIM="${1:?victim: trace2skill|skillgrad|skillopt|aevolve}"; VICTIM_MODEL="${2:?victim model, e.g. kimi-k2.5}"; shift 2

set -a; . "$ROOT/configs/paper.env"; set +a

GATE_MODE="${GATE_MODE:-ours_v5}"; ATTACK_TYPE="${ATTACK_TYPE:-beacon}"; N_BASE="${N_BASE:-20}"; ROUNDS="${ROUNDS:-5}"
TAG="${TAG:-sb_${VICTIM}_${VICTIM_MODEL}_${GATE_MODE}}"
EXTRA=()
[ "$VICTIM" = aevolve ] && EXTRA+=(--aevolve-native)
[ -n "${SURROGATE_MODEL:-}" ] && EXTRA+=(--surrogate-model "$SURROGATE_MODEL")
[ -n "${ABLATION:-}" ] && EXTRA+=(--study-ablation "$ABLATION")

cd "$ROOT"
python scripts/run_agent_optimization.py \
  --victim "$VICTIM" --victim-model "$VICTIM_MODEL" \
  --gate-mode "$GATE_MODE" --attack-type "$ATTACK_TYPE" --n-base "$N_BASE" --rounds "$ROUNDS" \
  --tag "$TAG" "${EXTRA[@]}" "$@"
