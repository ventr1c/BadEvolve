#!/usr/bin/env bash
# One WikiTQ / OfficeQA run with the paper's settings (the driver's defaults).
#   bash examples/run_qa.sh <dataset> <framework> <victim-model> [extra driver flags...]
#   e.g. bash examples/run_qa.sh wikitq skillopt kimi-k2.5
# Knobs (environment): GATE_MODE=ours_v5|textgrad|tba  ATTACK_TYPE=beacon|degradation  N_BASE=20  ROUNDS=5
#                      SURROGATE_MODEL=  ABLATION=  TAG=
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?dataset: wikitq|officeqa}"; FRAMEWORK="${2:?framework: trace2skill|skillgrad|skillopt|aevolve}"
VICTIM_MODEL="${3:?victim model, e.g. kimi-k2.5}"; shift 3

set -a; . "$ROOT/configs/paper.env"; set +a
[ "$FRAMEWORK" = skillgrad ] && export POISON_SG_TASK_FRAME=prompt

GATE_MODE="${GATE_MODE:-ours_v5}"; ATTACK_TYPE="${ATTACK_TYPE:-beacon}"; N_BASE="${N_BASE:-20}"; ROUNDS="${ROUNDS:-5}"
TAG="${TAG:-${DATASET}_${FRAMEWORK}_${VICTIM_MODEL}_${GATE_MODE}}"
EXTRA=()
[ -n "${SURROGATE_MODEL:-}" ] && EXTRA+=(--surrogate-model "$SURROGATE_MODEL")
[ -n "${ABLATION:-}" ] && EXTRA+=(--study-ablation "$ABLATION")

cd "$ROOT"
python scripts/run_wikitq_optimization.py \
  --dataset "$DATASET" --framework "$FRAMEWORK" --victim-model "$VICTIM_MODEL" \
  --gate-mode "$GATE_MODE" --attack-type "$ATTACK_TYPE" --n-base "$N_BASE" --rounds "$ROUNDS" \
  --tag "$TAG" "${EXTRA[@]}" "$@"
