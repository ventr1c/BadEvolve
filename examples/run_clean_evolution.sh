#!/usr/bin/env bash
# CACC_0: the framework evolves on the 40 clean train tasks with no poison, then the paired test is run.
#   bash examples/run_clean_evolution.sh <dataset> <framework> <victim-model>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?}"; FRAMEWORK="${2:?}"; VICTIM_MODEL="${3:?}"
export GATE_MODE=tba N_BASE=0 ROUNDS=0 TAG="${TAG:-${DATASET}_${FRAMEWORK}_${VICTIM_MODEL}_clean}"
if [ "$DATASET" = spreadsheetbench ]; then
  bash "$HERE/run_sb.sh" "$FRAMEWORK" "$VICTIM_MODEL"
else
  bash "$HERE/run_qa.sh" "$DATASET" "$FRAMEWORK" "$VICTIM_MODEL"
fi
