#!/usr/bin/env bash
# BadEvolve and both baselines on one dataset x framework x backbone (one row of the main table), sequentially.
#   bash examples/run_main_table.sh <dataset> <framework> <victim-model>     dataset: wikitq|officeqa|spreadsheetbench
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?}"; FRAMEWORK="${2:?}"; VICTIM_MODEL="${3:?}"
for arm in ours_v5 textgrad tba; do
  if [ "$DATASET" = spreadsheetbench ]; then
    GATE_MODE=$arm bash "$HERE/run_sb.sh" "$FRAMEWORK" "$VICTIM_MODEL"
  else
    GATE_MODE=$arm bash "$HERE/run_qa.sh" "$DATASET" "$FRAMEWORK" "$VICTIM_MODEL"
  fi
done
