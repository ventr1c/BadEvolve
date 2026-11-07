#!/usr/bin/env bash
# The four ablation variants of BadEvolve on one cell: full, no_stage_feedback, no_rollback, no_failure_memory.
#   bash examples/run_ablations.sh <dataset> <framework> <victim-model>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?}"; FRAMEWORK="${2:?}"; VICTIM_MODEL="${3:?}"
for v in full no_stage_feedback no_rollback no_failure_memory; do
  export ABLATION=$v TAG="${DATASET}_${FRAMEWORK}_${VICTIM_MODEL}_abl_$v"
  if [ "$DATASET" = spreadsheetbench ]; then
    bash "$HERE/run_sb.sh" "$FRAMEWORK" "$VICTIM_MODEL"
  else
    bash "$HERE/run_qa.sh" "$DATASET" "$FRAMEWORK" "$VICTIM_MODEL"
  fi
done
