#!/usr/bin/env bash
# Clone the four evolution frameworks at the commits the paper used and apply our patches
# (third_party/README.md). Safe to re-run: existing checkouts are moved to the pinned commit and an
# already-applied patch is left alone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EM="$ROOT/evolution_methods"
mkdir -p "$EM"

checkout() {
  local name="$1" url="$2" sha="$3"
  local dir="$EM/$name"
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$url" "$dir"
  fi
  if ! git -C "$dir" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$dir" fetch --quiet origin "$sha"
  fi
  git -C "$dir" checkout --quiet --detach "$sha"
  echo "[setup] $name @ $(git -C "$dir" rev-parse --short HEAD)"
}

apply_patch() {
  local name="$1" patch="$ROOT/third_party/patches/$1.patch"
  local dir="$EM/$name"
  if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
    echo "[setup] $name: patch already applied"
  else
    git -C "$dir" apply "$patch"
    echo "[setup] $name: applied $(basename "$patch")"
  fi
}

checkout Trace2Skill  https://github.com/Qwen-Applications/Trace2Skill.git 3d0b52a140f002a512930252b613c49048f7d5ac
checkout SkillGrad    https://github.com/wwwhy725/SkillGrad.git            9ecd0a633833a1cf21f6f94d8df42bcffaa66554
checkout SkillOpt     https://github.com/microsoft/SkillOpt.git            c1ac570d944ee7f83fc7c4273abfcb4bfdfea392
checkout A-EVOLVE-V3  https://github.com/A-EVO-Lab/a-evolve.git            0a97a300b4ffa4150f8d6216db0b08edbe950f2f

apply_patch SkillGrad

test -f "$EM/Trace2Skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400/dataset.json" \
  && echo "[setup] SpreadsheetBench Verified-400 present" \
  || echo "[setup] WARNING: SpreadsheetBench Verified-400 not found in the Trace2Skill checkout"
