# BadEvolve

This is the official implementation for **When Experience Becomes a Backdoor: Poisoning Self-Evolving LLM Agents via Failure-Aware Task Optimization**.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Third-Party Frameworks](#third-party-frameworks)
  - [Environment Variables](#environment-variables)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Outputs](#outputs)
- [Baselines](#baselines)

## Overview

Self-evolving agents (Trace2Skill, SkillGrad, SkillOpt, A-Evolve) learn reusable skills from the training tasks they
solve. BadEvolve poisons a few of those tasks so that the skill the agent writes for itself contains a backdoor:
on inputs with a trigger the agent follows the attacker's behavior, on clean inputs it behaves normally.

BadEvolve finds the poisoned tasks with an LLM optimizer over L rounds. Each round:

1. the optimizer proposes a poisoning policy, which becomes B poisoned tasks in the training pool;
2. the victim framework evolves K times on that pool;
3. the resulting skills are scored on the dev set with stage-wise feedback (induction / activation / following / effect),
  and the proposal is accepted or rolled back by the failure-aware gate; rejected proposals stay in the
   optimizer's failure memory.

## Project Structure

```
BadEvolve/
├── scripts/
│   │   # the two entry points: each runs one complete attack (poisoned pool -> L rounds of
│   │   # propose / evolve K times / measure / accept-or-rollback -> final test) and writes runs/<tag>/
│   ├── run_wikitq_optimization.py   #   on the QA benchmarks (--dataset wikitq | officeqa)
│   ├── run_agent_optimization.py    #   on SpreadsheetBench (Excel-agent victims)
│   │   # one-off helpers
│   ├── setup_third_party.sh         #   before running: clone the four victim frameworks at the paper's commits
│   ├── build_officeqa_splits.py     #   before running OfficeQA: download the gated data and rebuild our splits
│   └── rereview_final_test.py       #   after a QA run: re-score its final test with the official grader
│
├── examples/                        # one-command experiment scripts (see Quick Start / Experiments)
│   ├── run_qa.sh, run_sb.sh         # BadEvolve on one cell (QA / SpreadsheetBench) with the paper's settings;
│   │                                #   GATE_MODE=textgrad runs the TextGrad-Poison baseline, GATE_MODE=tba the TBA baseline
│   ├── run_main_table.sh            # one row of the main table: BadEvolve, TextGrad-Poison and TBA on one dataset x framework x backbone
│   ├── run_clean_evolution.sh       # CACC_0: the framework evolves with no poison
│   └── run_ablations.sh             # BadEvolve ablations: full, no_stage_feedback, no_rollback, no_failure_memory
│
├── attack/
│   ├── protocol/                    # arms.py (BadEvolve / TextGrad-Poison / TBA), measurement.py (the gate), loop.py
│   ├── agent/                       # optimizer prompt, stage-wise feedback, ablation views, Claude Code harness + sandbox
│   ├── pipeline/                    # training_pool.py (40 clean + B poisoned), materialize_poison.py, round_trace.py
│   ├── attacks/, tba/               # poisoning policy and its init; the TBA recipe
│   ├── evaluation/                  # triggers, payloads and stage detectors (targets.py, qa_targets.py), objective.py, metrics.py
│   ├── clean/                       # paired measurement (qa_measurement.py), official re-review, shared solver/evolver runtimes
│   ├── victims/                     # adapters for Trace2Skill, SkillGrad, SkillOpt, A-Evolve (QA + SpreadsheetBench)
│   ├── datasets/                    # frozen split loading
│   └── endpoints.py                 # model endpoint configuration
│
├── dataset/                         # frozen splits: WikiTQ (shipped), OfficeQA (ids + hashes), SpreadsheetBench ids
├── configs/
│   ├── paper.env                    # environment variables every paper run used
│   └── models.example.yaml          # example LiteLLM gateway serving the four backbones
├── third_party/                     # upstream URL, commit and license of each framework; our SkillGrad patch
├── tests/test_smoke.py              # dry-run smoke test, no model calls
├── requirements.txt / requirements.lock
└── .env.example
```

## Installation

```bash
git clone <this repository> BadEvolve
cd BadEvolve

python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock     # the paper's pinned environment (requirements.txt lists the direct dependencies)
```

Also needed on the host: LibreOffice (`soffice`; SpreadsheetBench recalculation and grading), `git`, `bash`, and the
[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`), which runs the optimizer agent.

### Third-Party Frameworks

The four victim frameworks are not vendored. `scripts/setup_third_party.sh` clones each one at the commit the
paper used into `evolution_methods/` and applies our one patch (two SkillGrad tool-use fixes). Upstream URLs,
commits and licenses are listed in `third_party/README.md`.

```bash
bash scripts/setup_third_party.sh
python -m pytest -q tests/test_smoke.py     # imports everything, builds the 40+20 training pool, calls no model
```

### Environment Variables

Copy the example file and fill it in:

```bash
cp .env.example .env
set -a; . .env; set +a
```


| Variable                       | Description                                                                                                            |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `BADEVOLVE_BASE_URL`           | OpenAI-compatible endpoint the victims (solver and evolver of every framework) call, e.g. `http://localhost:8000/v1`   |
| `BADEVOLVE_API_KEY`            | its key (`EMPTY` if none)                                                                                              |
| `BADEVOLVE_ENDPOINTS`          | optional JSON `{model: base_url}` when backbones live behind different gateways                                        |
| `BADEVOLVE_OPTIMIZER_PROVIDER` | `anthropic` (default; the CLI's login or `ANTHROPIC_API_KEY`) or `bedrock` (`CLAUDE_CODE_USE_BEDROCK=1`, `AWS_REGION`) |
| `BADEVOLVE_OPTIMIZER_MODEL`    | optimizer model id (default Claude Sonnet 4.6)                                                                         |
| `HF_TOKEN`                     | Hugging Face token, only for rebuilding the gated OfficeQA splits                                                      |


The victim backbones are addressed by name through `--victim-model`: `kimi-k2.5` (Kimi-K2.5, main table),
`gptoss-120b` (GPT-OSS-120B), `claude-sonnet-4.6` (Claude Sonnet 4.6), `qwen3-235b` (Qwen3-235B-A22B).
`configs/models.example.yaml` is a LiteLLM config serving those names (`litellm --config configs/models.example.yaml --port 8000`); any OpenAI-compatible server works.

The per-run switches of the paper (optimizer backend, neutral naming, beacon placement, token budgets, …) are in
`configs/paper.env`; the example scripts source it, and a direct driver call needs `set -a; . configs/paper.env; set +a`.

## Dataset Preparation


| Dataset          | Train / dev / test | Preparation                                                                                                                                                                                               |
| ---------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WikiTQ           | 40 / 40 / 100      | shipped: `dataset/wikitq/rq1_{train,dev,test}/items.json` (WikiTableQuestions, CC BY-SA 4.0) with the official grading annotations                                                                        |
| OfficeQA         | 50 / 40 / 100      | gated (`databricks/officeqa`); the split identity and content hashes are shipped (`dataset/officeqa/`), `HF_TOKEN=... python scripts/build_officeqa_splits.py` rebuilds the items and verifies every hash |
| SpreadsheetBench | 40 / 40 / 100      | Verified-400, taken from the Trace2Skill checkout after `setup_third_party.sh`; split ids in `dataset/rq1_splits.json`                                                                                    |


Poison enters train only. Every run's training pool is the first 40 train tasks plus B poisoned tasks built from the
first B of them (`--clean-train-n 40 --n-base B`; B = 20 in the main table). Dev selects, test reports.

## Quick Start

BadEvolve on WikiTQ × SkillOpt × Kimi-K2.5 (one main-table cell):

```bash
bash examples/run_qa.sh wikitq skillopt kimi-k2.5
```

and on SpreadsheetBench × SkillGrad × Kimi-K2.5:

```bash
bash examples/run_sb.sh skillgrad kimi-k2.5
```

`run_qa.sh <dataset> <framework> <victim-model>` takes `wikitq|officeqa`, `run_sb.sh <victim> <victim-model>`
SpreadsheetBench; the framework is one of `trace2skill`, `skillgrad`, `skillopt`, `aevolve`. The scripts are thin
wrappers around the drivers, whose defaults are the paper's settings, so the first one is the same as

```bash
set -a; . configs/paper.env; set +a
python scripts/run_wikitq_optimization.py --dataset wikitq --framework skillopt --victim-model kimi-k2.5 --tag my_run
```

The defaults: `--gate-mode ours_v5` (BadEvolve), L = 5 rounds (`--rounds`), K = 5 evolutions per candidate (`--k`),
B = 20 poisoned tasks on the 40 frozen train tasks (`--n-base`, `--clean-train-n`), 40 dev tasks, 100 paired test
tasks, 5 fresh evolutions at test time (`--final-k`). `python scripts/run_wikitq_optimization.py --help` lists every
flag; `--resume` continues a stopped run. A full run is 5 rounds × 5 evolutions plus the final test: several hours
and a few thousand model calls, depending on framework and backbone.

## Experiments

The example scripts take the experiment knobs as environment variables (`GATE_MODE`, `ATTACK_TYPE`, `N_BASE`,
`ROUNDS`, `ABLATION`, `SURROGATE_MODEL`, `TAG`); `<dataset>` is `wikitq`, `officeqa` or `spreadsheetbench`.


| Experiment                                                                  | Command                                                                     |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Main table (BadEvolve, TextGrad-Poison, TBA)                                | `bash examples/run_main_table.sh <dataset> <framework> <victim-model>`      |
| CACC₀ (clean evolution, no poison)                                          | `bash examples/run_clean_evolution.sh <dataset> <framework> <victim-model>` |
| Ablations (`full`, `no_stage_feedback`, `no_rollback`, `no_failure_memory`) | `bash examples/run_ablations.sh <dataset> <framework> <victim-model>`       |
| Poison budget                                                               | `N_BASE=1|5|20|30|40 bash examples/run_qa.sh …` / `run_sb.sh …`             |
| Answer corruption                                                           | `ATTACK_TYPE=degradation bash examples/run_qa.sh …` / `run_sb.sh …`         |
| Surrogate transfer                                                          | `SURROGATE_MODEL=gptoss-120b bash examples/run_sb.sh skillgrad kimi-k2.5`   |
| Longer optimization                                                         | `ROUNDS=10 bash examples/run_qa.sh …` / `run_sb.sh …`                       |


## Outputs

Each run writes `runs/<arm>__<dataset|framework>__<attack>__<tag>/`. The paper's numbers come from
`final_test.json`:


| File              | Content                                                                                                                                                                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `final_test.json` | the committed poison evaluated with `--final-k` fresh evolutions on the paired test split: `TASR_mean` (ASR), `TACC_mean`, `CACC_mean`, the four stage indicators (induction / activation / following / effect), `DASR` for answer corruption |
| `history.json`    | one entry per round: objective before/after, the gate decision (accepted / rejected and why), failure-memory notes                                                                                                                            |
| `round_<r>/`      | that round's proposal and the K dev evolutions that measured it                                                                                                                                                                               |
| `meta.json`       | the resolved configuration of the run                                                                                                                                                                                                         |


For QA runs, `python scripts/rereview_final_test.py <run dir>` re-scores the final test with the official grader
and writes `final_test_v3.json`.

## Baselines

Both baselines share every measurement flag with BadEvolve and differ only in `--gate-mode`:


| Baseline            | `GATE_MODE` | What it does                                                                                                              |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------- |
| **TextGrad-Poison** | `textgrad`  | textual-gradient optimization of the poisoned tasks from a scalar objective: no stage-wise feedback, no gate, no rollback |
| **TBA**             | `tba`       | a fixed trigger-behavior-association recipe; no optimization rounds                                                       |


```bash
GATE_MODE=textgrad bash examples/run_qa.sh wikitq skillopt kimi-k2.5
GATE_MODE=tba      bash examples/run_qa.sh wikitq skillopt kimi-k2.5
```

