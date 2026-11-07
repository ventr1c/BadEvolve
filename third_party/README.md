# Third-party self-evolving frameworks

BadEvolve attacks four existing skill-evolution frameworks. They are **not vendored**: `scripts/setup_third_party.sh`
clones each one at the exact commit the paper used into `evolution_methods/<name>/` (the directory names are the
ones the code expects) and applies our patches. Everything else we need from them is done by adapters in
`attack/victims/` that call the upstream code without modifying it.

| Directory (`evolution_methods/…`) | Upstream | Commit used in the paper | License (as declared upstream) | Our changes |
|---|---|---|---|---|
| `Trace2Skill` | https://github.com/Qwen-Applications/Trace2Skill | `3d0b52a140f002a512930252b613c49048f7d5ac` | Apache-2.0 (stated in its README; the repository has no LICENSE file) | none (adapter: `attack/victims/trace2skill_runner.py`, `attack/victims/evolution/trace2skill.py`) |
| `SkillGrad` | https://github.com/wwwhy725/SkillGrad | `9ecd0a633833a1cf21f6f94d8df42bcffaa66554` | MIT | `patches/SkillGrad.patch` (see below); adapters `attack/victims/skillgrad_runner.py`, `attack/victims/skillgrad_adapter.py`, `attack/victims/evolution/skillgrad.py` |
| `SkillOpt` | https://github.com/microsoft/SkillOpt | `c1ac570d944ee7f83fc7c4273abfcb4bfdfea392` | MIT | none (adapters `attack/victims/skillopt_runner.py`, `attack/victims/evolution/skillopt*.py`, `attack/victims/officeqa_clean.py`) |
| `A-EVOLVE-V3` | https://github.com/A-EVO-Lab/a-evolve | `0a97a300b4ffa4150f8d6216db0b08edbe950f2f` | MIT (declared in its `pyproject.toml`; the repository has no LICENSE file) | none (adapters `attack/victims/aevolve_*.py`, `attack/clean/aevolve_*.py`; the upstream Bedrock provider is replaced at runtime by its OpenAI provider, see `attack/victims/aevolve_clean.py`) |

The SpreadsheetBench data (SpreadsheetBench Verified-400) comes from the Trace2Skill checkout
(`data/spreadsheetbench_verified/spreadsheetbench_verified_400`). The WikiTQ official evaluator is vendored under
`attack/evaluation/vendor/wikitq/` with its CC BY-SA 4.0 license and provenance.

## `patches/SkillGrad.patch`

Two tool-use fixes that were in effect in the SkillGrad checkout every paper run used:

* `pipeline/helpers.py`: when a model passes a whole `write_file` payload as `read_file.file_path`, return a
  corrective hint instead of a file-not-found error;
* `prompts/momentum.py`: one sentence telling the momentum stage to write its outputs with `write_file`.

Neither touches SkillGrad's evolution algorithm. That checkout also had a modified `seeds/xlsx/SKILL.md` in its
working tree, written there by a victim rollout agent during an earlier, invalidated experiment. It is **not** part
of the patch. Each run stages its own private seed from `git archive HEAD seeds` (see
`attack/victims/skillgrad_runner.py` and `attack/victims/evolution/skillgrad.py`) and checks that it is pristine,
so no paper run started from the poisoned copy, and the code keeps that protection.

The upstream checkouts had no other tracked modifications that the paper's code paths load. SkillOpt's
`scripts/train.py` registers an extra AppWorld environment, which these experiments never use. Trace2Skill's
`analysis/evaluate_output.py` differs only in its shebang. We ship neither change.
