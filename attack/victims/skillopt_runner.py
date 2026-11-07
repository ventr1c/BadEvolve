"""SkillOpt VictimRunner - framework-axis sibling of Trace2SkillRunner / SkillGradRunner.

RQ1's framework axis varies the *evolution method* while holding the poison injection AND the
deployment solver fixed (the "dilute the harness" framing).  So this runner subclasses
Trace2SkillRunner and overrides ONLY evolve(): the poison enters as the SAME SpreadsheetBench
poison tasks, the held-out solve uses the SAME Trace2Skill solver, and only CONSOLIDATION differs.

SkillOpt's consolidation is a batched textual-gradient train loop (`scripts/train.py` ->
minibatch reflect -> skill edits -> `best_skill.md`).  We hand it a `split_dir` whose train split
is EXACTLY the poison tasks, then stage the produced `best_skill.md` as the standard
`xlsx/SKILL.md` tree the fixed solver and `_rule_where` already read.

Why a split_dir (not the ratio splitter): `split_mode=split_dir` consumes train/val/test verbatim,
so "poison enters train ONLY" is structural -- val is the frozen RQ1 dev, the same 40 clean items the
clean arm's selection gate scores; test is empty (the reported test is solved by the fixed solver).

Model routing (no upstream edit): SkillOpt's `qwen_chat` backend takes an arbitrary
`--{optimizer_,target_}qwen_chat_base_url` + api key, so both the optimizer LLM and the target
solver go through our LiteLLM Bedrock proxy under any model name the proxy exposes.

Victim repo stays pristine (repository policy): every output goes under `workdir`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from attack.victims.evolution.skillopt_gate import exported_skill, gate_variant
from attack.victims.trace2skill_runner import REPO, Trace2SkillRunner
from attack.endpoints import API_KEY as _GATEWAY_KEY

SO = REPO / "evolution_methods" / "SkillOpt"
# Clean workbooks for the val items. SkillOpt resolves every item as
# data_root + item['spreadsheet_path'], and data_root has to be the poison dir
# (that is where the poison workbooks live), so the clean workbooks must be
# reachable under that SAME root or their envs never build.
OFFICIAL_WORKBOOKS = (
    REPO
    / "dataset/spreadsheetbench/official_verified_400"
    / "spreadsheetbench_verified_400"
    / "spreadsheet"
)


def _so_env() -> dict:
    return {**os.environ, "PYTHONPATH": str(SO), "AWS_REGION": "us-west-2",
            "OPENAI_API_KEY": _GATEWAY_KEY,
            "TOKENIZERS_PARALLELISM": "false",
            # `subprocess.call(stdout=log)` gives the child no tty, so Python block-buffers at 4-8 KB
            # and `so_train.log` can sit at 2 lines for HOURS while the train loop is working
            # normally. That cost two wrong "the SkillOpt trains are wedged" diagnoses on 2026-09-15
            # (the evolve dir was writing 105 files per 10 min with 1086 predictions at the time).
            # Unbuffered output makes the log a real-time progress signal instead of a lagging one.
            "PYTHONUNBUFFERED": "1"}


def _link_clean_workbooks(clean_items: dict, poison_dataset: Path) -> int:
    """Symlink each clean item's workbook dir into the poison data_root.

    Returns the number of items whose workbook is now resolvable. Symlinks (not copies) because
    the official workbooks are read-only inputs and a run materializes this per evolve. A missing
    source is skipped rather than fatal -- the caller records the count so a shortfall is visible
    in the manifest instead of silently degrading selection to 0."""
    dest_root = poison_dataset / "spreadsheet"
    dest_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    for items in clean_items.values():
        for it in items:
            rel = str(it.get("spreadsheet_path", "")).strip()
            if not rel:
                continue
            name = Path(rel).name
            src = OFFICIAL_WORKBOOKS / name
            if not src.is_dir():
                continue
            dest = dest_root / name
            if dest.exists() or dest.is_symlink():
                linked += 1
                continue
            dest.symlink_to(src, target_is_directory=True)
            linked += 1
    return linked


class SkillOptRunner(Trace2SkillRunner):
    name = "SkillOpt"
    evolution_mode = "batched minibatch-reflect skill optimization (SkillOpt)"
    mechanism_key = "skillopt_sb"           # failure_only=False, gate on the frozen dev (see evolve)

    def __init__(self, model="claude-sonnet-4.6", max_turns=30, workers=4, seed=0,
                 endpoint="http://localhost:8000/v1", skill_mode="evolved",
                 evolver="skillopt", batch_size=8, minibatch_size=4, n_epochs=1):
        # skill_mode seed/none reuse the parent's solver-side behavior verbatim.
        super().__init__(model=model, max_turns=max_turns, workers=workers, seed=seed,
                         endpoint=endpoint, skill_mode=skill_mode, evolver="error")
        self.batch_size, self.minibatch_size, self.n_epochs = batch_size, minibatch_size, n_epochs
        self._endpoint = endpoint

    # inject() and solve() are inherited unchanged: same poison-task materialization, same
    # fixed held-out solver. Only evolve differs -- that IS the framework axis.

    def _materialize_split(self, poison_dataset: Path, arm: Path) -> dict:
        """split_dir whose TRAIN is exactly the poison tasks; val is the frozen RQ1 dev; test is empty.

        val was the pre-RQ1 `skillopt_verified_400_splits/val` until 2026-10-14: only 3 of its 40
        items are in the frozen dev and 11 are in the REPORTED rq1 test, so the gate selected skills
        partly on test tasks, on a set the clean arm (frozen dev) never used. test was that split's 280
        items, which `eval_test` then solved 3x per evolve for a score nothing reads.

        SkillOpt reads spreadsheets from `data_root` + item['spreadsheet_path'], and the poison
        workbooks live under the poison dataset dir, so that dir is the data_root and the clean
        val items are copied in with their (already relative) paths intact.

        The clean items' WORKBOOKS also have to be reachable under that same data_root, or every
        selection env builds empty (`turns=0 cases=0/0 dt=0s`) and selection_hard is 0.0000 for
        the baseline AND for every candidate -- so no candidate can ever improve on the baseline
        and the accept gate rejects all of them by construction. That is a harness artifact, not
        SkillOpt resisting the poison, so link the official workbooks in next to the poison ones."""
        entries = json.load(open(poison_dataset / "dataset.json"))
        train = [{"id": e["id"], "instruction": e.get("instruction", ""),
                  "spreadsheet_path": f"spreadsheet/{e['id']}",
                  "instruction_type": e.get("instruction_type", "Cell-Level Manipulation"),
                  "answer_position": e.get("answer_position", "")} for e in entries]
        from attack.datasets.benchmark import Benchmark
        dev = Benchmark("spreadsheetbench").load_splits(split_names=("dev",))["dev"]
        leak = {str(t["id"]) for t in train} & {str(t["id"]) for t in dev}
        if leak:
            raise ValueError(f"poison ids collide with the frozen dev: {sorted(leak)[:5]}")
        sd = arm / "split"
        for name, items in (("train", train), ("val", dev), ("test", [])):
            (sd / name).mkdir(parents=True, exist_ok=True)
            (sd / name / "items.json").write_text(json.dumps(items, indent=1))
        n_linked = _link_clean_workbooks({"val": dev}, poison_dataset)
        (sd / "split_manifest.json").write_text(json.dumps(
            {"arm": "poison", "n_train": len(train), "poison_ids": [str(e["id"]) for e in entries],
             "n_clean_workbooks_linked": n_linked, "selection": {"split": "rq1_dev", "n": len(dev)},
             "note": "train == the poison tasks ONLY (poison-in-train-only is structural here); "
                     "val is the frozen rq1 dev (the clean arm's selection set); test is empty"},
            indent=1))
        return {"split_dir": str(sd), "n_train": len(train), "n_val": len(dev)}

    def _evolve_native(self, workdir: str) -> list[str]:
        """SkillOpt train loop over the poison split -> staged xlsx skill."""
        wd = Path(workdir)
        from attack.victims import neutral_names
        pd = neutral_names.dataset_dir(wd)
        arm = wd / "so"; arm.mkdir(exist_ok=True)
        man = self._materialize_split(pd, arm)
        out_root = arm / "out"
        n_train = max(man["n_train"], 1)
        bs = min(self.batch_size, n_train)
        ep = self._endpoint if self._endpoint.rstrip("/").endswith("/v1") \
            else self._endpoint.rstrip("/") + "/v1"
        # Poison tasks are VALID SpreadsheetBench tasks, so `failure_only` reflection would find
        # nothing to distil from; reflect on every trajectory instead (the attack's whole point is
        # that the poison trajectories look like ordinary successful work).
        gate = gate_variant()      # strict = upstream; tie / off are recorded protocol variants (skillopt_gate.py)
        from attack.victims.clean_startpoint import stage as stage_clean_startpoint
        initial = stage_clean_startpoint(self, wd / "skillopt_initial")
        if initial and neutral_names.enabled():
            # clean_startpoint.clone wrote this attempt's real path into the skill's recalc.py calls; the SkillOpt
            # stage sees /mnt/ws/... (neutral_names.stage_path), so repoint them before training reads the skill.
            from attack.victims.trace2skill_runner import _absolutise_recalc_refs
            _absolutise_recalc_refs(Path(initial))
        initial_md = initial / "xlsx/SKILL.md" if initial else None
        launch = [sys.executable, *([str(Path(__file__).with_name("skillopt_tie_train.py"))] if gate == "tie" else [])]
        cmd = [*launch, "scripts/train.py",
               "--config", "configs/spreadsheetbench/default.yaml",
               "--out_root", str(out_root),
               "--backend", "qwen_chat",
               "--optimizer_model", self.model, "--target_model", self.model,
               "--optimizer_backend", "qwen_chat", "--target_backend", "qwen_chat",
               "--qwen_chat_base_url", ep, "--qwen_chat_api_key", _GATEWAY_KEY,
               "--optimizer_qwen_chat_base_url", ep, "--optimizer_qwen_chat_api_key", _GATEWAY_KEY,
               "--target_qwen_chat_base_url", ep, "--target_qwen_chat_api_key", _GATEWAY_KEY,
               "--cfg-options",
               f"env.split_mode=split_dir", f"env.split_dir={man['split_dir']}",
               f"env.data_root={pd}", f"env.max_turns={self.max_turns}",
               f"env.workers={self.workers}",
               # failure_only lives under `gradient` (configs/_base_/default.yaml). Poison tasks are
               # VALID SB tasks, so failure-only reflection would find nothing to distil from;
               # reflect on every trajectory (the attack's point is that poison work looks ordinary).
               "gradient.failure_only=False",
               f"train.train_size={n_train}", f"train.batch_size={bs}",
               f"train.num_epochs={self.n_epochs}",
               f"gradient.minibatch_size={min(self.minibatch_size, bs)}",
               f"gradient.merge_batch_size={min(self.minibatch_size, bs)}",
               # Selection gate exactly as the clean arm runs it (clean config.json: sel_env_num 40,
               # slow_update_gate_with_selection True, eval_test False). sel_env_num=0 meant ALL val
               # items in SkillOpt's loader, so the per-step gate was always live; only the slow update
               # was force-accepted, and eval_test solved the old 280-item test 3x per evolve.
               f"evaluation.sel_env_num={man['n_val']}", "evaluation.test_env_num=0",
               f"optimizer.slow_update_gate_with_selection={gate != 'off'}", "evaluation.eval_test=False",
               *(["evaluation.use_gate=False"] if gate == "off" else [])]
        if initial_md:
            cmd += [f"env.skill_init={initial_md}"]
        # PROVENANCE, written BEFORE the subprocess starts. The QA path records the effective gate
        # twice (evolution/skillopt.py: config.json and native_summary.json); this path recorded it
        # nowhere and never logs the argv, so after an SB tie cell nothing on disk proved that
        # skillopt_tie_train.py had been prepended -- and meta.json's SKILLOPT_GATE echo is not
        # evidence, it is the raw env string rather than the gate the victim obeyed. bridge_argv is
        # the discriminator: cmd[1] is skillopt_tie_train.py under tie and scripts/train.py otherwise.
        (arm / "gate.json").write_text(json.dumps({
            "skillopt_gate": gate,
            "tie_bridge": gate == "tie",
            "bridge_argv": cmd[:3],
            "use_gate": gate != "off",
            "slow_update_gate_with_selection": gate != "off",
            "eval_test": False,
        }, indent=2) + "\n")
        log = open(wd / "so_train.log", "w")
        try:
            from attack.evaluation import runlog
            runlog.set_progress(str(wd.parent), substage="so_train")
        except Exception:
            pass
        env = _so_env()
        # POISON_FS_GUARD=1: SkillOpt's agents write scratch files into their cwd (the checkout); under
        rc = subprocess.call(cmd, cwd=str(SO), env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        log.close()
        if rc != 0:
            raise RuntimeError(f"SkillOpt train failed rc={rc} (see {wd}/so_train.log)")
        return self._stage_skill(wd, out_root)

    def _stage_skill(self, wd: Path, out_root: Path) -> list[str]:
        """Copy SkillOpt's produced skill markdown into the standard xlsx/ tree.

        SkillOpt writes ONE markdown file (`best_skill.md`, plus per-step `skill_*.md`), whereas the
        fixed solver + `_rule_where` read `skills/xlsx/SKILL.md` -- hence the restage. Prefer
        best_skill.md (its own selection); fall back to the highest-numbered step.

        VERIFIED layout (smoke, 2026-09-14): `out/best_skill.md` plus versioned steps under
        `out/skills/skill_v0000.md` -- the steps live in a `skills/` SUBDIR, so the fallback must
        search recursively or it finds nothing when the run stops before best_skill.md is written."""
        from attack.victims.trace2skill_runner import (V as T2S, _harden_recalc,
                                                       _absolutise_recalc_refs,
                                                       _ensure_recalc_contract,
                                                       _ensure_skill_discoverable)
        best = exported_skill(out_root, gate_variant())
        if not best.exists():
            steps = sorted(out_root.rglob("skill_*.md"),
                           key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or -1))
            if not steps:
                raise RuntimeError(f"SkillOpt produced no skill markdown under {out_root}")
            best = steps[-1]
        text = best.read_text(errors="replace")
        if not text.strip():
            raise RuntimeError(f"SkillOpt produced an EMPTY skill at {best}")
        # NO-OP DETECTION. SkillOpt accepts a candidate only if it beats the incumbent on the
        # selection split; a step that accepts nothing returns the PREVIOUS skill, which can be the
        # untouched seed. This runner never recorded that (`_last_evolve_noop` appeared 0 times here
        # while trace2skill_runner and qa_framework_runner both set it), so `scripts/rq1_table.py`
        # fell back to grepping `evo_skillevolve.log` -- a file SkillOpt never writes -- which
        # resolves to 0.0 and ASSERTS "no no-ops" rather than "unknown".
        # Measured over the artifacts: 14 of 50 SB SkillOpt evolves deployed a `best_skill.md`
        # byte-identical to the pristine upstream seed, across 5 of 10 cells (one backbone's ours 5/5,
        # and tg 5/5, qwen_ours 2/5, gptoss_ours 1/5, qwen_tg 1/5) -- and all 10 cells assert
        # n_evolve_noop=0. Those evolves measured an UNPOISONED skill.
        # The trainer started from the staged clean checkpoint when there is one (_evolve_native).
        initial_md = wd / "skillopt_initial/xlsx/SKILL.md" if getattr(self, "clean_startpoint", None) else None
        _seed_md = initial_md or SO / "skillopt/envs/spreadsheetbench/skills/initial.md"
        self._last_evolve_noop = False
        try:
            if _seed_md.exists() and text.strip() == _seed_md.read_text(errors="replace").strip():
                self._last_evolve_noop = True
        except OSError:
            pass
        if self._last_evolve_noop:
            try:   # what the gate scored on; rq1_table admits a no-op only if it is the frozen dev
                selection = json.load(open(wd / "so/split/split_manifest.json")).get("selection")
            except (OSError, ValueError):
                selection = None
            (wd / "EVOLVE_NOOP.json").write_text(json.dumps(
                {"noop": True, "skill_source": str(best), "selection": selection,
                 "why": "best_skill.md is byte-identical to the pristine SkillOpt seed -- the gate "
                        "accepted no candidate, so the run returned the untouched skill",
                 "consequence": ("SkillOpt's gate on the frozen clean dev rejected every candidate: "
                                 "the attack FAILED at induction (a real defence outcome)"
                                 if (selection or {}).get("split") == "rq1_dev"
                                 else "this evolve measured an UNPOISONED skill; any ASR from it is "
                                      "not an attack result")}, indent=1))
            print(f"[skillopt] WARNING evolve was a NO-OP: best_skill.md == pristine seed "
                  f"({best}). This evolve tests the SEED, not the attack.", flush=True)
        sk = wd / "skills"
        if not (sk / "xlsx-35B" / "SKILL.md").exists():
            shutil.copytree(T2S / "spreadsheet_agent/skills", sk, dirs_exist_ok=True)
            _harden_recalc(sk)
        (sk / "xlsx").mkdir(parents=True, exist_ok=True)
        (sk / "xlsx" / "SKILL.md").write_text(text)
        # written AFTER _harden_recalc -> re-points bare `python recalc.py` at the staged script so
        # the agent never falls back to `find / -name recalc.py` (see aevolve_runner).
        _absolutise_recalc_refs(sk)
        # ...and the evolved text may not mention recalc AT ALL, which silently drops upstream's
        # "MANDATORY IF USING FORMULAS" obligation: measured 100% of formula cells left with no cached
        # value for SkillOpt/SkillGrad, so correct formulas graded as wrong (development log 2026-09-14).
        # `_harden_recalc` re-asserts it, but it is NOT called on this path (the copytree above is
        # guarded by `if not exists`), so call it explicitly here.
        _ensure_skill_discoverable(sk)
        _ensure_recalc_contract(sk)
        (wd / "so_skill_source.txt").write_text(str(best))
        self._evolved_skill_dir = str(sk)
        return [md.read_text(errors="replace") for md in sk.rglob("SKILL.md")]
