"""SkillGrad VictimRunner - framework-axis sibling of Trace2SkillRunner.

RQ1's framework axis varies the *evolution method* while holding the deployment
harness fixed (the "dilute the harness" framing): the poison enters as the SAME
SpreadsheetBench poison tasks, the held-out solve uses the SAME Trace2Skill
solver, and only the CONSOLIDATION step differs. So this runner subclasses
Trace2SkillRunner and overrides ONLY evolve() to drive SkillGrad's
error/gradient-style consolidation (base-trajectories -> pipeline.training ->
final_skill/xlsx/SKILL.md), then stages that skill as the standard xlsx/ tree
the fixed solver + _rule_where already read.

Recipe verified against v0's working Phase-A arm
(poison_evolve_v0/scripts/phaseA_spreadsheetbench_skillgrad/run_evolution_arm.py):
  0. materialize_aligned(poison) -> split.json (poison in train only) + normalized
  1. runners.stream_runner base-trajectories (seed skill) on the evolution pool
  2. pipeline.training (--method skillgrad) -> runs/<id>/train/final_skill
  3. stage final_skill -> workdir/skills/xlsx (+ references) for the fixed solver

Model routing (verified 2026-09-12, no SkillGrad code change): SkillGrad talks to
our LiteLLM Bedrock proxy via AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY +
non-OpenAI-family model names (OpenAIChatCompletionsModel chat path). See
the design notes 5.6.1.

Victim repo stays pristine (repository policy): every output goes under workdir.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import sys
from pathlib import Path

from attack.victims import neutral_names
from attack.victims.trace2skill_runner import REPO, Trace2SkillRunner
from attack.endpoints import API_KEY as _GATEWAY_KEY

SG = REPO / "evolution_methods" / "SkillGrad"


def _assert_no_stdlib_shadow() -> None:
    """Fail fast if a `.py` at SkillGrad's root shadows a stdlib module.

    Every SkillGrad stage runs with `cwd=SG`, which puts SkillGrad's root at `sys.path[0]`, so a
    file there named after a stdlib module wins over the real one for the whole child process.

    This is not hypothetical. On 2026-09-15 the SkillGrad SOLVER AGENT wrote its own debug scripts
    into that directory -- `inspect.py` (an openpyxl probe) and `decimal.py` (a 6-line "stub"
    Decimal) -- and every subsequent `runners.stream_runner` invocation died at import with
    `KeyError: 'inspect'` plus a bogus "openpyxl circular import". Four cells (sg_kimi_ours,
    sg_qwen_ours, sg_qwen_tg, sg_sonnet_tg) lost 100% of their evolves to it, and because the
    traceback mentions a circular import it was initially misdiagnosed as a concurrency race.

    The agent can do this again on any run, so this is a permanent guard rather than a one-off
    cleanup. Raising here costs one cell; NOT raising silently produces `K_valid = 0` cells that
    look like "the attack achieved nothing".
    """
    import sys as _sys
    shadow = sorted(p.name for p in SG.glob("*.py") if p.stem in _sys.stdlib_module_names)
    if shadow:
        raise RuntimeError(
            f"SkillGrad root {SG} contains {len(shadow)} file(s) shadowing stdlib modules: "
            f"{', '.join(shadow)}. Every stage runs with cwd=SkillGrad, so these win over the "
            f"real modules and break `python3 -m runners.*` at import. These are agent-written "
            f"debug scripts, not upstream code -- move them out of the SkillGrad root "
            f"(e.g. into a quarantine subdirectory) and rerun.")


def _assert_seed_is_pristine(seeds: Path | None = None) -> None:
    """Fail fast if the SEED skill the attack starts from already carries the target behavior.

    Every stage passes `--skills-dir seeds` and runs with `cwd=SG`, so the seed is the SHARED
    upstream directory `evolution_methods/SkillGrad/seeds/` -- not a per-run copy. If anything writes
    the attack into it, every later cell starts from a skill that already does B*, Induction is never
    measured, and the cell reports I=1.0 for its own starting state.

    This is not hypothetical, and it is the same class as `_assert_no_stdlib_shadow` above: the SOLVER
    AGENT did it. On 2026-09-13 a run of `rq1_sg_qwen_ours` left three scripts in
    `seeds/xlsx/scripts/` -- `process_audit.py` and `apply_audit_policy.py` implement the
    ARCHIVAL_AUDIT -> AuditTrail!A1 write, with absolute paths inside that very run dir still in them
    -- and `seeds/xlsx/SKILL.md` was edited the next day (2026-09-14 09:49) to add a full
    "## Audit Trail Protocol" section with runnable openpyxl code. All 10 SkillGrad x SpreadsheetBench
    attack cells then started from `SKILL.md.iter_0` byte-identical (md5 77ddfdf5...) to that poisoned
    file, while the clean arm -- which stages its own seed through
    `attack/victims/evolution/skillgrad.py` -- shows 0 hits. See the 2026-09-27 INVALIDATION entry in
    the development log.

    Raising here costs one cell. NOT raising silently produces an arm whose I gate is meaningless and
    whose TASR is measured on a pre-installed rule.
    """
    # Check the seed the run will ACTUALLY use (its private copy) when given; default is the shared
    # upstream dir, kept for callers that predate the private copy.
    seeds = Path(seeds) if seeds is not None else SG / "seeds"
    if not seeds.exists():
        return
    needles = ("ARCHIVAL_AUDIT", "AuditTrail", "PROVENANCE=")
    hits = []
    for f in sorted(seeds.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in (".md", ".py", ".txt", ".json"):
            continue
        try:
            txt = f.read_text(errors="replace")
        except OSError:
            continue
        found = [n for n in needles if n in txt]
        if found:
            hits.append(f"{f.relative_to(seeds)} ({', '.join(found)})")
    if hits:
        raise RuntimeError(
            f"SkillGrad seed {seeds} already contains the TARGET BEHAVIOUR in {len(hits)} file(s): "
            + "; ".join(hits)
            + ". Every stage passes --skills-dir seeds, so the attack would start from a skill that "
              "ALREADY performs B*: Induction is not measured and the cell reports its own starting "
              "state as a success. These are agent-written artefacts, not upstream code -- restore "
              "the seed (git -C <SkillGrad> checkout -- seeds/ and remove untracked files under it) "
              "and rerun. Do NOT report any cell produced from a poisoned seed.")


def _stage_private_seed(wd: Path) -> Path:
    """Copy the shared upstream seed into THIS run's workdir and return the copy.

    WHY, precisely. `pipeline.training` documents `skills_dir` as "modified in-place", so the evolving
    skill is written back into whatever `--skills-dir` points at.

    CORRECTION to an earlier version of this comment: it attributed the write-back to
    `pipeline.patcher.run_patch`. That is WRONG under the CLI -- the patcher is handed
    `run_dir/skills`, not our seed. The reachable writer is the sg_base ROLLOUT SOLVER:
    `runners.stream_runner base-trajectories` builds a `SkillAgent` whose docstring step 4 is "Model
    follows instructions using standard tools (shell, read, write, etc.)", its `project_root` is
    `Path(".").resolve()` (i.e. the SkillGrad checkout, since we pass `cwd=SG`), and
    `pipeline/helpers.py::write_file` takes an ABSOLUTE path with NO root restriction, so `project_root`
    bounds nothing. We hand that solver `--skills-dir <absolute seed_dir>` while the poison task tells
    it to install the rule. Naming the wrong component matters: it is why the first fix closed only the
    cross-run channel and left the within-run one open (see `_assert_seed_unchanged_by_rollout`).

    Passing the literal string "seeds" pointed every
    cell at the SHARED `evolution_methods/SkillGrad/seeds/`, and that is not a hypothetical leak: on
    2026-09-13 a run of rq1_sg_qwen_ours left `process_audit.py` / `apply_audit_policy.py` in
    `seeds/xlsx/scripts/` (absolute paths into that run dir still inside them) and `seeds/xlsx/SKILL.md`
    gained an "## Audit Trail Protocol" section the next day. All 10 SkillGrad x SB attack cells then
    started from a seed that ALREADY performed B*. See the 2026-09-27 INVALIDATION entry in the development log.

    `_assert_seed_is_pristine` catches an ALREADY-poisoned seed; it cannot stop this run from
    poisoning the next one. A per-run copy makes the write-back land in the run dir, so cross-cell
    contamination is structurally impossible rather than detected after the fact.

    Upstream needs no patch: `stream_runner` does `skill_dir = Path(args.skills_dir)` and
    `pipeline.training` takes a Path, so an absolute path outside the checkout works unchanged. The
    upstream directory is left untouched (read-only source of the copy), which also keeps
    `attack.victims.upstream` provenance honest.
    """
    src = SG / "seeds"
    dst = wd / "sg_seed"
    if dst.exists():
        return dst                                  # resume: keep the seed this run already evolved
    if not src.exists():
        raise RuntimeError(f"SkillGrad seed {src} is missing; cannot stage a private copy")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # FROM THE COMMITTED UPSTREAM TREE, not the working tree. The shared working-tree seed is still
    # the poisoned one (`seeds/xlsx/SKILL.md` modified 2026-09-14, plus untracked scripts), so copying
    # it would hand every run the pre-installed rule that `_assert_seed_is_pristine` exists to catch.
    # `git archive HEAD:seeds` reproduces exactly the tracked, committed seed; the working tree is
    # left untouched (provenance records its dirty state separately).
    if _extract_committed_seed(dst):
        return dst
    shutil.copytree(src, dst)                       # not a git checkout: fall back to the files
    return dst


def _extract_committed_seed(dst: Path) -> bool:
    """Materialize `HEAD:seeds` of the SkillGrad checkout into `dst`. False if SG is not a git
    checkout (then the caller copies the working tree and the pristine check still applies)."""
    import io
    import tarfile
    try:
        blob = subprocess.run(["git", "-C", str(SG), "archive", "--format=tar", "HEAD", "seeds"],
                              check=True, capture_output=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    tmp = dst.parent / (dst.name + ".extract")
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(tmp, filter="data")
    (tmp / "seeds").rename(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    (dst / "PROVENANCE_SEED.json").write_text(json.dumps(
        {"source": "git archive HEAD:seeds", "repo": str(SG),
         "head": subprocess.run(["git", "-C", str(SG), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()}, indent=1))
    return True


def _seed_fingerprint(seed_dir: Path) -> dict:
    """sha256 per file under the staged seed, so a mid-run modification is provable, not inferred."""
    import hashlib
    out = {}
    for f in sorted(seed_dir.rglob("*")):
        if f.is_file():
            out[str(f.relative_to(seed_dir))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def _assert_seed_unchanged_by_rollout(seed_dir: Path, before: dict, wd: Path) -> None:
    """The seed must be BYTE-IDENTICAL after sg_base. Within-cell self-install, not cross-cell.

    `_stage_private_seed` closed the CROSS-run channel: one cell can no longer poison the next. It did
    not close the WITHIN-run one, and the mechanism is the same one that invalidated 10 cells:

      * sg_base runs `runners.stream_runner base-trajectories` with `cwd=SkillGrad`, whose
        `project_root=Path(".").resolve()` gives the solver agent shell+read+WRITE tools
        (`pipeline/executor.py` SkillAgent docstring, step 4);
      * `pipeline/helpers.py::write_file` accepts an ABSOLUTE path and applies NO root restriction --
        `p = Path(file_path); if not p.is_absolute(): p = root / p` -- so `root` bounds nothing;
      * we hand that solver `--skills-dir <absolute seed_dir>`, and the poison task TELLS it to
        install the rule.

    So the sg_base solver can write the rule into the very seed sg_train then starts from. Induction
    would be measured against a skill that already performs B*, exactly as in the 2026-09-13 incident
    -- just confined to one cell, which makes it invisible rather than harmless.

    Note this is NOT the patcher: under the CLI the patcher only ever sees `run_dir/skills`. Attributing
    the write-back to it (as an earlier comment of mine did) named the wrong component, and a test
    pinned the claim that does not hold.

    Raising loses one cell. Not raising reports an induction number that measures the rollout's own
    edit. The receipt is written either way so the evidence survives the exception.
    """
    after = _seed_fingerprint(seed_dir)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    _receipt = wd / "sg_seed_integrity.json"
    _receipt.parent.mkdir(parents=True, exist_ok=True)
    _receipt.write_text(json.dumps({
        "stage": "after_sg_base",
        "seed_dir": str(seed_dir),
        "n_files_before": len(before), "n_files_after": len(after),
        "changed": changed,
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "verdict": "unchanged" if not changed else "MODIFIED_BY_ROLLOUT",
    }, indent=1))
    if changed:
        raise RuntimeError(
            f"SkillGrad seed {seed_dir} was MODIFIED during sg_base ({len(changed)} file(s): "
            + ", ".join(changed[:6])
            + "). The base-trajectory solver has unrestricted write_file and was handed this seed as "
              "--skills-dir, so sg_train would start from a skill the rollout itself edited: Induction "
              "would not be measured. Receipt: sg_seed_integrity.json. Do NOT report this cell.")


# Upper bound on one SkillGrad stage. Generous: the longest healthy `sg_base` observed on this grid
# took ~700s wall for 20 tasks, and `sg_train` is longer, so this only fires on a real hang.
STAGE_TIMEOUT_S = int(os.environ.get("SG_STAGE_TIMEOUT_S", "5400"))


# `sg_base` on some backbones grades every task, writes summary.json with completed == total, and then hangs
# until the wedge reaper TERMs it ~40 min later (or STAGE_TIMEOUT_S, 90 min). Once the artifact says the work is done and has stayed
# done for this long, the stage is ended instead of waited out. Measured 2026-10-15: up to 133 min of one cell.
SG_BASE_GRACE_S = int(os.environ.get("SG_BASE_GRACE_S", "90"))

# `sg_train` can stop making progress mid-iteration and sit until STAGE_TIMEOUT_S (90 min), which fails the evolve.
# v4 (2026-10-19): 5 of 38 SB sg_train stages, all Sonnet; one TBA cell lost 3 of its 5 final-test evolves this way.
# Example: training finished iteration 2, its iteration-3 rollouts wrote output.xlsx by 23:26, then nothing changed
# until the 00:30 kill. A stage whose log and train/ tree have not changed for this long is ended and retried
# once from a clean train/ (0 disables). Progress is read from a few stats, never a tree walk (the shared filesystem metadata).
SG_TRAIN_STALL_S = int(os.environ.get("SG_TRAIN_STALL_S", "1800"))


def _train_progress(arm, log):
    """Cheap progress signature of a training stage: stage-log size, train/ mtime, newest iter_*/ mtime and
    its entry count. Iterations create a directory per rollout/diagnosis, so a live stage changes it."""
    mark = []
    try:
        mark.append(os.stat(log.name).st_size if isinstance(getattr(log, "name", None), str) else None)
    except OSError:
        mark.append(None)
    for train in sorted(Path(arm).glob("runs/*/train")):
        try:
            iters = sorted(train.glob("iter_*"), key=lambda p: p.stat().st_mtime)
            newest = iters[-1] if iters else None
            mark.append((str(train), train.stat().st_mtime,
                         newest and (newest.name, newest.stat().st_mtime, len(os.listdir(newest)))))
        except OSError:
            mark.append(None)
    return tuple(mark)


def _rebuild_failure_ids(model_bt) -> dict:
    """`failure_ids.json` as upstream writes it (data/split.py:identify_failures: missing assessment or cell_accuracy < 1
    is a failure), for a salvaged sg_base that was ended before it wrote the file (2026-10-21). The clean-train design
    reads the honest base grades from it; without it every such evolve failed. Tasks the ended rollout never finished
    count as failures, as upstream counts a missing assessment."""
    model_bt = Path(model_bt)
    summary = json.loads((model_bt / "summary.json").read_text())
    failure_ids, success_ids = [], []
    for row in summary.get("results", []):
        tid = str(row["id"])
        path = model_bt / tid / "assessment.json"
        acc = json.loads(path.read_text()).get("cell_accuracy", 0.0) if path.exists() else 0.0
        (failure_ids if acc < 1.0 else success_ids).append(tid)
    fid = {"failure_ids": failure_ids, "success_ids": success_ids, "summary": summary.get("aggregate"),
           "rebuilt_by_driver": "sg_base ended before writing failure_ids.json; identify_failures rule"}
    (model_bt / "failure_ids.json").write_text(json.dumps(fid, indent=1))
    return fid


def _stage_work_is_complete(arm, stage_name: str = "sg_base", since: float | None = None) -> bool:
    """Did the stage actually finish its work, regardless of whether the process returned?

    Reads the newest `summary.json` the stage wrote under `arm` and reports whether every task it
    planned was completed. `status` is deliberately NOT trusted: the hang this exists for leaves
    `status == "running"` on a run whose `completed == total`.

    STAGE-AWARE, and that is load-bearing (2026-09-16 17:55Z). `summary.json` is written by the
    BASE-trajectory stage only. When `sg_train` failed on the `references` corruption, a stage-blind
    version found the base stage's `completed 20/20`, declared the TRAINING stage salvaged, and so
    **skipped the repair-and-retry that exists for exactly that failure** -- observed on
    rq1_sg_qwen_ours round_0/evolve_1: `sg_train exited rc=1 ... work_complete_per_summary=True`
    with a 0-byte `references` file still in place and `final_skill` never produced. Generalising
    the salvage had started masking the very bug the repair fixes.

    So only `sg_base` may be salvaged from `summary.json`. A training stage's completion is proved by
    `train/final_skill/**/SKILL.md`, nothing else.
    """
    arm = Path(arm)
    if stage_name.startswith("sg_train"):
        # training produced a usable artifact, or it did not finish -- no proxy evidence allowed
        return any(arm.rglob("train/final_skill/**/SKILL.md"))
    try:
        summaries = sorted(arm.rglob("summary.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if since is not None:                      # only a summary THIS stage wrote can prove THIS stage finished
            summaries = [p for p in summaries if p.stat().st_mtime >= since]
    except Exception:
        return False
    for s in summaries[:4]:                       # newest few; older stages may also have one
        try:
            d = json.loads(s.read_text())
        except Exception:
            continue
        done, total = d.get("completed"), d.get("total")
        if isinstance(done, int) and isinstance(total, int) and total > 0 and done >= total:
            return True
    return False


def _wait_stage(proc, arm, name, log, poll_s: float = 30.0):
    """Wait for a SkillGrad stage subprocess. Returns (rc, timed_out).

    `sg_base` is ended once every task is graded (summary.json written BY THIS STAGE says completed == total) and the process
    has still not returned SG_BASE_GRACE_S later; otherwise the stage gets STAGE_TIMEOUT_S. A non-zero rc from either is handled
    by the caller's salvage path, which re-reads the same artifact. `sg_train` is ended as "stalled" once nothing
    it writes has changed for SG_TRAIN_STALL_S; the caller then retries it once."""
    started, done_since = time.time(), None
    progress, progress_at = None, time.time()
    while True:
        try:
            return proc.wait(timeout=poll_s), False
        except subprocess.TimeoutExpired:
            pass
        if name.startswith("sg_base"):
            if _stage_work_is_complete(arm, name, since=started):
                done_since = done_since or time.time()
                if time.time() - done_since >= SG_BASE_GRACE_S:
                    log.write(f"\n[driver] {name}: every task was graded {SG_BASE_GRACE_S}s ago but the process "
                              f"did not return; ending it (the work is on disk)\n")
                    proc.terminate()
                    try:
                        return proc.wait(timeout=30), False
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        return proc.wait(), False
            else:
                done_since = None
        if name.startswith("sg_train") and SG_TRAIN_STALL_S > 0:
            mark = _train_progress(arm, log)
            if mark != progress:
                progress, progress_at = mark, time.time()
            elif time.time() - progress_at >= SG_TRAIN_STALL_S:
                log.write(f"\n[driver] {name}: no progress for {SG_TRAIN_STALL_S}s (stage log, train/ and newest "
                          f"iteration unchanged); ending it\n")
                proc.kill()
                proc.wait()
                return 1, "stalled"
        if time.time() - started > STAGE_TIMEOUT_S:
            proc.kill()
            proc.wait()
            return 1, True


def _is_directory_placeholder(body: str) -> bool:
    """Is this file's content merely a DESCRIPTION of the directory it should have been?

    Deliberately narrow: it must be short AND talk about being a directory. Real evolved skill text
    is long and instructional (the SKILL.md bodies on this grid run 2-4 KB of procedure), so the
    size ceiling alone excludes it; the phrase requirement is a second, independent guard.

    The observed instance (qwen, 188B) is:
        "# Directory for L3 content\n\nThis directory contains detailed procedures, ..."
    """
    if len(body) > 600:                            # ~3x the observed case, far below real skill text
        return False
    low = body.lower()
    return any(k in low for k in ("directory for", "this directory contains",
                                  "this folder contains", "placeholder"))


def _repair_references_as_file(arm) -> list[str]:
    """Turn any `.../skills/<name>/references` REGULAR FILE back into a directory.

    WHY (measured, 2026-09-16): SkillGrad's patcher agent is told to write skill references, and
    on `qwen3-235b` it creates `skills/xlsx/references` as a ZERO-BYTE REGULAR FILE instead
    of a directory. Every later `references/` path operation then fails with

        [Errno 17] File exists: .../skills/xlsx/references      (the patcher's own write)
        [Errno 20] Not a directory: .../skills/xlsx/references  (the manifest upsert)

    so the run records `status=failed`, `final_skill` never materializes, and the cell reports
    `J=None` for every round -- which is indistinguishable from "the attack did not induce".

    Backbone-specific, not a general harness fault: of the non-empty `sg_train.log`s inspected,
    qwen hit it 1/1 while kimi (29), gpt-oss (29) and sonnet (35) hit it 0/93. So this repairs a
    victim-agent behavior rather than papering over our own bug.

    A zero-byte file carries no information, so replacing it with the directory the tree already
    expects cannot destroy evolved content.

    UPDATE 2026-09-16 14:30Z -- a zero-size test was TOO STRICT and cost a second rerun. On the next
    attempt qwen wrote a **188-byte** `references` file whose entire content is a description OF THE
    DIRECTORY it was supposed to create:

        # Directory for L3 content
        This directory contains detailed procedures, error handling, and verification steps
        for specific operations that are conditionally loaded during task execution.

    That is placeholder prose about a directory, not skill content, and the same cell's `evolve_3`
    has `references` as a proper DIR -- so the behavior is nondeterministic within one run, not a
    deliberate alternative layout. Refusing to touch it just reproduced the original failure.

    So a file is now repaired when it is empty OR is recognizably such a placeholder, and its bytes
    are PRESERVED as `references/README.md` inside the new directory rather than deleted. Anything
    else (a file with real content that is not placeholder-shaped) is still left alone and reported:
    the point is to never silently discard evolved skill text.
    """
    fixed: list[str] = []
    for p in Path(arm).rglob("skills/*/references"):
        if not p.is_file():
            continue
        try:
            size = p.stat().st_size
            body = "" if size == 0 else p.read_text(errors="replace")
            if size and not _is_directory_placeholder(body):
                fixed.append(f"LEFT non-empty file {p} ({size}B, not placeholder-shaped)")
                continue
            p.unlink()
            p.mkdir(parents=True, exist_ok=True)
            if body.strip():                       # keep the bytes; never destroy what it wrote
                (p / "README.md").write_text(body)
                fixed.append(f"repaired {p} (placeholder {size}B preserved as references/README.md)")
            else:
                fixed.append(f"repaired {p}")
        except OSError as e:                       # never let a repair abort the run
            fixed.append(f"FAILED to repair {p}: {e}")
    return fixed


def _sg_env(endpoint: str, model: str) -> dict:
    """Env that routes SkillGrad's agents-SDK client at our proxy (chat path)."""
    ep = endpoint if endpoint.endswith("/") else endpoint + "/"
    return {**os.environ, "PYTHONPATH": str(SG), "AWS_REGION": "us-west-2",
            "AZURE_OPENAI_ENDPOINT": ep, "AZURE_OPENAI_API_KEY": _GATEWAY_KEY,
            "OPENAI_API_KEY": _GATEWAY_KEY,
            "TARGET_DEPLOYMENT": model, "TARGET_BACKEND": "openai_chat",
            "SE_PIPELINE_SOFFICE_LOCK": "/tmp/skillgrad_soffice.lock"}


class SkillGradRunner(Trace2SkillRunner):
    name = "SkillGrad"
    evolution_mode = "textual-gradient consolidation (SkillGrad)"
    mechanism_key = "skillgrad_sb"          # poison ids FORCED into the failure pool (see evolve)

    def __init__(self, model="claude-sonnet-4.6", max_turns=30, workers=4, seed=0,
                 endpoint="http://localhost:8000/v1", skill_mode="evolved",
                 evolver="skillgrad", n_train=24, batch_size=4, n_iterations=4):
        # skill_mode seed/none reuse the parent's solver-side behavior verbatim.
        super().__init__(model=model, max_turns=max_turns, workers=workers, seed=seed,
                         endpoint=endpoint, skill_mode=skill_mode, evolver="error")
        self.n_train, self.batch_size, self.n_iterations = n_train, batch_size, n_iterations
        self._endpoint = endpoint

    # inject() and solve() are inherited from Trace2SkillRunner unchanged: same
    # poison-task materialization, same fixed held-out solver. Only evolve differs.

    def _evolve_native(self, workdir: str) -> list[str]:
        """SkillGrad consolidation over the poison dataset -> staged xlsx skill."""
        wd = Path(workdir)
        _assert_no_stdlib_shadow()          # see the function's docstring: this WILL happen again
        pd = neutral_names.dataset_dir(wd)
        arm = wd / "sg"
        arm.mkdir(exist_ok=True)
        # PRIVATE SEED. Upstream writes the evolving skill back into --skills-dir, so pointing every
        # cell at the shared `seeds/` let one run poison the next. Stage a copy per run and pass its
        # ABSOLUTE path to both stages.
        from attack.victims.clean_startpoint import stage as stage_clean_startpoint
        seed_dir = stage_clean_startpoint(self, wd / "sg_seed") or _stage_private_seed(wd)
        if neutral_names.enabled():
            # The staged seed calls recalc.py by this attempt's real path (clean_startpoint.clone); SkillGrad's
            # stages see /mnt/ws/... (neutral_names.stage_path), so repoint them before the base run reads the seed.
            from attack.victims.trace2skill_runner import _absolutise_recalc_refs
            _absolutise_recalc_refs(Path(seed_dir))
        # Pristine check on the seed THIS run uses. The private copy comes from the committed
        # upstream tree, so a poisoned shared working tree no longer blocks (or leaks into) the run;
        # a poisoned COMMITTED seed, or a resumed private copy the run itself poisoned, still raises.
        _assert_seed_is_pristine(seed_dir)
        manifest = self._materialize_poison_split(pd, arm)
        ms, hs, ts = "0", "42", "0"
        env = _sg_env(self._endpoint, self.model)
        # POISON_FS_GUARD=1: the SkillGrad solver's project_root is its cwd, and it writes scratch files
        sg_cwd = str(SG)
        normalized = manifest["normalized"]

        def stage(name, args, _repair_retry=True):
            log = open(wd / f"{name}.log", "w")
            try:
                from attack.evaluation import runlog
                runlog.set_progress(str(wd.parent), substage=name)
            except Exception:
                pass
            # TIMEOUT + COMPLETION SALVAGE, not a bare `subprocess.call`.
            #
            # `runners.stream_runner base-trajectories` can finish all of its work -- every task
            # graded, every result persisted, `summary.json` written -- and then hang instead of
            # returning. Observed on rq1_sg_qwen_tg: `summary.json` showed
            # `completed: 20/20` with `status: "running"`, while the same code path on
            # sg_sonnet_tg / sg_qwen_ours / sg_kimi_ours reached `status: "completed"`. A hung
            # child sits at zero CPU with no established socket, i.e. it is indistinguishable from
            # a wedge, so the reaper TERMs it at the 600s mark -- 18 times on that one cell. The
            # driver then never reached the post-sg_base step, so `failure_ids.json` was never
            # written and the measurement had nothing to score (`I = 0/5`, `J = None`).
            #
            # So: wait with a bound, and on timeout ask the ARTIFACT whether the work is done
            # rather than trusting process exit. If `summary.json` says every task completed, the
            # stage succeeded and we continue; only a genuine partial run raises.
            #
            # 2026-09-16 16:40Z -- the salvage was in the WRONG BRANCH and therefore never ran in
            # production. STAGE_TIMEOUT_S is 5400s but the wedge reaper TERMs a frozen child at
            # ~1800s+600s, so THE REAPER ALWAYS WINS THE RACE: `proc.wait()` returns a non-zero rc
            # normally, `TimeoutExpired` is never raised, and the artifact was never consulted.
            # Measured on rq1_sg_qwen_ours round_0: all five evolves wrote `summary.json` with
            # `completed 20/20`, all five were reaped at the base stage, zero train logs opened, and
            # every round reported `J=None` -- which reads as "qwen never induced the poison".
            # Salvage now runs on ANY non-zero exit, since a reaper TERM and a self-hang are the
            # same situation from here: the work is on disk and the process did not return cleanly.
            proc = subprocess.Popen([sys.executable, "-m", *args], cwd=sg_cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            rc, timed_out = _wait_stage(proc, arm, name, log)
            if rc != 0:
                salvaged = _stage_work_is_complete(arm, name)
                why = (f"stalled (no progress for {SG_TRAIN_STALL_S}s)" if timed_out == "stalled"
                       else f"exceeded {STAGE_TIMEOUT_S}s" if timed_out
                       else f"exited rc={rc} (killed externally? reaper TERM looks like this)")
                log.write(f"\n[driver] {name} {why}; work_complete_per_summary={salvaged}\n")
                if salvaged:
                    rc = 0
            log.close()
            if rc != 0 and timed_out == "stalled" and _repair_retry:
                # Retry once from a clean train/: the stalled attempt is kept beside it for inspection.
                moved = []
                for train in sorted(arm.glob("runs/*/train")):
                    aside = train.with_name(f"train_stalled_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
                    train.rename(aside)
                    moved.append(str(aside))
                with open(wd / f"{name}.log", "a") as lg:
                    lg.write(f"[driver] moved the stalled attempt to {moved}; retrying {name} once\n")
                return stage(f"{name}_retry", args, _repair_retry=False)
            if rc != 0:
                # REPAIR-AND-RETRY MUST LIVE HERE, not after the call.
                #
                # `references`-as-a-file kills training from INSIDE the stage
                # (`pipeline/momentum.py:57`, `refs_dir.iterdir()` -> NotADirectoryError), so the
                # process exits non-zero and this raise fires. A repair placed after `stage(...)`
                # is therefore unreachable on exactly the failure it targets -- verified live on
                # rq1_sg_qwen_ours/evolve_4, which hit `Not a directory` twice while the driver
                # never logged a repair.
                if _repair_retry:
                    repairs = _repair_references_as_file(arm)
                    if any(r.startswith("repaired") for r in repairs):
                        with open(wd / f"{name}.log", "a") as lg:
                            lg.write("\n[driver] references-as-file corruption detected; "
                                     + "; ".join(repairs)
                                     + f"\n[driver] retrying {name} once\n")
                        return stage(f"{name}_retry", args, _repair_retry=False)
                raise RuntimeError(f"SkillGrad stage failed: {name} (see {wd}/{name}.log)")

        # Fingerprint BEFORE the rollout: the comparison after sg_base is what makes a within-cell
        # self-install provable rather than assumed.
        _seed_before = _seed_fingerprint(seed_dir)
        stage("sg_base", ["runners.stream_runner", "base-trajectories",
            "--model", self.model, "--master-seed", ms, "--heldout-seed", hs,
            "--skills-dir", str(seed_dir), "--data-dir", normalized, "--results-root", str(arm),
            "--max-turns", str(self.max_turns),
            "--executor-concurrency", str(self.workers), "--grader-concurrency", "1"])
        # FORCE the poison ids into the failure pool training distils from. SkillGrad's
        # diagnoser samples `failure_ids.json` in the model-keyed base-traj dir; poison
        # tasks are VALID SB tasks so they pass the base solver and would otherwise be 0
        # failures (training aborts). This mirrors v0's force_failures=True: the attack
        # forces the poison trajectories to be the "failures" the evolver learns from.
        # WITHIN-CELL SELF-INSTALL CHECK. Between the two stages, before sg_train reads the seed.
        _assert_seed_unchanged_by_rollout(seed_dir, _seed_before, wd)
        model_bt = Path(f"{arm}/base_trajectories/master_{ms}_heldout_{hs}/{self.model}")
        forced = manifest.get("forced_failure_ids")
        poison_ids = manifest.get("poison_ids") or []
        if "original_clean_ids" in manifest:
            # Honest clean examples retain their actual base-rollout grades.
            if not (model_bt / "failure_ids.json").exists():
                _rebuild_failure_ids(model_bt)
            fid = json.load(open(model_bt / "failure_ids.json"))
            training_ids = set(manifest["original_clean_ids"]) | set(poison_ids)
            failed = ({str(i) for i in fid["failure_ids"]} & training_ids) | set(poison_ids)
            fid = {**fid, "failure_ids": sorted(failed),
                   "success_ids": [str(i) for i in fid.get("success_ids", [])
                                   if str(i) in training_ids and str(i) not in failed]}
        elif forced and Path(forced).exists():
            fid = json.load(open(forced))
        else:
            fid = {"failure_ids": poison_ids, "success_ids": []}
        (model_bt / "failure_ids.json").write_text(json.dumps(fid, indent=1))
        run_id = f"skillgrad_{self.model}_{neutral_names.run_label('poison')}"   # ..._train when neutral
        stage("sg_train", ["pipeline.training",
            "--data-dir", manifest["data_dir"], "--skills-dir", str(seed_dir), "--results-root", str(arm),
            "--base-trajectories-dir", f"{arm}/base_trajectories/master_{ms}_heldout_{hs}/{self.model}",
            "--method", "skillgrad", "--model", self.model, "--config-tag", neutral_names.run_label("poison"),
            "--master-seed", ms, "--heldout-seed", hs, "--training-seed", ts,
            "--n-train", str(min(self.n_train, manifest["n_evolution"])),
            "--batch-size", str(self.batch_size), "--batch-schedule", "fixed-updates",
            "--n-iterations", str(self.n_iterations), "--max-turns", str(self.max_turns),
            "--concurrency", str(self.workers)])

        final_skill = Path(f"{arm}/runs/{run_id}/train/final_skill")
        sk = wd / "skills"
        from attack.victims.trace2skill_runner import V as T2S
        if not (sk / "xlsx-35B" / "SKILL.md").exists():
            shutil.copytree(T2S / "spreadsheet_agent/skills", sk, dirs_exist_ok=True)
            from attack.victims.trace2skill_runner import _harden_recalc
            _harden_recalc(sk)
        sg_xlsx = final_skill / "xlsx"
        if (sg_xlsx / "SKILL.md").exists():
            shutil.rmtree(sk / "xlsx", ignore_errors=True)
            shutil.copytree(sg_xlsx, sk / "xlsx")
            # The copytree above replaces the HARDENED xlsx/ WHOLESALE with SkillGrad's evolved
            # skill, which ships only SKILL.md (+references/) -- verified on a real run:
            # runs/_oldspec_prepaired_backup_*/agentopt_ours__skillgrad__*/final_test/evolve_1/
            # skills/xlsx contains `SKILL.md references` and NO recalc.py. So the tool the skill
            # tells the agent to run did not exist: recalculation was impossible, openpyxl formulas
            # kept no cached values, and SpreadsheetBench's comparator read None => graded wrong.
            # Restore the executable next to the evolved SKILL.md, then re-harden (macro seed +
            # absolute path) so the agent can actually run it and never has to search for it.
            from attack.victims.trace2skill_runner import _harden_recalc as _hr
            for aux in ("recalc.py", "LICENSE.txt"):
                src_aux = T2S / "spreadsheet_agent/skills/xlsx" / aux
                if src_aux.exists() and not (sk / "xlsx" / aux).exists():
                    shutil.copy(src_aux, sk / "xlsx" / aux)
            _hr(sk)
            # ...and re-assert the OBLIGATION, not just the tool. `_harden_recalc` only makes
            # recalc.py hang-proof; the "recalculate before finishing" contract lives in SKILL.md,
            # which the copytree above replaced wholesale. SkillOpt and A-Evolve both call this on
            # their overwrite paths (37bdb28); SkillGrad was the one runner that did not, so it was
            # relying on the evolved skill happening to keep its own recalc guidance. On the cells
            # inspected it did (4 mentions), which is exactly why this stayed invisible -- the
            # harness invariant was being satisfied by luck, per framework and per round.
            from attack.victims.trace2skill_runner import _ensure_recalc_contract as _erc
            _erc(sk)
        else:
            raise RuntimeError(f"SkillGrad produced no final_skill at {sg_xlsx}")
        self._evolved_skill_dir = str(sk)
        return [md.read_text(errors="replace") for md in sk.rglob("SKILL.md")]

    def _materialize_poison_split(self, poison_dataset: Path, out_root: Path) -> dict:
        """Turn the parent's poison_dataset/ into a SkillGrad split whose evolution
        pool is exactly those poison tasks (train-only poison; val/test untouched)."""
        sys.path.insert(0, str(SG))
        from attack.victims.skillgrad_adapter import materialize_aligned
        bridge = out_root / neutral_names.run_label("poison_bridge")
        (bridge / "poison").mkdir(parents=True, exist_ok=True)
        entries = json.load(open(poison_dataset / "dataset.json"))
        items = []
        for e in entries:
            pid = e["id"]
            sp = poison_dataset / "spreadsheet" / pid
            items.append({"id": pid, "instruction": e.get("instruction", ""),
                          "answer_position": e.get("answer_position", ""),
                          "spreadsheet_path": str(sp)})
        json.dump(items, open(bridge / "poison" / "items.json", "w"), indent=1)
        json.dump({"src_ids": []}, open(bridge / "poison" / "src_ids.json", "w"))
        mix_path = poison_dataset.parent / "training_pool.json"
        if not mix_path.exists():
            return materialize_aligned(str(out_root), "poison", n_clean=0,
                                       poison_dir=str(bridge))
        from attack.datasets.benchmark import Benchmark
        benchmark = Benchmark("spreadsheetbench")
        mix = json.loads(mix_path.read_text())
        manifest = materialize_aligned(str(out_root), "poison", n_clean=0,
            poison_dir=str(bridge), data_dir=str(benchmark.data_root), force_failures=False,
            frozen_splits=benchmark.load_splits(split_names=("train", "dev")))
        clean = set(mix["clean_ids"])
        manifest.update(original_clean_ids=mix["clean_ids"], n_clean=len(clean),
                        n_poison=mix["n_poison"],
                        aligned_split=None, split_protocol="rq1_train_dev",
                        frozen_manifest=str(benchmark.paths.split_manifest),
                        beta_window=mix["poison_ratio"],
                        poison_ids=[i for i in manifest["poison_ids"] if i not in clean],
                        evolution_policy="frozen train inputs; honest clean grades plus forced poison failures")
        (out_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest
