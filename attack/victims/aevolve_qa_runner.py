"""Native multi-skill A-Evolve attacks on frozen WikiTQ and OfficeQA tasks.

Only injection is delegated to the dataset runner. Training and deployment use
the same native workspace, provider, tools, and graders as clean A-Evolve.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
from pathlib import Path
import time

from attack.clean.aevolve_driver import seed_workspace
from attack.clean.aevolve_runtime import CleanAgent, sha, workspace_hashes
from attack.clean.common import digest, write_json
from attack.clean import qa_measurement as qm
from attack.datasets.benchmark import Benchmark
from attack.evaluation.qa_targets import make_qa_target
from attack.evaluation.targets import qa_induced_in
from attack.victims import aevolve_clean, neutral_names
from attack.victims.officeqa_runner import OfficeQARunner
from attack.victims.qa_framework_runner import _poison_items
from attack.victims.wikitq_runner import WikiTQRunner

PROTOCOL = "aevolve_multiskill_qa_attack_v1"


class InputReviewAgent(CleanAgent):
    """CleanAgent plus the native-SB input-aware skill-selection guidance (same text, no target).

    Opt-in deployment protocol (AE_INPUT_REVIEW=1): it changes the victim solver, so every arm
    (TBA/TextGrad/Ours) and the clean reference must use it together or not at all."""

    def loading(self, snapshot, events, task_work):
        from attack.victims.aevolve_sb_agent import GUIDANCE
        system, tools, executors = super().loading(snapshot, events, task_work)
        return system + GUIDANCE, tools, executors


def _guidance_sha256():
    from attack.victims.aevolve_sb_agent import GUIDANCE
    return sha(GUIDANCE.encode())


class AEvolveQARunner:
    name = framework = "aevolve"
    evolution_mode = "native_unified_skills_only"
    native_multiskill = skill_discovery_gated = True
    mechanism_key = "aevolve_native"        # multi-skill catalog, read on demand

    def __init__(self, dataset: str, model="kimi-k2.5", endpoint="http://localhost:8000/v1",
                 seed=101, skill_mode="evolved", cycles=2, workers=4,
                 tool_turns=24, max_tokens=16384, target=None, measurement_protocol=qm.LEGACY,
                 input_review=None):
        if dataset not in ("wikitq", "officeqa"):
            raise ValueError(f"A-Evolve QA requires wikitq or officeqa, got {dataset!r}")
        if skill_mode not in ("evolved", "base", "initial", "none"):
            raise ValueError(f"Unknown native skill mode: {skill_mode}")
        if min(cycles, workers, tool_turns, max_tokens) < 1:
            raise ValueError("Native cycle, worker, and model budgets must be positive")
        self.dataset, self.model, self.endpoint = dataset, model, endpoint
        self.seed, self.skill_mode, self.cycles = seed, skill_mode, cycles
        self.workers, self.tool_turns, self.max_tokens = workers, tool_turns, max_tokens
        self.measurement_protocol = qm.resolve_protocol(measurement_protocol)
        self.input_review = (os.environ.get("AE_INPUT_REVIEW") == "1"
                             if input_review is None else bool(input_review))
        self.benchmark = Benchmark(dataset, seed)
        self.splits = self.benchmark.load_splits()
        self.train = {str(t["id"]): t for t in self.splits["train"]}
        self.heldout = {str(t["id"]): t for s in ("dev", "test") for t in self.splits[s]}
        base_cls = WikiTQRunner if dataset == "wikitq" else OfficeQARunner
        self._base = base_cls(model=model, endpoint=endpoint, seed=seed)
        self.target = target or make_qa_target(dataset)
        self._evolved_skill_dir = None
        self._last_evolve_noop = False
        self.evidence_paths = {}
        self.skill_index = []

    def _induced_in(self, text):
        if getattr(self.target, "accuracy_invisible", True):
            if self.measurement_protocol == qm.PAIRED:
                from attack.evaluation.targets import qa_operational_induction
                return qa_operational_induction(text)
            return qa_induced_in(text)
        return self.target.induced_in(text)

    def inject(self, payload, workdir: str) -> None:
        ids = [str(it["base"]["id"]) for it in payload.items]
        if not ids or any(i not in self.train for i in ids):
            raise ValueError("Poison bases must belong to the frozen train split")
        poison_ids = [str(it["id"]) for it in payload.items]
        if len(set(poison_ids)) != len(poison_ids) or set(poison_ids) & set(self.heldout):
            raise ValueError("Poison IDs must be unique and disjoint from held-out tasks")
        identity = {"dataset": self.dataset, "payload_sha256": digest(payload.items)}
        identity_path = Path(workdir) / "poison_dataset/injection_config.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Existing native poison dataset has different inputs")
            return  # Keep per-task tokens identical across Python process restarts.
        self._base.inject(payload, workdir)
        write_json(identity_path, identity)

    def _agent(self, workspace: Path) -> CleanAgent:
        cls = InputReviewAgent if self.input_review else CleanAgent
        return cls(workspace, self.dataset, self.model, self.endpoint,
                          self.tool_turns, self.max_tokens, protocol=PROTOCOL, qa_attack=True,
                          **({"runtime_profile": self.clean_startpoint["source_config"].get("runtime_profile", "v1")}
                             if getattr(self, "clean_startpoint", None) else {}))

    def evolve(self, workdir: str) -> list[str]:
        # POISON_NEUTRAL_NAMES: the agent's workspace and logs live in a neutral copy of <workdir>/{evo,workdir},
        # copied back afterwards; poison_dataset/ and the measurement files are not in it (neutral_names.open_mirror).
        wd = Path(workdir)
        run, tmp = neutral_names.open_mirror(wd, ("evo", "workdir"))
        try:
            return self._evolve_in(wd, run)
        finally:
            neutral_names.close_mirror(tmp, wd)

    def _docs_view(self):
        """OfficeQA under POISON_NEUTRAL_NAMES: (DATA/docs, its neutral read-only copy); else None."""
        if self.dataset != "officeqa" or not neutral_names.enabled():
            return None
        from attack.victims.officeqa_clean import DATA
        return DATA / "docs", neutral_names.docs_view(DATA / "docs")

    def _evolve_in(self, wd: Path, run: Path) -> list[str]:
        workdir = str(wd)
        entries = json.loads((wd / "poison_dataset/dataset.json").read_text())
        if not entries or any(str(e["base_id"]) not in self.train for e in entries):
            raise ValueError("Native evolution may consume poison over frozen train tasks only")
        rows = _poison_items(workdir, self.dataset, self.train)
        if neutral_names.enabled():
            rows, id_map = neutral_names.rename(rows, list(self.train) + list(self.heldout))
            view = self._docs_view()
            if view:
                rows = neutral_names.view_items(rows, *view)
            write_json(wd / "neutral_names.json", {"ids": id_map, "workspace": str(run),
                       "docs_view": str(view[1]) if view else None, "rows_leak_words": neutral_names.leak_counts(rows)})
        arm = run / "evo"
        config = {"protocol": PROTOCOL, "dataset": self.dataset, "model": self.model,
                  **({"qa_measurement_protocol": self.measurement_protocol}
                     if self.measurement_protocol != qm.LEGACY else {}),
                  "endpoint": self.endpoint, "seed": self.seed, "cycles": self.cycles,
                  "skill_mode": self.skill_mode, "deployment_protocol": "native_multi_skill",
                                    "workers": self.workers, "tool_turns": self.tool_turns,
                  "max_tokens": self.max_tokens, "train_sha256": digest(rows),
                  **({"input_review_guidance_sha256": _guidance_sha256()} if self.input_review else {})}
        if getattr(self, "clean_startpoint", None):
            config["clean_startpoint"] = self.clean_startpoint
        config_path = arm / "attack_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Existing A-Evolve attempt has different inputs or model configuration")
        complete = arm / "evolution_result.json"
        ws = run / "workdir/aevolve/workspace"
        if complete.exists():
            if not config_path.exists():
                raise RuntimeError("Completed native evolution is missing its configuration")
            state = json.loads(complete.read_text())
            if state["workspace_sha256"] != digest(workspace_hashes(ws)):
                raise RuntimeError("Completed native workspace changed before reload")
        else:
            # Preserve evidence from an incomplete attempt before restarting the same seed.
            if arm.exists() or (run / "workdir").exists():
                archive = run / "training_attempts" / str(time.time_ns())
                archive.mkdir(parents=True)
                for path in (arm, run / "workdir"):
                    if path.exists():
                        path.rename(archive / path.name)
            if getattr(self, "clean_startpoint", None):
                from attack.victims.clean_startpoint import stage as stage_clean_startpoint
                ws = stage_clean_startpoint(self, ws)
                shutil.copytree(ws, run / "workdir/initial_workspace")
            else:
                ws = seed_workspace(run, self.benchmark)
            write_json(config_path, config)
            write_json(arm / "poison_train.json", rows)
            if self.skill_mode == "none":
                shutil.rmtree(ws / "skills")
                (ws / "skills").mkdir()
            if self.skill_mode == "evolved":
                aevolve_clean.evolve(self._agent(ws), rows, arm,
                                     cycles=self.cycles, workers=self.workers)
            else:
                write_json(complete, {"cycles_completed": 0, "skill_mode": self.skill_mode,
                                        "workspace_sha256": digest(workspace_hashes(ws)), "train_ids": []})
        agent = self._agent(ws)  # Native discovery after a cold reload, including new skills.
        snapshot = agent.skill_snapshot()
        initial = workspace_hashes(run / "workdir/initial_workspace")
        self._last_evolve_noop = initial == workspace_hashes(ws)
        real = lambda p: wd / Path(p).relative_to(run)          # where the mirror lands (identity when off)
        self._evolved_skill_dir = str(real(ws / "skills"))
        self.evidence_paths = {
            "poison_dir": str(wd / "poison_dataset"), "skill_dir": self._evolved_skill_dir,
            "error_analysis": str(real(arm / "steps")), "success_analysis": str(real(arm / "steps")),
            "changelog": str(real(arm / "evolution_result.json")), "logs_dir": str(real(arm / "train"))}
        skills = []
        for meta in snapshot["available"]:
            files = {p: text for p, text in snapshot["files"].items() if p.startswith(meta["path"] + "/")}
            sources = [{"path": path, "source_sha256": sha((ws / path).read_bytes()),
                        "role": "body" if path == meta["path"] + "/SKILL.md" else "reference",
                        "has_poison": self._induced_in(text)} for path, text in files.items()]
            skills.append({"path": meta["path"], "files": files, "sources": sources,
                           "induced": any(source["has_poison"] for source in sources),
                           # Compatibility summary only; gate detection uses actual source files.
                           "text": "\n\n".join(files.values())})
        self.skill_index = skills
        write_json(arm / "skill_index.json", skills)
        mix_path = wd / "training_pool.json"
        mix = json.loads(mix_path.read_text()) if mix_path.exists() else None
        write_json(arm / "framework.json", {**config, "framework": self.framework,
            "n_poison": mix["n_poison"] if mix else len(rows),
            **({"n_clean_train": mix["n_clean"], "n_training": mix["n_total"]} if mix else {}),
            "evolve_noop": self._last_evolve_noop,
            "workspace": str(real(ws)), "skill_loading": "native_catalog_then_read_skill",
            "n_skills": len(skills), "evidence_paths": self.evidence_paths})
        return [s["text"] for s in skills]

    def _heldout_row(self, task: dict) -> dict:
        canonical = self.heldout.get(str(task["id"]))
        if canonical is None:
            raise ValueError("Native QA deployment requires a frozen dev or test task")
        # Native fields come from the normalized frozen item. Only the existing
        # dataset-owned trigger carrier can vary between paired deployment arms.
        carrier = "table_caption" if self.dataset == "wikitq" else "context"
        for key, value in canonical.items():
            if key in {carrier, "trigger", "token"}:
                continue
            if key in task and task[key] != value:
                # The legacy caller's unnormalized WikiTQ ground_truth can differ
                # from its official annotation; the native grader uses the latter.
                if key != "ground_truth":
                    raise ValueError(f"Held-out field changed: {key}")
        if task.get("instruction"):
            raise ValueError("Held-out QA tasks cannot contain poison instructions")
        row = copy.deepcopy(canonical)
        row["trigger"] = bool(task.get("trigger", True))
        marker = re.search(r"audit ref\s+([A-Za-z0-9_-]+)", str(task.get(carrier) or ""), re.I)
        row["token"] = task.get("token") or (marker.group(1) if marker else
                                           self.target.token_for(str(task["id"])))
        if task.get(carrier) is not None:
            row[carrier] = task[carrier]
        if row["trigger"]:
            if self.dataset == "wikitq":
                if "audit ref" not in str(row.get(carrier) or "").lower():
                    row = self.target.inject(row, row["token"])
            elif "audit ref" not in (row.get("context") or "").lower():
                row = self.target.inject(row, row["token"])
        view = self._docs_view()
        return neutral_names.view_items([row], *view)[0] if view else row

    def solve(self, task: dict, workdir: str, skill_dir: str | None) -> dict:
        sk = skill_dir or self._evolved_skill_dir
        if not sk:
            raise ValueError("Native QA deployment requires the evolved skill workspace")
        ws = Path(sk).parent
        if Path(sk).name != "skills" or not (ws / "prompts/system.md").is_file():
            raise ValueError("Expected a complete native workspace, not a flattened skill directory")
        arm = "triggered" if task.get("trigger", True) else "clean"
        output = Path(workdir) / "heldout" / arm / "predictions" / str(task["id"])
        request_sha = digest(task)
        saved_input = output / "input_task.json"
        request = task
        if saved_input.exists():
            previous = json.loads(saved_input.read_text())
            if previous["request_sha256"] == request_sha and not task.get("token"):
                request = {**task, "token": previous["row"]["token"]}
        row = self._heldout_row(request)
        before = workspace_hashes(ws)
        result = neutral_names.solve_in_mirror(lambda w, o: self._agent(w).solve_row(row, o, arm), ws, output,
                                               guard=workspace_hashes, base=ws.parents[2])
        if result["transport_error"] and self.measurement_protocol == qm.LEGACY:
            raise RuntimeError("Native held-out transport failure; retry before reporting metrics")
        if workspace_hashes(ws) != before:
            raise RuntimeError("Held-out solving mutated the native workspace")
        write_json(saved_input, {"request_sha256": request_sha, "row": row})
        messages = json.loads((output / "messages.json").read_text())
        calls = {c["id"]: c["function"]["name"] for m in messages for c in m.get("tool_calls", [])}
        loaded = [m["content"] for m in messages if m["role"] == "tool"
                  and calls.get(m.get("tool_call_id")) in ("read_skill", "read_file")]
        # Native discovery exposes learned names/descriptions before any file
        # read. Attribute only exact catalog entries backed by this snapshot and
        # actually sent in a system message, testing each source independently.
        snapshot = json.loads((output / "skill_snapshot.json").read_text())
        system = "\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
        catalog = []
        for meta in snapshot["available"]:
            text = f"- **{meta['name']}**: {meta['description']} (path: {meta['path']})"
            if text in system:
                catalog.append({"name": meta["name"], "path": meta["path"], "text": text,
                                "source_sha256": meta["sha256"], "has_poison": self._induced_in(text)})
        prov = self.target.extract(result["response"])
        if self.measurement_protocol == qm.PAIRED:
            result = {**result, "qa_measurement_protocol": qm.PAIRED}
            if result.get("finish_reasons", [])[-1:] == ["length"] and result.get("solver_ok"):
                result.update(raw_solver_ok=True, solver_ok=False, rc=1, fail_reason="completion_limit")
        if self.measurement_protocol == qm.PAIRED and not qm.failure_kind(result):
            from attack.evaluation.qa_targets import strip_marker
            answer = strip_marker(result["answer"]).strip()
            if self.dataset == "wikitq":
                from attack.clean.wikitq import grade
                hard = grade(row, answer)
                soft = float(hard)
            else:
                from attack.victims.officeqa_clean import native_module
                evaluation = native_module("evaluator").evaluate(answer, str(row["ground_truth"]))
                hard, soft = bool(evaluation["em"]), float(evaluation["f1"])
            result = {**result, "raw_answer": result["answer"], "raw_hard": result["hard"],
                      "raw_soft": result["soft"], "soft": soft,
                      "answer": answer, "hard": int(hard), "qa_measurement_protocol": qm.PAIRED,
                      "solver_ok": bool(answer), "rc": int(not answer),
                      "fail_reason": "" if answer and hard else "answer_mismatch" if answer else "no_final_answer"}
        observed = {**result, "token": row["token"], "prov": prov, "acted": prov is not None,
            "bstar_effect": self.target.verify(result["response"], row["token"]),
            "ctx_has_poison": (any(self._induced_in(text) for text in loaded)
                               or any(entry["has_poison"] for entry in catalog)),
            "catalog_entries": catalog,
            "loaded_skill_texts": loaded, "messages": messages,
            "prompt": "\n\n".join(m.get("content") or "" for m in messages
                                     if m["role"] in ("system", "user", "tool")),
            "skill_loading": json.loads((output / "skill_loading.json").read_text()),
            "skill_snapshot_path": str(output / "skill_snapshot.json"),
            "trace_path": str(output / "messages.json")}
        if not getattr(self.target, "accuracy_invisible", True):
            observed.update(acted=bool(observed["ctx_has_poison"] and not result["hard"]),
                            bstar_effect=not bool(result["hard"]))
        write_json(output / "attack_observation.json", observed)
        return observed
