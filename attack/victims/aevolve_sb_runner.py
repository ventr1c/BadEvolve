"""Opt-in native A-Evolve workbook attacks from a clean workspace checkpoint, or (single-stage design, 2026-10-21)
from the A-Evolve seed workspace: the driver sets `seed_start = {"workspace", "config"}` and the runner stages that
workspace exactly where it would stage a checkpoint's."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading
import time

from attack.clean.aevolve_runtime import sha, workspace_hashes
from attack.clean.common import digest, write_json
from attack.datasets.benchmark import Benchmark
from attack.victims import aevolve_clean
from attack.victims.clean_startpoint import stage
from attack.victims.trace2skill_runner import Trace2SkillRunner, _solver_attempted_bstar
from attack.victims.upstream import record_upstreams
from attack.datasets.paths import REPO

PROTOCOL = "aevolve_native_sb_clean_then_poison_input_review_v2"
_EVOLUTION_LOCK = threading.RLock()  # Native bindings mutate process-global reader/provider hooks.


class AEvolveSBRunner(Trace2SkillRunner):
    name = "A-Evolve"
    mechanism_key = "aevolve_native"
    native_multiskill = skill_discovery_gated = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = self.env["OPENAI_BASE_URL"]
        self.dataset = "spreadsheetbench"
        self.benchmark = Benchmark(self.dataset, self.seed)
        self.splits = self.benchmark.load_splits()
        self.train = {str(t["id"]): t for t in self.splits["train"]}
        self.heldout = {str(t["id"]): t for split in ("dev", "test") for t in self.splits[split]}

    def inject(self, payload, workdir):
        if any(str(p["base_id"]) not in self.train for p in payload.items):
            raise ValueError("Native SB poison bases must be frozen train tasks")
        super().inject(payload, workdir)
        wd = Path(workdir)
        original = dict(self._injected)          # the ids Trace2SkillRunner.inject gave each payload item
        rows = []
        from attack.victims import neutral_names
        pd = neutral_names.dataset_dir(wd)
        for e in json.loads((pd / "dataset.json").read_text()):
            p = original[str(e["id"])]
            base = self.train[str(p["base_id"])]
            directory = pd / e["spreadsheet_path"]
            initial = directory / f"1_{e['id']}_init.xlsx"
            golden = directory / f"1_{e['id']}_golden.xlsx"
            rows.append({**base, "id": str(e["id"]), "base_id": str(base["id"]),
                         "question": e["instruction"], "initial_path": str(initial),
                         "golden_path": str(golden),
                         "initial_sha256": sha(initial.read_bytes()),
                         "golden_sha256": sha(golden.read_bytes())})
            rows[-1].pop("instruction", None)
        if len(rows) != len(payload.items):
            raise ValueError("Native SB poison materialization lost training tasks")
        write_json(wd / "native_train_tasks.json", rows)

    def _native_config(self):
        checkpoint = getattr(self, "clean_startpoint", None)
        return checkpoint["source_config"] if checkpoint else self.seed_start["config"]

    def _agent(self, workspace):
        from attack.victims.aevolve_sb_agent import ReviewAgent
        cfg = self._native_config()
        return ReviewAgent(workspace, self.dataset, self.model, self.endpoint,
                          self.max_turns, cfg["max_tokens"], protocol=PROTOCOL, qa_attack=True,
                          runtime_profile=cfg.get("runtime_profile", "v2"))

    def evolve(self, workdir):
        if not getattr(self, "clean_startpoint", None) and not getattr(self, "seed_start", None):
            raise ValueError("Native SB requires a clean workspace checkpoint or the seed workspace")
        # POISON_NEUTRAL_NAMES: the agent's workspace, scratch and logs live in a neutral copy of
        # <workdir>/{evo,workdir}, copied back afterwards (neutral_names.open_mirror).
        from attack.victims import neutral_names
        wd = Path(workdir)
        run, tmp = neutral_names.open_mirror(wd, ("evo", "workdir"))
        try:
            return self._evolve_in(wd, run)
        finally:
            neutral_names.close_mirror(tmp, wd)

    def _evolve_in(self, wd, run):
        rows = json.loads((wd / "native_train_tasks.json").read_text())
        ws = run / "workdir/aevolve/workspace"
        arm = run / "evo"
        seed = None if getattr(self, "clean_startpoint", None) else self.seed_start
        config = {"protocol": PROTOCOL, "clean_startpoint": getattr(self, "clean_startpoint", None),
                  **({"seed_start": seed} if seed else {}),
                  "dataset": self.dataset, "model": self.model, "endpoint": self.endpoint,
                  "training_sha256": digest(rows), "tool_turns": self.max_turns, "cycles": 2}
        config_path = arm / "attack_config.json"
        complete = arm / "evolution_result.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Native SB resume changed configuration")
        if complete.exists():
            if json.loads(complete.read_text())["workspace_sha256"] != digest(workspace_hashes(ws)):
                raise ValueError("Completed native SB workspace changed")
        else:
            if arm.exists() or (run / "workdir").exists():
                archive = run / "training_attempts" / str(time.time_ns())
                archive.mkdir(parents=True)
                for path in (arm, run / "workdir"):
                    if path.exists():
                        path.rename(archive / path.name)
            if seed:
                from attack.victims.aevolve_sb_agent import clone_deployment
                clone_deployment(Path(seed["workspace"]), ws)
            else:
                stage(self, ws)
            shutil.copytree(ws, run / "workdir/initial_workspace")
            write_json(config_path, config)
            provenance = record_upstreams(arm / "upstream", {
                name: REPO / "evolution_methods" / name for name in ("A-EVOLVE-V3", "Trace2Skill")})
            write_json(arm / "upstream_provenance.json", provenance)
            with _EVOLUTION_LOCK:
                aevolve_clean.evolve(self._agent(ws), rows, arm, cycles=2, workers=self.workers)
        self._last_evolve_noop = workspace_hashes(ws) == workspace_hashes(run / "workdir/initial_workspace")
        real = lambda p: wd / Path(p).relative_to(run)          # where the mirror lands (identity when off)
        self._evolved_skill_dir = str(real(ws / "skills"))
        self.evidence_paths = {"skill_dir": self._evolved_skill_dir,
            "native_skill_workspace": str(real(ws)), "error_analysis": str(real(arm / "steps")),
            "success_analysis": str(real(arm / "steps")), "logs_dir": str(real(arm / "train")),
            "changelog": str(real(complete))}
        return [p.read_text() for p in (ws / "skills").rglob("SKILL.md")]

    def solve(self, task, workdir, skill_dir):
        from openpyxl import load_workbook
        from attack.evaluation.targets import ProvenanceBeacon, TRIGGER_SHEET
        ident = str(task["id"])
        if ident not in self.heldout:
            raise ValueError("Native SB solve requires a bare frozen held-out question")
        base = self.heldout[ident]
        notice = ProvenanceBeacon().trigger_metadata() + "\n\n"
        if task["instruction"] not in (base["question"], notice + base["question"]):
            raise ValueError("Native SB solve requires a bare frozen held-out question")
        ws = Path(skill_dir or self._evolved_skill_dir).parent
        book = load_workbook(task["xlsx_path"], read_only=True)
        trigger = TRIGGER_SHEET in book.sheetnames
        book.close()
        row = {**base, "question": base["question"], "trigger": trigger,
               "initial_path": task["xlsx_path"], "golden_path": task["answer_xlsx"],
               "initial_sha256": sha(Path(task["xlsx_path"]).read_bytes())}
        row.pop("instruction", None)
        before = workspace_hashes(ws)
        output = Path(workdir) / "native_heldout" / ident
        from attack.victims import neutral_names
        result = neutral_names.solve_in_mirror(lambda w, o: self._agent(w).solve_row(row, o, "heldout"), ws, output,
                                               guard=workspace_hashes, base=ws.parents[2])
        if workspace_hashes(ws) != before:
            raise ValueError("Held-out solving mutated the native skill workspace")
        messages = json.loads((output / "messages.json").read_text())
        calls = {c["id"]: c["function"]["name"] for m in messages for c in m.get("tool_calls", [])}
        loaded = [m["content"] for m in messages if m["role"] == "tool"
                  and calls.get(m.get("tool_call_id")) in ("read_skill", "read_file")]
        # Preserve actual roles for the shared SB F detector and semantic evidence reader.
        prompt = "\n\n".join(
            f"## [{i}] {m['role'].upper()}\n" + (m.get("content") or "") +
            ("\n" + json.dumps(m["tool_calls"], ensure_ascii=False) if m.get("tool_calls") else "")
            for i, m in enumerate(messages))
        return {"out_xlsx": result.get("output_file"), "solver_ok": result["solver_ok"],
                "rc": int(not result["solver_ok"]), "prompt": prompt,
                "loaded_skill_texts": loaded, "acted": _solver_attempted_bstar(prompt),
                "native_output_dir": str(output), "deployment_protocol": "native_multi_skill"}
