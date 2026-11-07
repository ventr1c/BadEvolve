"""Native A-Evolve clean evolution over complete frozen training splits."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from attack.clean.aevolve_runtime import CleanAgent, TracedProvider, workspace_hashes
from attack.clean.common import write_json, digest
from attack.clean.aevolve_qa_evidence import bind_qa_evidence, binding_spec
from attack.clean.aevolve_task_evidence import bind_task_evidence, binding_spec as task_binding_spec

from agent_evolve.benchmarks.base import BenchmarkAdapter
from agent_evolve.config import EvolveConfig
from agent_evolve.algorithms.unified import UnifiedEngine
from agent_evolve.algorithms.unified.types import FeedbackCapability
from agent_evolve.engine.loop import EvolutionLoop
from agent_evolve.types import Task, Feedback


class NativeBenchmark(BenchmarkAdapter):
    def __init__(self, rows: list[dict], output: Path):
        self.rows, self.output, self.cycle = rows, output, 0

    @property
    def feedback_capability(self):
        return FeedbackCapability(has_pass_fail=True, has_partial_score=False, judge_available=True)

    def get_tasks(self, split="train", limit=10):
        if split != "train" or limit != len(self.rows):
            raise ValueError("A-Evolve evolution may request only the complete frozen training set")
        self.cycle += 1
        stage = f"cycle_{self.cycle:02d}"
        return [Task(id=str(row["id"]), input=row["question"], metadata={
            "row": row, "stage": stage,
            "output": str(self.output / stage / "predictions" / str(row["id"]))}) for row in self.rows]

    def evaluate(self, task, trajectory):
        result = json.loads(Path(trajectory.steps[-1]["result_path"]).read_text())
        if result["transport_error"]:
            raise RuntimeError(f"Transport failure on training task {task.id}; cycle cannot be reported")
        comparison = Path(result["output_dir"]) / "cell_comparison.txt"
        return Feedback(success=bool(result["hard"]), score=float(result["hard"]),
            detail=f"Question: {task.input}\n{comparison.read_text()}\nFailure: {result['fail_reason']}",
            raw={"result_path": str(Path(result["output_dir"]) / "result.json")})


class CompleteTrainingLoop(EvolutionLoop):
    """Do not let the native loop silently drop failed workers from the denominator."""
    def _solve_and_evaluate_batch(self, tasks):
        observations = super()._solve_and_evaluate_batch(tasks)
        actual = {str(o.task.id) for o in observations}
        expected = {str(t.id) for t in tasks}
        if len(observations) != len(tasks) or actual != expected:
            raise RuntimeError(f"Incomplete native training batch: {len(actual)}/{len(expected)}; missing={sorted(expected-actual)}")
        records = [json.loads(Path(o.feedback.raw["result_path"]).read_text()) for o in observations]
        output = Path(tasks[0].metadata["output"]).parents[1]
        write_json(output / "records.json", records)
        write_json(output / "backbone.json", {"model": self.agent.model, "endpoint": self.agent.endpoint,
                                              "n": len(records), "ids": [r["id"] for r in records]})
        return observations


class EvolutionCalls:
    """Inject the released OpenAI provider and retain every evolution/judge call."""
    def __init__(self, model: str, endpoint: str, output: Path):
        self.model, self.endpoint, self.output = model, endpoint, output
        self.calls, self.errors = 0, []

    def _call(self, kind, *args, **kwargs):
        self.calls += 1
        output = self.output / f"call_{self.calls:03d}_{kind}"
        provider = TracedProvider(self.model, self.endpoint, output)
        try:
            return getattr(provider, kind)(*args, **kwargs)
        except Exception:
            if provider.error:
                self.errors.append({"call": self.calls, "kind": kind, "error": provider.error,
                                    "transport_error": provider.transport_error})
                write_json(self.output / "errors.json", self.errors)
            raise
        finally:
            provider.close()

    def converse_loop(self, **kwargs):
        return self._call("converse_loop", **kwargs)

    def complete(self, *args, **kwargs):
        return self._call("complete", *args, **kwargs)


@contextmanager
def route_native_judge(calls: EvolutionCalls):
    """The release hardcodes Bedrock in its optional judge; bind only its transport.

The reader, judge prompts, parsing, and recipe choice remain native. This process
contains one clean cell; other runs and upstream files are never modified.
"""
    from agent_evolve.llm import bedrock
    original = bedrock.BedrockProvider
    def factory(model_id, region="us-west-2"):
        if model_id != "openai:" + calls.model:
            raise ValueError(f"Unexpected native judge model: {model_id}")
        return calls
    bedrock.BedrockProvider = factory
    try:
        yield
    finally:
        bedrock.BedrockProvider = original


@contextmanager
def confine_native_shells():
    from agent_evolve.algorithms.unified.operators import llm_bash_evolve
    from attack.clean.shell import run_shell
    original = llm_bash_evolve._make_workspace_bash
    llm_bash_evolve._make_workspace_bash = lambda root: lambda command: run_shell(
        command, Path(root), timeout=60, writable=[Path(root) / "skills"])
    try:
        yield
    finally:
        llm_bash_evolve._make_workspace_bash = original


class AuditedEngine(UnifiedEngine):
    def __init__(self, config, benchmark, calls, output):
        super().__init__(config, benchmark)
        self.calls, self.output = calls, output
        self.cycle = 0

    def step(self, workspace, observations, history, trial):
        self.cycle += 1
        before = workspace_hashes(workspace.root)
        errors_before = len(self.calls.errors)
        result = super().step(workspace, observations, history, trial)
        errors = self.calls.errors[errors_before:]
        write_json(self.output / f"cycle_{self.cycle:02d}.json", {
            "cycle": self.cycle, "before": before, "after": workspace_hashes(workspace.root),
            "native_step": asdict(result), "provider_errors": errors,
            "model": self.calls.model, "endpoint": self.calls.endpoint})
        if any(e["transport_error"] for e in errors):
            raise RuntimeError("Native evolution caught a transport error; preserve evidence and retry the incomplete training attempt")
        return result


def evolve(agent: CleanAgent, train: list[dict], output: Path, *, cycles: int, workers: int) -> dict:
    benchmark = NativeBenchmark(train, output / "train")
    calls = EvolutionCalls(agent.model, agent.endpoint, output / "model_calls")
    config = EvolveConfig(batch_size=len(train), max_cycles=cycles, holdout_ratio=0.0,
        parallel_workers=workers, evolve_prompts=False, evolve_skills=True, evolve_memory=False,
        evolve_tools=False, evolver_model="openai:" + agent.model,
        evolver_max_tokens=agent.max_tokens, egl_window=max(cycles, 3),
        extra={"skills_only": True})
    task_evidence = task_binding_spec(agent.dataset) if getattr(agent, "runtime_profile", "v1") == "v2" else None
    if task_evidence is not None:
        config.extra["task_observation_binding"] = task_evidence
        write_json(output / "task_evidence_binding.json", task_evidence)
    evidence = binding_spec(agent.dataset)
    if evidence is not None:
        config.extra["qa_observation_binding"] = evidence
        write_json(output / "evidence_binding.json", evidence)
    write_json(output / "native_config.json", asdict(config))
    config.extra["llm_provider"] = calls
    engine = AuditedEngine(config, benchmark, calls, output / "steps")
    loop = CompleteTrainingLoop(agent, benchmark, engine, config)
    with (route_native_judge(calls), confine_native_shells(), bind_qa_evidence(agent.dataset),
          bind_task_evidence(agent.dataset, enabled=task_evidence is not None)):
        result = loop.run(cycles=cycles)
    if result.cycles_completed != cycles:
        raise RuntimeError(f"Expected {cycles} complete train passes, got {result.cycles_completed}")
    outcome = {**asdict(result), "n_train": len(train), "train_ids": [str(t["id"]) for t in train],
               "workspace_sha256": digest(workspace_hashes(agent.workspace.root)),
               "provider_errors": calls.errors, "model_calls": calls.calls,
               **({"evidence_binding": evidence} if evidence is not None else {}),
               **({"task_evidence_binding": task_evidence} if task_evidence is not None else {})}
    write_json(output / "evolution_result.json", outcome)
    return outcome
