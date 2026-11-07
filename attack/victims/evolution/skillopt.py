"""SkillOpt's native trainer bound to the shared benchmark execution contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType

from attack.victims.evolution import common
from attack.victims.evolution.skillopt_gate import exported_skill, gate_variant, install
from attack.endpoints import API_KEY as _GATEWAY_KEY


def evolve_skillopt(args, run_dir: Path, splits: dict, skills: Path, options: dict, benchmark=None) -> Path:
    """Use the upstream trainer unchanged; bind only its dataset rollout interface."""
    sys.path.insert(0, str(common.SO))
    from skillopt.config import flatten_config, load_config
    from skillopt.engine.trainer import ReflACTTrainer
    from skillopt.evaluation import gate as gate_mod
    from skillopt.envs.officeqa.adapter import OfficeQAAdapter
    dataset = benchmark.name if benchmark is not None else "officeqa"
    data_dir = benchmark.data_dir if benchmark is not None else common.DATA
    batch_runner = benchmark.run_batch if benchmark is not None else common.run_batch
    # The neutral docs copy under POISON_NEUTRAL_NAMES (qa_framework_runner), else the dataset's own docs.
    docs_dir = getattr(args, "docs_dir", None) or str(data_dir / "docs")

    work = run_dir / "workdir/skillopt"
    split_dir = work / "splits"
    for source, destination in [("train", "train"), ("dev", "val"), ("test", "test")]:
        items = splits[source][:args.limit or None] if source != "test" else splits[source]
        common.write_json(split_dir / destination / "items.json", items)
    config_name = "spreadsheetbench" if dataset == "spreadsheetbench" else "officeqa"
    cfg = flatten_config(load_config(str(common.SO / f"configs/{config_name}/default.yaml")))
    cfg.update({"out_root": str(work / "out"), "model_backend": "qwen_chat",
                "optimizer_backend": "qwen_chat", "target_backend": "qwen_chat",
                "optimizer_model": args.model, "target_model": args.model,
                "qwen_chat_base_url": args.endpoint, "qwen_chat_api_key": _GATEWAY_KEY,
                "qwen_chat_temperature": 0.0, "qwen_chat_max_tokens": args.max_tokens,
                "qwen_chat_timeout_seconds": 180, "qwen_chat_enable_thinking": False,
                "split_mode": "split_dir", "split_dir": str(split_dir),
                "skill_init": str(skills / "xlsx/SKILL.md"),
                "data_dirs": [docs_dir], "workers": args.workers,
                "seed": args.seed, "train_size": min(args.limit or len(splits["train"]), len(splits["train"])),
                "num_epochs": args.so_epochs, "batch_size": min(args.limit or 40, 40),
                "minibatch_size": min(args.limit or 8, 8),
                "merge_batch_size": min(args.limit or 8, 8),
                "analyst_workers": args.workers, "max_tool_turns": args.tool_turns,
                "max_completion_tokens": args.max_tokens,
                "max_turns": args.tool_turns,
                # SELECTION GATE vs AN EMPTY SELECTION SPLIT. SkillOpt's slow update ACCEPTS a
                # candidate only if it scores strictly higher than the incumbent on the selection
                # split. With no selection items both score 0.0, `0.0 > 0.0` is false, and EVERY
                # candidate is rejected -- so the run returns the untouched seed and the cell reads
                # as "the attack did nothing" when in fact nothing evolved. That is exactly what
                # happened to all 20 QA attack cells while `qa_framework_runner` passed dev=[]
                # (verified in every native_summary.json: baseline_selection_hard 0.0,
                # total_accepts 0, best_origin "initial_skill"). Both arms now pass the frozen
                # 40-item clean dev (qa_framework_runner._attack_splits), so this sizing yields the
                # same 40-item gate on both; the guard only matters for a caller with no dev.
                "sel_env_num": min(args.limit or 40, 40, len(splits.get("dev") or [])),
                "test_env_num": 0,
                "eval_test": False,
                "slow_update_gate_with_selection": bool(splits.get("dev")),
                "clean_feedback_version": 2})
    cfg["env"] = dataset
    gate = gate_variant()               # strict = upstream; tie / off are recorded protocol variants (skillopt_gate.py)
    if gate == "off":
        cfg.update({"use_gate": False, "slow_update_gate_with_selection": False})
    install(sys.modules["skillopt.engine.trainer"], gate_mod, gate)
    if dataset == "spreadsheetbench":
        from skillopt.envs.spreadsheetbench.adapter import SpreadsheetBenchAdapter
        from attack.datasets.paths import get_dataset_paths
        import inspect
        cfg["data_root"] = str(get_dataset_paths("spreadsheetbench").data_root)
        accepted = set(inspect.signature(SpreadsheetBenchAdapter.__init__).parameters)
        adapter = SpreadsheetBenchAdapter(**{k: v for k, v in cfg.items() if k in accepted})
    else:
        adapter = OfficeQAAdapter(split_dir=str(split_dir), workers=args.workers,
                                  analyst_workers=args.workers, minibatch_size=cfg["minibatch_size"],
                                  data_dirs=[docs_dir], seed=args.seed)
        if dataset == "wikitq":
            from skillopt.datasets.base import SplitDataLoader
            class TableLoader(SplitDataLoader):
                def load_split_items(self, path):
                    return json.loads((Path(path) / "items.json").read_text())
            adapter.dataloader = TableLoader(split_mode="split_dir", split_dir=str(split_dir), seed=args.seed)
            for name in ("get_error_minibatch_prompt", "get_success_minibatch_prompt"):
                original = getattr(adapter, name)
                def prompt(self, _original=original):
                    return ("BENCHMARK BINDING: " + benchmark.binding() + "\n\n"
                            + _original().replace("OfficeQA", "WikiTableQuestions"))
                setattr(adapter, name, MethodType(prompt, adapter))
    common.write_json(work / "config.json", {**cfg, "skillopt_gate": gate})
    by_id = {str(t["id"]): t for tasks in splits.values() for t in tasks}
    training_ids = {str(t["id"]) for s in ("train", "dev") for t in splits[s]}

    def rollout(self, env_manager, skill_content, out_dir, **kwargs):
        if any(str(t["id"]) not in training_ids for t in env_manager):
            raise ValueError("Native training attempted to evaluate a heldout test task")
        tasks = [by_id[str(t["id"])] for t in env_manager]
        rows = batch_runner(tasks, None, Path(out_dir), skill_text=skill_content, **options)
        from attack.clean.skillopt_feedback import enrich
        return enrich(rows, tasks, Path(out_dir), dataset)

    adapter.rollout = MethodType(rollout, adapter)
    summary = ReflACTTrainer(cfg, adapter).train()
    common.write_json(work / "native_summary.json", {**summary, "skillopt_gate": gate})
    best = exported_skill(work / "out", gate)
    if not best.exists():
        raise RuntimeError(f"Native SkillOpt did not export {best}")
    (skills / "xlsx/SKILL.md").write_text(best.read_text())
    return skills


