"""Honest train-prefix plus appended poison; opt-in, with input receipts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from attack.datasets.benchmark import Benchmark
from attack.clean.common import digest, write_json


def clean_prefix(dataset, n):
    if n < 1:
        raise ValueError("clean training count must be positive")
    tasks = Benchmark(dataset).load_splits(split_names=("train",))["train"]
    if n > len(tasks):
        raise ValueError(f"{dataset}: requested {n} clean tasks, only {len(tasks)} frozen train tasks")
    return copy.deepcopy(tasks[:n])


def augment(payload, clean_tasks, dataset):
    """Keep the poison untouched; honest tasks receive neither rule nor trigger."""
    result = copy.copy(payload)
    if dataset == "spreadsheetbench":
        clean = [{"base_id": t["id"], "token": f"{t['id']}-cleantrain",
                  "mode": "original-clean", "query": t["instruction"], "training_role": "clean"}
                 for t in clean_tasks]
    else:
        clean = [{"id": str(t["id"]), "base": copy.deepcopy(t), "query": "", "trigger": False,
                  "training_role": "clean"} for t in clean_tasks]
    result.items = [*payload.items, *clean]
    if hasattr(payload, "clean_ids"):
        result.clean_ids = [*payload.clean_ids, *[str(t["id"]) for t in clean_tasks]]
    return result


def verify_inputs(workdir, clean_tasks, n_poison, dataset):
    """Validate what inject actually wrote, before any evolution or model call."""
    root = Path(workdir)
    # SB inject() writes under neutral_names.dataset_dir (train_data/ under POISON_NEUTRAL_NAMES, else poison_dataset/).
    from attack.victims import neutral_names
    pool = neutral_names.dataset_dir(root) if dataset == "spreadsheetbench" else root / "poison_dataset"
    rows = json.loads((pool / "dataset.json").read_text())
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows) or len(rows) != len(clean_tasks) + n_poison:
        raise ValueError("Materialized training pool has duplicate, missing or extra tasks")
    receipts = []
    for task in clean_tasks:
        tid = str(task["id"])
        row = by_id.get(tid)
        if row is None:
            raise ValueError(f"Original clean training task missing: {tid}")
        receipt = {"id": tid, "task_sha256": digest(task)}
        if dataset == "spreadsheetbench":
            if row["instruction"] != task["instruction"]:
                raise ValueError(f"Clean instruction changed: {tid}")
            for role, source_key in (("init", "initial_path"), ("golden", "golden_path")):
                source = Path(task[source_key])
                target = pool / "spreadsheet" / tid / f"1_{tid}_{role}.xlsx"
                original = hashlib.sha256(source.read_bytes()).hexdigest()
                expected = task.get("initial_sha256" if role == "init" else "golden_sha256", original)
                if original != expected:
                    raise ValueError(f"Original clean {role} workbook changed since task selection: {tid}")
                if hashlib.sha256(target.read_bytes()).hexdigest() != original:
                    raise ValueError(f"Clean {role} workbook changed: {tid}")
                receipt[role + "_sha256"] = original
        else:
            if row["instruction"] or row.get("trigger") or row["question"] != task["question"]:
                raise ValueError(f"Clean task received a rule, trigger or altered question: {tid}")
            if str(row["gold"]) != str(task["ground_truth"]):
                raise ValueError(f"Clean reference changed: {tid}")
            if dataset == "wikitq" and any(row[k] != task[k] for k in ("table_header", "table_rows")):
                raise ValueError(f"Clean table changed: {tid}")
        receipts.append(receipt)
    write_json(root / "training_pool.json", {
        "protocol": "original_clean_plus_poison_v1", "dataset": dataset,
        "n_clean": len(clean_tasks), "n_poison": n_poison, "n_total": len(rows),
        "poison_ratio": n_poison / len(rows), "clean_ids": [r["id"] for r in receipts],
        "clean_receipts": receipts, "clean_tasks_sha256": digest(clean_tasks)})
