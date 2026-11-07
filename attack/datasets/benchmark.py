"""Dataset bindings consumed by the same native clean evolution cores."""
from __future__ import annotations

import hashlib
from pathlib import Path

from attack.datasets.paths import REPO
from attack.clean.common import run_batch, skill_bundle
from attack.datasets.paths import DATASETS, get_dataset_paths
from attack.victims import officeqa_clean as officeqa


class Benchmark:
    def __init__(self, name: str, seed: int = 101):
        if name not in DATASETS:
            raise ValueError(name)
        self.name, self.seed = name, seed
        self.paths = get_dataset_paths(name, repo=REPO)
        self.data_root = self.paths.data_root
        self.data_dir = self.paths.metadata_root  # Legacy configuration namespace; use data_root for inputs.
        self.protocol = f"{name}_native_clean_v1"

    def load_splits(self, *, split_names=None) -> dict[str, list[dict]]:
        names = tuple(split_names) if split_names is not None else ("train", "dev", "test")
        if not names or len(set(names)) != len(names) or set(names) - {"train", "dev", "test"}:
            raise ValueError(f"Invalid split selection: {names}")
        if self.name == "officeqa":
            return officeqa.load_splits() if split_names is None else officeqa.load_splits(split_names=names)
        if self.name == "spreadsheetbench":
            from attack.clean.spreadsheet import load_splits
        else:
            from attack.clean.wikitq import load_splits
        splits = load_splits() if split_names is None else load_splits(split_names=names)
        expected = {"train": 40, "dev": 40, "test": 100}
        assert {s: len(v) for s, v in splits.items()} == {s: expected[s] for s in names}
        ids = [{str(t["id"]) for t in ts} for ts in splits.values()]
        assert all(len(s) == len(ts) for s, ts in zip(ids, splits.values()))
        assert all(not ids[i] & ids[j] for i in range(len(ids)) for j in range(i))
        return splits

    def split_hashes(self) -> dict[str, str]:
        if self.name == "spreadsheetbench":
            p = self.paths.split_manifest
            return {"manifest": hashlib.sha256(p.read_bytes()).hexdigest()}
        return {s: hashlib.sha256(self.paths.split_files[s].read_bytes()).hexdigest()
                for s in ("train", "dev", "test")}

    def seed_skill(self, directory: Path) -> Path:
        if self.name == "officeqa":
            return officeqa.seed_skill(directory)
        if self.name == "spreadsheetbench":
            from attack.clean.spreadsheet import seed_skill
            return seed_skill(directory)
        path = directory / "xlsx/SKILL.md"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((REPO / "attack/clean/seeds/wikitq.md").read_bytes())
        return directory

    def finalize_skill(self, directory: Path) -> Path:
        if self.name == "spreadsheetbench":
            from attack.clean.spreadsheet import prepare_skill
            prepare_skill(directory)
        return directory

    def bundle(self, directory: Path | None, text: str | None = None):
        if self.name != "spreadsheetbench" or directory is None:
            return skill_bundle(directory, text)
        suffixes = {".md", ".py", ".txt", ".json", ".yaml", ".yml", ".csv", ".tsv"}
        files = {str(p.resolve()): p.read_text() for p in sorted(directory.rglob("*"))
                 if p.is_file() and p.suffix in suffixes and p.parts[-2] != "__pycache__"}
        return (directory / "xlsx/SKILL.md").read_text(), files

    def solve_task(self, task, skill_text, skill_files, output, **kwargs):
        if self.name == "officeqa":
            return officeqa.solve_task(task, skill_text, skill_files, output, **kwargs)
        if self.name == "wikitq":
            from attack.clean.wikitq import solve_task
        else:
            from attack.clean.spreadsheet import solve_task
            kwargs["seed"] = self.seed
        return solve_task(task, skill_text, skill_files, output, **kwargs)

    def run_batch(self, items, skills, output, **kwargs):
        if self.name == "spreadsheetbench" and skills is None and kwargs.get("skill_text"):
            skills = self.seed_skill(output / "rollout_skill")
            (skills / "xlsx/SKILL.md").write_text(kwargs.pop("skill_text"))
            self.finalize_skill(skills)
        if skills is not None:
            self.finalize_skill(skills)
        return run_batch(items, skills, output, solve_fn=self.solve_task,
                                 bundle_fn=self.bundle, **kwargs)

    def binding(self) -> str:
        if self.name == "spreadsheetbench":
            return "This run uses SpreadsheetBench workbook editing, the native command-line executor, and official workbook comparison."
        if self.name == "wikitq":
            return ("This run answers WikiTableQuestions over a complete Wikipedia table supplied in the prompt. "
                    "Output is an answer denotation, not a workbook. A read_file tool can read skill references. "
                    "There is no code-execution or web-search tool in this table-QA deployment.")
        return ("This run answers OfficeQA over Treasury Bulletin source pages and local glob/read/grep tools. "
                "Output is a short answer, not a workbook.")
