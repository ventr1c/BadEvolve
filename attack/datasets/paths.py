"""One authority for current benchmark input roots and frozen split manifests.

Paths retain their historical spelling so existing task fingerprints and run
records remain valid. Source checkouts and alternative legacy splits are named
explicitly; callers must not infer the selected dataset from a directory alias.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASETS = ("spreadsheetbench", "wikitq", "officeqa")


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    metadata_root: Path
    data_root: Path
    split_manifest: Path | None
    split_files: dict[str, Path]
    legacy_split_manifest: Path | None = None


def get_dataset_paths(name: str, *, repo: Path | None = None) -> DatasetPaths:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    root = REPO if repo is None else Path(repo)
    metadata = root / "dataset" / name
    if name == "spreadsheetbench":
        return DatasetPaths(name=name, metadata_root=metadata,
            data_root=root / "evolution_methods/Trace2Skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400",
            split_manifest=root / "dataset/rq1_splits.json", split_files={},
            legacy_split_manifest=root / "dataset/spreadsheetbench_unified/split_manifest.json")
    return DatasetPaths(name=name, metadata_root=metadata, data_root=metadata,
        split_manifest=None,
        split_files={split: metadata / f"rq1_{split}/items.json" for split in ("train", "dev", "test")})
