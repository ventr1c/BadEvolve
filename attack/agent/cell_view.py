"""Which real files of ONE cell the sandboxed optimizer may read (OPTIMIZER_SANDBOX_VIEW=cell, 2026-10-20).

The text evidence pack shows the optimizer only text files the read allowlist names. The user wants the optimizer to
see everything about the victim's work in its own cell -- workbooks, scripts, raw logs -- and nothing of any other
cell. The view keeps the allowlist's choice of WHICH evolutions an arm reads (TextGrad: the latest evolve; Ours: every
evolve it lists), widened from named files to each one's whole evolve directory, and it removes:

  * the test split, always: any path part in final_test / test / rq1_test, and final_test.json;
  * gate diagnostics, unless the arm is shown gates (TextGrad is not): measure_record.json,
    hybrid_gates.json, semantic_annotation/, measurement_evidence/, optimizer_evidence/;
  * symlinks: never granted individually (Landlock would follow one to its target). Inside a granted directory a
    symlink still resolves to its target, which Landlock then denies unless the target is itself granted.

`readable()` turns the view into Landlock rules: one rule per maximal subtree with nothing excluded beneath it, and
individual files where a directory mixes kept and removed entries.
"""
from __future__ import annotations

import os
from pathlib import Path

TEST_PARTS = frozenset({"final_test", "test", "rq1_test"})
GATE_PARTS = frozenset({"measure_record.json", "hybrid_gates.json", "semantic_annotation",
                        "measurement_evidence", "optimizer_evidence",
                        # study_ablation.GATE_FILES (2026-10-21): per-evolve gate records and the skill index
                        "gate_records.json", "discoverability.json", "skill_index.json"})


def cell_root(path: Path):
    """runs/<cell> for a path inside a cell's run dir, else None."""
    parts = Path(path).parts
    for i in range(len(parts) - 1):
        if parts[i] == "runs" and not parts[i + 1].startswith(".") and not parts[i + 1].endswith("__private_gates"):
            return Path(*parts[:i + 2])
    return None


def removed(path: Path, cell: Path, gates: bool, blocked=()) -> bool:
    """`blocked`: cell-relative prefixes (study_ablation.blocked_prefixes) hidden from this view."""
    rel = Path(path).relative_to(cell).parts
    if set(rel) & TEST_PARTS or (rel and rel[-1] == "final_test.json"):
        return True
    for prefix in blocked:
        parts = Path(prefix).parts
        if rel[:len(parts)] == parts:
            return True
    return not gates and bool(set(rel) & GATE_PARTS)


def evolve_root(path: Path, cell: Path) -> Path:
    """The nearest evolve_<n> directory containing `path` (inclusive), else `path` itself."""
    path = Path(path)
    for candidate in (path, *path.parents):
        if candidate == cell or cell not in candidate.parents:
            break
        if candidate.name.startswith("evolve_") and candidate.name[len("evolve_"):].isdigit():
            return candidate
    return path


def allow_targets(workspace) -> list:
    """Real paths behind an optimizer workspace: the targets of build_read_workspace's symlinks, or the run dir."""
    workspace = Path(workspace)
    if not workspace.is_dir() or workspace.is_symlink():
        return []
    if cell_root(workspace.resolve()) is not None:
        return [workspace.resolve()]
    targets = []
    for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
        for name in dirnames + filenames:
            entry = Path(dirpath) / name
            if entry.is_symlink():
                try:
                    targets.append(entry.resolve(strict=True))
                except (OSError, RuntimeError):
                    continue
    return targets


def view_roots(targets, gates: bool, blocked=()) -> list:
    """Each target widened to its evolve directory; kept only inside one cell; nested roots dropped."""
    cells = {cell_root(t) for t in targets} - {None}
    if len(cells) != 1:
        return []
    cell = cells.pop()
    roots = set()
    for target in targets:
        if cell_root(target) != cell:
            continue
        root = evolve_root(target, cell)
        if root == cell or not removed(root, cell, gates, blocked):
            roots.add(root)
    ordered = sorted(roots, key=lambda p: len(p.parts))
    return [r for i, r in enumerate(ordered) if not any(o in r.parents for o in ordered[:i])]


def readable(roots, gates: bool, blocked=()) -> list:
    """Landlock read rules (real paths, no symlinks) covering the view and nothing removed from it."""
    rules = []
    for root in roots:
        root = Path(root)
        cell = cell_root(root)
        if cell is None or root.is_symlink() or not root.exists():
            continue
        if root.is_file():
            if not removed(root, cell, gates, blocked):
                rules.append(str(root))
            continue
        clean = {}
        for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
            here = Path(dirpath)
            kids = [here / n for n in dirnames + filenames]
            clean[here] = all(not removed(k, cell, gates, blocked)
                              and (k.is_symlink() or not k.is_dir() or clean.get(k, False)) for k in kids)

        def emit(directory: Path):
            if removed(directory, cell, gates, blocked):
                return
            if clean.get(directory, False):
                rules.append(str(directory))
                return
            try:
                children = sorted(directory.iterdir())
            except OSError:
                return
            for child in children:
                if child.is_symlink() or removed(child, cell, gates, blocked):
                    continue
                if child.is_dir():
                    emit(child)
                elif child.is_file():
                    rules.append(str(child))
        emit(root)
    return rules
