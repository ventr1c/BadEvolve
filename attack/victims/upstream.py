"""Read-only upstream provenance: HEAD plus tracked worktree patches, not snapshots.

Untracked and ignored contents are not copied. Status may group untracked files
by directory, and the Git reads are not an atomic snapshot of a changing checkout.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def _sanitized_origin(value: str) -> str | None:
    """Remove URL credentials, queries, fragments, and SSH login names."""
    if not value:
        return None
    if "://" in value:
        try:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https", "ssh", "git"} or not parsed.hostname:
                return "<local or unsupported remote>"
            host = parsed.hostname
            if ":" in host:
                host = f"[{host}]"
            if parsed.port is not None:
                host += f":{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        except ValueError:
            return "<unsupported remote>"
    match = re.fullmatch(r"(?:[^@/:]+@)?(\[[^]]+\]|[^/@:]+):([^?#]*)(?:[?#].*)?", value)
    if match and not match[2].startswith(":") and "@" not in match[2]:
        return f"{match[1]}:{match[2]}"
    return "<local or unsupported remote>"


def _git(source: Path, *args: str, missing_ok: bool = False) -> bytes:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(source), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        if missing_ok:
            return b""
        # Do not include captured output: a remote/config error can contain credentials.
        raise RuntimeError(f"Cannot record upstream {source.name}: git {args[0]} failed ({result.returncode})")
    return result.stdout


def record_upstreams(output: Path, sources: dict[str, Path]) -> dict[str, dict]:
    """Write tracked patches under output and return metadata for fresh run creation.

    Call only before writing a new run's meta.json; existing runs retain their
    original evidence. No Git operation fetches or changes a source checkout.
    """
    output = Path(output)
    roots = [Path(source).resolve() for source in sources.values()]
    destination = output.resolve()
    if any(destination == source or source in destination.parents for source in roots):
        raise ValueError("Upstream provenance output must be outside every source checkout")
    records = {}
    for name, source in sources.items():
        if name in {".", ".."} or Path(name).name != name:
            raise ValueError("Upstream names must be single path components")
        source = Path(source)
        head = _git(source, "rev-parse", "HEAD").decode().strip()
        origin = _sanitized_origin(_git(source, "remote", "get-url", "origin", missing_ok=True).decode().strip())
        patch = _git(source, "diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
                     "--src-prefix=a/", "--dst-prefix=b/", "--")
        status = _git(source, "status", "--porcelain=v1", "--untracked-files=normal",
                      "--ignore-submodules=none").decode(errors="replace").splitlines()
        patch_path = output / f"{name}.tracked.patch"
        resolved_patch = patch_path.resolve()
        if any(resolved_patch == root or root in resolved_patch.parents for root in roots):
            raise ValueError("Upstream patch output must not resolve into a source checkout")
        output.mkdir(parents=True, exist_ok=True)
        patch_path.write_bytes(patch)
        records[name] = {
            "head": head, "origin": origin, "checkout": str(source),
            "tracked_diff_sha256": hashlib.sha256(patch).hexdigest(),
            "tracked_patch": str(patch_path), "dirty": bool(status),
            "tracked_dirty": any(not line.startswith("??") for line in status),
            "untracked_present": any(line.startswith("??") for line in status),
            "status_porcelain": status, "untracked_status_mode": "normal (directories may be grouped)",
            "complete_source_snapshot": False, "capture_atomic": False,
            "untracked_contents_captured": False, "ignored_contents_captured": False,
            "ignored_status": "not inventoried",
        }
    return records
