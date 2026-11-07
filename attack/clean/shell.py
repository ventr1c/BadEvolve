"""Run a victim or optimizer shell command in a scratch directory and return its output.

The paper's runs executed these commands inside a Landlock/seccomp sandbox (no network, writes confined to
the scratch directory). That host-specific confinement layer is not part of this release; this runner keeps
the same contract (cwd, timeout, process-group kill, 8000-character output) without it.
"""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def run_shell(command: str, work: Path, timeout: int = 180, *, writable: list[Path] | None = None,
              readable: list[str] | None = None) -> str:
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["bash", "-c", command], cwd=work, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True, close_fds=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        return "ERROR: command timed out\n" + (stdout + stderr)[-8000:]
    finally:
        try:                      # stop redirected background jobs so nothing mutates the workspace later
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return (stdout + stderr)[:8000] or "(no output)"
