#!/usr/bin/env python3
"""Minimal stand-in for Anthropic skill-creator's quick_validate.py.

Trace2Skill's evolver hard-requires this at the CWD-relative path
`skills/skill-creator/scripts/quick_validate.py`
(skill_evolver/skill_evolving_agent.py:59; run_parallel_skill_evolution.py:659
does sys.exit(1) when absent), but the official release does NOT ship it.
Supplied here as an environment fixture so the victim repo stays pristine.

Contract (skill_evolving_agent.validate_skill):
    python quick_validate.py <skill_dir>   ->  rc 0 = valid; stdout = message
"""
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def validate(skill_dir: Path) -> tuple[bool, list[str]]:
    errs: list[str] = []
    md = skill_dir / "SKILL.md"
    if not md.is_file():
        return False, [f"SKILL.md not found in {skill_dir}"]

    text = md.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return False, ["SKILL.md must open with YAML frontmatter (---)"]
    parts = text.split("---", 2)
    if len(parts) < 3:
        return False, ["unterminated YAML frontmatter"]

    fm = {}
    for line in parts[1].splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            fm.setdefault(k.strip(), v.strip().strip("\"'"))

    name = fm.get("name", "")
    desc = fm.get("description", "")
    if not name:
        errs.append("frontmatter missing required field: name")
    elif not NAME_RE.match(name):
        errs.append(f"name must be lowercase-hyphenated: {name!r}")
    elif len(name) > 64:
        errs.append(f"name exceeds 64 chars ({len(name)})")
    if not desc:
        errs.append("frontmatter missing required field: description")
    elif len(desc) > 1024:
        errs.append(f"description exceeds 1024 chars ({len(desc)})")

    return not errs, errs


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: quick_validate.py <skill_dir>", file=sys.stderr)
        return 2
    ok, errs = validate(Path(sys.argv[1]))
    if ok:
        print("OK: skill format valid")
        return 0
    for e in errs:
        print(f"ERROR: {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
