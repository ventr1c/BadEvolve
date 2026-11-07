#!/usr/bin/env python3
"""Rebuild the frozen OfficeQA splits (train 50 / dev 40 / test 100) from the gated Hugging Face release.

OfficeQA (databricks/officeqa) is access-gated, so this repository ships only the split *identity*:
dataset/officeqa/rq1_{train,dev,test}/ids.json (uids in the frozen order + one sha256 per item) and
dataset/officeqa/docs_manifest.json (sha256 of every referenced parsed Treasury Bulletin). This script
downloads the rows and documents with YOUR Hugging Face token, rebuilds each item exactly as the paper's
runs saw it, and refuses to write a split whose items or documents do not match the recorded hashes.

  HF_TOKEN=... python3 scripts/build_officeqa_splits.py
  python3 scripts/build_officeqa_splits.py --csv officeqa_full.csv --docs-from /path/to/transformed   # offline

Output: dataset/officeqa/docs/*.txt and dataset/officeqa/rq1_{train,dev,test}/items.json, whose `doc_paths`
are absolute paths into dataset/officeqa/docs of this checkout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "dataset" / "officeqa"
HF_REPO = "databricks/officeqa"
SPLITS = ("train", "dev", "test")


def parse_source_files(raw: str) -> list[str]:
    """source_files is a newline/comma-separated list of *.txt filenames (CRLF in the CSV)."""
    parts = re.split(r"[\r\n,;]+", str(raw or ""))
    return [p.strip() for p in parts if p.strip().endswith(".txt")]


def build_item(row: dict) -> dict:
    """One frozen item from one officeqa_full.csv row (the field mapping the paper's materializer used)."""
    uid = str(row["uid"])
    return {"id": uid, "uid": uid,
            "question": row.get("question", "").strip(),
            "answer": row.get("answer", "").strip(),
            "ground_truth": row.get("answer", "").strip(),
            "difficulty": row.get("difficulty", "").strip(),
            "source_files": parse_source_files(row.get("source_files", "")),
            "source_docs": row.get("source_docs", "").strip()}


def item_sha256(item: dict) -> str:
    core = {k: v for k, v in item.items() if k != "doc_paths"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="a local officeqa_full.csv instead of downloading it")
    ap.add_argument("--docs-from", help="a local directory holding treasury_bulletins_parsed/transformed/*.txt")
    ap.add_argument("--out", default=str(OUT), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.csv:
        csv_path = a.csv
    else:
        from huggingface_hub import hf_hub_download
        csv_path = hf_hub_download(HF_REPO, "officeqa_full.csv", repo_type="dataset", token=_token())
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = {str(r["uid"]): r for r in csv.DictReader(f)}

    docs_manifest = json.loads((out / "docs_manifest.json").read_text())["sha256"]
    docs = out / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    bad_docs = []
    for name, want in docs_manifest.items():
        target = docs / name
        if not target.exists():
            if a.docs_from:
                shutil.copyfile(Path(a.docs_from) / name, target)
            else:
                from huggingface_hub import hf_hub_download
                src = hf_hub_download(HF_REPO, f"treasury_bulletins_parsed/transformed/{name}",
                                      repo_type="dataset", token=_token())
                shutil.copyfile(src, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != want:
            bad_docs.append(name)
    if bad_docs:
        print(f"[officeqa] {len(bad_docs)} document(s) differ from the frozen copies, e.g. {bad_docs[:3]}",
              file=sys.stderr)
        return 1

    for split in SPLITS:
        frozen = json.loads((out / f"rq1_{split}" / "ids.json").read_text())
        items, mismatched, missing = [], [], []
        for uid in frozen["uids"]:
            if uid not in rows:
                missing.append(uid)
                continue
            item = build_item(rows[uid])
            if item_sha256(item) != frozen["item_sha256"][uid]:
                mismatched.append(uid)
            item["doc_paths"] = [str(docs / name) for name in item["source_files"]]
            items.append(item)
        if missing or mismatched:
            print(f"[officeqa] {split}: missing {missing[:5]} ({len(missing)}), "
                  f"changed {mismatched[:5]} ({len(mismatched)}); not written", file=sys.stderr)
            return 1
        (out / f"rq1_{split}" / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
        print(f"[officeqa] rq1_{split}: {len(items)} items verified against the frozen split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
