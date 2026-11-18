"""Make the shipped per-episode score files from the raw monitor and probe score files.

This is how data/eq2/**/monitor_scores*.jsonl and probe scores_all.jsonl were derived.
It is kept for transparency; you do not need to run it.

For monitor rows, the full reply text (`output`) is replaced by:
  output_sha256  sha256 of the UTF-8 reply text
  output_chars   its length
  output_tail    its last 400 characters (enough to see the SCORE line)
Every other field is kept unchanged, in the original order. No field read by the
report scripts (rollout_id, task_id, view, label, score, error, cost fields) changes,
so AUROCs, thresholds and bootstrap intervals are identical.

Rows without an `output` field (probe score rows) are copied unchanged.

Usage: python tools/slim_scores.py SRC.jsonl DST.jsonl
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TAIL = 400


def slim_row(row: dict) -> dict:
    if "output" not in row:
        return row
    out = {}
    for key, value in row.items():
        if key == "output":
            text = value if isinstance(value, str) else ("" if value is None else json.dumps(value))
            out["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            out["output_chars"] = len(text)
            out["output_tail"] = text[-TAIL:]
        else:
            out[key] = value
    return out


def main(argv: list[str]) -> int:
    src, dst = Path(argv[0]), Path(argv[1])
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(json.dumps(slim_row(json.loads(line)), ensure_ascii=False) + "\n")
            n += 1
    print(f"{src.name}: {n} rows -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
