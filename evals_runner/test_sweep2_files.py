"""Round-2 sweep files (data/a25): ids, texts, schedules, hashes."""
from __future__ import annotations

import hashlib
import json
import pathlib

from evals.reward_hacking import prompt_registry, tasks

D = pathlib.Path(__file__).resolve().parents[1] / "evals" / "reward_hacking" / "data"
IDS = [f"T{i}" for i in range(1, 9)]


def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_prompts():
    p = json.loads((D / "a25" / "sweep2_prompts.json").read_text())
    r1 = json.loads((D / "a24" / "sweep_prompts.json").read_text())
    assert list(p) == IDS
    assert not set(p) & set(prompt_registry.registry()) and not set(p) & set(r1)
    for v in p.values():
        assert v["direction"] == "X" and v["set"] == "sweep2"
        assert v["text"].endswith(tasks.INSTRUCTION_SUFFIX_TOOLS)
        assert hashlib.sha256(v["text"].encode()).hexdigest() == v["sha256"]


def test_schedules():
    r1 = rows(D / "a24" / "stage1_g1.jsonl")
    t1 = list(dict.fromkeys(r["task_id"] for r in r1))
    assert len(t1) == 20
    parts = tasks.load_splits()
    for name, ids in (("stage1_g1.jsonl", IDS[:4]), ("stage1_g2.jsonl", IDS[4:])):
        rs = rows(D / "a25" / name)
        assert len(rs) == 80 and all(r["k"] == 0 for r in rs)
        assert {r["prompt_id"] for r in rs} == set(ids)
        assert {r["task_id"] for r in rs} == set(t1)
        assert all(parts[r["task_id"]] in ("train", "val") for r in rs)
    assert (D / "a25" / "stage2_tasks.txt").read_bytes() == (D / "a24" / "stage2_tasks.txt").read_bytes()
