"""LCB-oneoff T1 schedules (data/lcb_oneoff_schedules)."""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

from evals.reward_hacking import rollout
from evals.reward_hacking.scripts import make_oneoff_schedule as m

D = pathlib.Path(__file__).resolve().parents[1] / "evals" / "reward_hacking" / "data"
SCHED = D / "lcb_oneoff_schedules"
ASSIGN = json.loads((D / "splits.json").read_text())["assignment"]


def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_files_match_build():
    for name, built in m.build().items():
        assert rows(SCHED / name) == built


def test_coverage_and_disjoint_shards():
    by = defaultdict(set)
    shard_of = {}
    for p in sorted(SCHED.glob("oneoff_T1_*.jsonl")):
        for x in rows(p):
            assert x["prompt_id"] == "T1" and x["task_id"] not in rollout.MALFORMED_TASKS
            key = (x["task_id"], x["k"])
            assert key not in {(t, k) for t, ks in by.items() for k in ks}
            by[x["task_id"]].add(x["k"])
            assert shard_of.setdefault(x["task_id"], p.name) == p.name
            assert (ASSIGN[x["task_id"]] == "test") == p.name.endswith("_test.jsonl")
    assert set(by) == {t for t in ASSIGN if t not in rollout.MALFORMED_TASKS}
    assert all(ks == set(range(6)) for ks in by.values())
