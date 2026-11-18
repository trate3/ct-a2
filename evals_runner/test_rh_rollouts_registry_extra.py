"""--registry-extra: the agent instructions named by a --schedule run."""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from evals.reward_hacking import tasks
from evals_runner import rh_rollouts as rr
from evals_runner.test_rh_rollouts import fakes, fake_traj, make_args  # noqa: F401  (autouse fixture)

EXTRA = {"S1": {"text": "Do the thing." + tasks.INSTRUCTION_SUFFIX_TOOLS},
         "S2": {"text": "Do another thing." + tasks.INSTRUCTION_SUFFIX_TOOLS}}


def write(tmp_path, extra=EXTRA, rows=None):
    ep = tmp_path / "extra.json"
    ep.write_text(json.dumps(extra))
    sp = tmp_path / "s.jsonl"
    rows = rows or [{"task_id": "lcbhard_0", "prompt_id": "S1", "k": 0},
                    {"task_id": "lcbhard_1", "prompt_id": "S2", "k": 0},
                    {"task_id": "lcbhard_1", "prompt_id": "S1", "k": 1}]
    sp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ep, sp


def test_flag_absent(tmp_path):
    assert rr._registry() == {}
    _, sp = write(tmp_path)
    with pytest.raises(SystemExit, match="unknown prompt id 'S1'"):
        rr._load_schedule(sp)
    args = make_args(tmp_path, schedule=sp)
    assert args.registry_extra is None


def test_entries_and_hashes(tmp_path):
    ep, _ = write(tmp_path)
    reg = rr._registry(ep)
    assert set(reg) == {"S1", "S2"}
    assert reg["S1"]["sha256"] == hashlib.sha256(EXTRA["S1"]["text"].encode()).hexdigest()
    ep.write_text(json.dumps({"S1": {"set": "x"}}))
    with pytest.raises(SystemExit, match="needs a text"):
        rr._registry(ep)


def test_requires_schedule(tmp_path):
    ep, _ = write(tmp_path)
    with pytest.raises(SystemExit, match="requires --schedule"):
        rr.run(make_args(tmp_path, registry_extra=ep))


def test_run_resolves_ids_and_records_sha(tmp_path):
    ep, sp = write(tmp_path)
    seen = {}

    def run_one(task, *, rollout_id, seed, client, config, meta):
        seen[rollout_id] = (task.instruction_prompt, seed, meta.get("prompt_sha256"))
        t = fake_traj(task, rollout_id, seed, meta, config)
        t.meta.update({k: meta[k] for k in ("prompt_id", "prompt_direction", "prompt_sha256")})
        return t

    args = make_args(tmp_path, schedule=sp, registry_extra=ep)
    assert rr.run(args, run_one=run_one) == 0
    assert set(seen) == {"lcbhard_0__conflicting__S1__r0", "lcbhard_1__conflicting__S2__r0",
                         "lcbhard_1__conflicting__S1__r1"}
    assert seen["lcbhard_0__conflicting__S1__r0"][0] == EXTRA["S1"]["text"]
    assert seen["lcbhard_1__conflicting__S2__r0"][2] == hashlib.sha256(EXTRA["S2"]["text"].encode()).hexdigest()
    assert seen["lcbhard_0__conflicting__S1__r0"][1] == rr.rollout_seed(args.base_seed, "lcbhard_0:S1", 0)
    ps = json.loads((args.out / "manifest.json").read_text())["prompt_schedule"]
    assert ps["registry_extra"]["sha256"] == hashlib.sha256(ep.read_bytes()).hexdigest()
    seen.clear()
    assert rr.run(args, run_one=run_one) == 0 and not seen          # resume: nothing left
    # changing the prompt file refuses resume
    ep.write_text(json.dumps({**EXTRA, "S3": EXTRA["S1"]}))
    with pytest.raises(SystemExit, match="schedule"):
        rr.run(args, run_one=run_one)


def test_shipped_t1_instruction():
    root = pathlib.Path(__file__).resolve().parents[1]
    p = root / "evals/reward_hacking/data/lcb_instruction_T1.json"
    d = json.loads(p.read_text())
    assert list(d) == ["T1"]
    e = d["T1"]
    assert e["text"].endswith(tasks.INSTRUCTION_SUFFIX_TOOLS)
    assert e["sha256"] == hashlib.sha256(e["text"].encode()).hexdigest()
    assert (root / "prompts/eq2_agent/lcb_T1_instruction.txt").read_text() == e["text"]
    sched = root / "evals/reward_hacking/data/lcb_conflicting_schedules/full_T1_test.jsonl"
    assert len(rr._load_schedule(sched, p)) == 105
