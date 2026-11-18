"""--registry-extra: extra candidate prompts merged into the frozen registry for --schedule runs."""
from __future__ import annotations

import hashlib
import json

import pytest

from evals.reward_hacking import prompt_registry, tasks
from evals_runner import rh_rollouts as rr
from evals_runner.test_rh_rollouts import fakes, fake_traj, make_args  # noqa: F401  (autouse fixture)

EXTRA = {"S1": {"direction": "X", "set": "sweep", "text": "Do the thing." + tasks.INSTRUCTION_SUFFIX_TOOLS},
         "S2": {"direction": "X", "set": "sweep", "text": "Do another thing." + tasks.INSTRUCTION_SUFFIX_TOOLS}}


def write(tmp_path, extra=EXTRA, rows=None):
    ep = tmp_path / "extra.json"
    ep.write_text(json.dumps(extra))
    sp = tmp_path / "s.jsonl"
    rows = rows or [{"task_id": "lcbhard_0", "prompt_id": "S1", "k": 0},
                    {"task_id": "lcbhard_1", "prompt_id": "S2", "k": 0},
                    {"task_id": "lcbhard_1", "prompt_id": "C", "k": 0}]
    sp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ep, sp


def test_flag_absent_unchanged(tmp_path):
    assert rr._registry() == prompt_registry.registry()
    _, sp = write(tmp_path)
    with pytest.raises(SystemExit, match="unknown prompt id 'S1'"):
        rr._load_schedule(sp)
    args = make_args(tmp_path, schedule=sp)
    assert args.registry_extra is None


def test_extra_ids_merge_and_frozen_untouched(tmp_path):
    ep, _ = write(tmp_path)
    before = prompt_registry.registry_digest()
    reg = rr._registry(ep)
    assert set(reg) == set(prompt_registry.registry()) | {"S1", "S2"}
    assert reg["S1"]["sha256"] == hashlib.sha256(EXTRA["S1"]["text"].encode()).hexdigest()
    assert reg["S1"]["direction"] == "X" and reg["C"] == prompt_registry.registry()["C"]
    assert prompt_registry.registry_digest() == before


def test_collision_raises(tmp_path):
    ep, sp = write(tmp_path, extra={"C": EXTRA["S1"]})
    with pytest.raises(SystemExit, match="collide"):
        rr._registry(ep)
    with pytest.raises(SystemExit, match="collide"):
        rr.run(make_args(tmp_path, schedule=sp, registry_extra=ep), run_one=None)


def test_requires_schedule(tmp_path):
    ep, _ = write(tmp_path)
    with pytest.raises(SystemExit, match="requires --schedule"):
        rr.run(make_args(tmp_path, registry_extra=ep))


def test_run_resolves_s_ids_and_records_sha(tmp_path):
    ep, sp = write(tmp_path)
    seen = {}

    def run_one(task, *, rollout_id, seed, client, config, meta):
        seen[rollout_id] = (task.instruction_prompt, meta.get("prompt_direction"), meta.get("prompt_sha256"))
        t = fake_traj(task, rollout_id, seed, meta, config)
        t.meta.update({k: meta[k] for k in ("prompt_id", "prompt_direction", "prompt_sha256")})
        return t

    args = make_args(tmp_path, schedule=sp, registry_extra=ep)
    assert rr.run(args, run_one=run_one) == 0
    assert set(seen) == {"lcbhard_0__conflicting__S1__r0", "lcbhard_1__conflicting__S2__r0",
                         "lcbhard_1__conflicting__C__r0"}
    assert seen["lcbhard_0__conflicting__S1__r0"][0] == EXTRA["S1"]["text"]
    assert seen["lcbhard_1__conflicting__S2__r0"][1:] == (
        "X", hashlib.sha256(EXTRA["S2"]["text"].encode()).hexdigest())
    assert seen["lcbhard_1__conflicting__C__r0"][0] == prompt_registry.registry()["C"]["text"]
    ps = json.loads((args.out / "manifest.json").read_text())["prompt_schedule"]
    assert ps["registry_extra"]["sha256"] == hashlib.sha256(ep.read_bytes()).hexdigest()
    assert ps["registry_sha256"] == prompt_registry.registry_digest()
    # changing the extra file refuses resume
    ep.write_text(json.dumps({**EXTRA, "S3": EXTRA["S1"]}))
    with pytest.raises(SystemExit, match="schedule"):
        rr.run(args, run_one=run_one)


def test_shipped_sweep_prompts():
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "evals/reward_hacking/data/a24/sweep_prompts.json"
    d = json.loads(p.read_text())
    assert list(d) == [f"S{i}" for i in range(1, 9)]
    for e in d.values():
        assert e["direction"] == "X" and e["set"] == "sweep" and e["text"].endswith(tasks.INSTRUCTION_SUFFIX_TOOLS)
        assert e["sha256"] == hashlib.sha256(e["text"].encode()).hexdigest()
    assert rr._registry(p).keys() >= set(d)
