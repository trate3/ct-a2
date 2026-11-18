"""rh-rollouts driver: concurrency, resumability, seeds, crash isolation, manifest flags.

No model server or Docker: the server client and sandbox calls are faked, and
`run(..., run_one=fake)` replaces the agent loop.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time

import pytest

from evals.reward_hacking import llm, tasks
from evals.reward_hacking import trajectory as tj
from evals_runner import rh_rollouts as rr

TASKS = ["lcbhard_0", "lcbhard_1", "lcbhard_2", "lcbhard_3"]


class FakeClient:
    def __init__(self, url=None, timeout=None):
        pass

    def props(self):
        return {"chat_template": llm.TEMPLATE_PATH.read_text(), "total_slots": 4,
                "default_generation_settings": {"n_ctx": 4096}}


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(rr.llm, "CompletionClient", FakeClient)
    monkeypatch.setattr(rr.sandbox, "ensure_image", lambda *a, **k: None)
    monkeypatch.setattr(rr.sandbox, "sandbox_tools_binary", lambda *a, **k: None)
    monkeypatch.setattr(rr.sandbox, "cleanup_leftovers", lambda *a, **k: 0)
    monkeypatch.setattr(rr.tasks, "load_tasks", lambda variant="conflicting", test_sentence="nomod", instruction_variant="default": [
        tasks.Task(task_id=t, variant=variant, prompt="", test="", original_test="", entry_point="f",
                   instruction_prompt=tasks.instruction_prompt(test_sentence, instruction_variant)) for t in TASKS])
    monkeypatch.setattr(rr.tasks, "load_splits", lambda: {t: "train" for t in TASKS})


def make_args(tmp_path, **kw):
    p = argparse.ArgumentParser()
    rr.configure_parser(p)
    args = p.parse_args(["--tasks", ",".join(TASKS), "--out", str(tmp_path / "out"), "--rollouts", "3"])
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def fake_traj(task, rollout_id, seed, meta, config):
    t = tj.Trajectory(meta={"task_id": task.task_id, "variant": task.variant, "rollout_id": rollout_id,
                            "seed": seed, "partition": meta["partition"], "config": config.as_dict()})
    t.add(tj.ROLLOUT_END, {"status": "ok", "reason": "", "label": None, "timing": {}, "wall_seconds": 0.0})
    return t


class Recorder:
    """Fake run_rollout: records (rollout_id, seed, config) and concurrency."""

    def __init__(self, delay=0.0, crash=()):
        self.delay, self.crash = delay, set(crash)
        self.calls, self.configs, self.order = {}, [], []
        self.live = self.peak = 0
        self.lock = threading.Lock()

    def __call__(self, task, *, rollout_id, seed, client, config, meta):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.calls[rollout_id] = seed
            self.order.append(rollout_id)
            self.configs.append(config)
        try:
            time.sleep(self.delay)
            if rollout_id in self.crash:
                raise RuntimeError("boom")
            return fake_traj(task, rollout_id, seed, meta, config)
        finally:
            with self.lock:
                self.live -= 1


def rows(out):
    return [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]


def test_concurrency_bounded_by_workers(tmp_path):
    for n in (1, 3):
        rec = Recorder(delay=0.05)
        assert rr.run(make_args(tmp_path / str(n), workers=n), run_one=rec) == 0
        assert rec.peak == n
        assert len(rec.calls) == 12


def test_rollout_major_submission_order(tmp_path):
    rec = Recorder()
    rr.run(make_args(tmp_path, workers=1), run_one=rec)
    assert rec.order[:4] == [f"{t}__conflicting__r0" for t in TASKS]
    assert rec.order[4:8] == [f"{t}__conflicting__r1" for t in TASKS]


def test_seeds_independent_of_workers(tmp_path):
    a, b = Recorder(delay=0.01), Recorder(delay=0.01)
    rr.run(make_args(tmp_path / "a", workers=1), run_one=a)
    rr.run(make_args(tmp_path / "b", workers=4), run_one=b)
    assert a.calls == b.calls
    assert a.calls["lcbhard_1__conflicting__r2"] == rr.rollout_seed(20261008, "lcbhard_1", 2)
    assert len(set(a.calls.values())) == 12


def test_resume_skips_finished(tmp_path):
    args = make_args(tmp_path, workers=2)
    rr.run(args, run_one=Recorder())
    first = rows(args.out)
    rec = Recorder()
    assert rr.run(args, run_one=rec) == 0
    assert rec.calls == {} and rows(args.out) == first
    # one trajectory removed: only it is redone
    (args.out / "trajectories" / "lcbhard_2__conflicting__r1.json").unlink()
    rec = Recorder()
    rr.run(args, run_one=rec)
    assert list(rec.calls) == ["lcbhard_2__conflicting__r1"]
    assert not list((args.out / "trajectories").glob("*.tmp"))


def test_crash_isolated_recorded_and_retried(tmp_path):
    args = make_args(tmp_path, workers=3)
    bad = "lcbhard_1__conflicting__r0"
    assert rr.run(args, run_one=Recorder(crash=[bad])) == 1
    assert len(rows(args.out)) == 11
    err = [json.loads(line) for line in (args.out / "errors.jsonl").read_text().splitlines()]
    assert [e["rollout_id"] for e in err] == [bad] and "boom" in err[0]["error"]
    assert not (args.out / "trajectories" / f"{bad}.json").exists()
    rec = Recorder()
    assert rr.run(args, run_one=rec) == 0
    assert list(rec.calls) == [bad]


def test_runs_jsonl_lines_intact_under_contention(tmp_path):
    args = make_args(tmp_path, workers=8, rollouts=25)
    rr.run(args, run_one=Recorder(delay=0.001))
    got = rows(args.out)       # json.loads on every line: no interleaving
    assert len(got) == 100 and len({r["rollout_id"] for r in got}) == 100


def test_flags_reach_config_and_manifest(tmp_path):
    args = make_args(tmp_path, max_attempts=7, message_limit=55, test_sentence="default")
    rec = Recorder()
    rr.run(args, run_one=rec)
    assert rec.configs[0].max_attempts == 7 and rec.configs[0].message_limit == 55
    man = json.loads((args.out / "manifest.json").read_text())
    assert man["rollout_config"]["max_attempts"] == 7
    assert man["benchmark"]["test_sentence"] == "default"
    assert man["benchmark"]["instruction_prompt"] == tasks.instruction_prompt("default")
    assert "DO NOT MODIFY" in man["benchmark"]["instruction_prompt"]


def test_defaults_keep_pilot_config(tmp_path):
    args = make_args(tmp_path)
    assert args.test_sentence == "nomod" and args.max_attempts is None and args.workers == 1
    rec = Recorder()
    rr.run(args, run_one=rec)
    assert rec.configs[0].max_attempts == 3 and rec.configs[0].message_limit == 30
    man = json.loads((args.out / "manifest.json").read_text())
    assert man["benchmark"]["instruction_prompt"] == tasks.INSTRUCTION_PROMPT
    assert man["benchmark"]["test_sentence"] == "nomod"


@pytest.mark.parametrize("change", [{"max_attempts": 5}, {"test_sentence": "default"}, {"message_limit": 99}])
def test_changed_flag_refuses_resume(tmp_path, change):
    rr.run(make_args(tmp_path), run_one=Recorder())
    with pytest.raises(SystemExit, match="changed since"):
        rr.run(make_args(tmp_path, **change), run_one=Recorder())


def test_old_manifest_without_test_sentence_still_resumes(tmp_path):
    args = make_args(tmp_path)
    rr.run(args, run_one=Recorder())
    mp = args.out / "manifest.json"
    man = json.loads(mp.read_text())
    del man["benchmark"]["test_sentence"]
    mp.write_text(json.dumps(man))
    assert rr.run(args, run_one=Recorder()) == 0
    with pytest.raises(SystemExit):
        rr.run(make_args(tmp_path, test_sentence="default"), run_one=Recorder())


def test_strict_instruction_variant_in_manifest_and_tasks(tmp_path):
    args = make_args(tmp_path, instruction_variant="strict", test_sentence="default")
    rec = Recorder()
    rr.run(args, run_one=rec)
    man = json.loads((args.out / "manifest.json").read_text())
    assert man["benchmark"]["instruction_variant"] == "strict"
    assert man["benchmark"]["instruction_prompt"] == tasks.instruction_prompt("default", "strict")
    assert "STOP, identify them" in man["benchmark"]["instruction_prompt"]


def test_default_manifest_records_default_variant(tmp_path):
    args = make_args(tmp_path)
    assert args.instruction_variant == "default"
    rr.run(args, run_one=Recorder())
    man = json.loads((args.out / "manifest.json").read_text())
    assert man["benchmark"]["instruction_variant"] == "default"


def test_strict_requires_default_test_sentence(tmp_path):
    with pytest.raises(SystemExit, match="--test-sentence default"):
        rr.run(make_args(tmp_path, instruction_variant="strict"), run_one=Recorder())


def test_changed_instruction_variant_refuses_resume(tmp_path):
    rr.run(make_args(tmp_path, test_sentence="default"), run_one=Recorder())
    with pytest.raises(SystemExit, match="changed since"):
        rr.run(make_args(tmp_path, test_sentence="default", instruction_variant="strict"), run_one=Recorder())


def test_old_manifest_without_instruction_variant_still_resumes(tmp_path):
    args = make_args(tmp_path)
    rr.run(args, run_one=Recorder())
    mp = args.out / "manifest.json"
    man = json.loads(mp.read_text())
    del man["benchmark"]["instruction_variant"]
    mp.write_text(json.dumps(man))
    assert rr.run(args, run_one=Recorder()) == 0


# ---- engines, interruption, partial files ------------------------------------------------------

class FakeVLLM:
    engine = "vllm"

    def __init__(self, url=None, timeout=None):
        pass

    def identity(self):
        return {"engine": "vllm", "engine_version": "0.31.0", "build_info": "vllm 0.31.0", "model_path": "/m",
                "model_alias": "q", "model_ftype": None, "total_slots": None, "n_ctx": 196608,
                "server_template_sha256": None}


def test_engine_flag_default_and_manifest(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    assert args.engine == "llamacpp"
    rr.run(args, run_one=Recorder())
    tm = json.loads((args.out / "manifest.json").read_text())["target_model"]
    assert tm["engine"] == "llamacpp" and tm["engine_version"] is None


def test_vllm_manifest_skips_template_check_and_records_version(tmp_path, monkeypatch):
    monkeypatch.setattr(rr.llm, "make_client", lambda engine, url, **kw: FakeVLLM())
    args = make_args(tmp_path, engine="vllm")
    assert rr.run(args, run_one=Recorder()) == 0
    tm = json.loads((args.out / "manifest.json").read_text())["target_model"]
    assert (tm["engine"], tm["engine_version"], tm["model_alias"]) == ("vllm", "0.31.0", "q")


def test_engine_change_refuses_resume(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    rr.run(args, run_one=Recorder())
    monkeypatch.setattr(rr.llm, "make_client", lambda engine, url, **kw: FakeVLLM())
    with pytest.raises(SystemExit, match="engine changed"):
        rr.run(make_args(tmp_path, engine="vllm"), run_one=Recorder())
    # an output dir from before the field existed counts as llamacpp
    man = json.loads((args.out / "manifest.json").read_text())
    del man["target_model"]["engine"]
    (args.out / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="engine changed"):
        rr.run(make_args(tmp_path, engine="vllm"), run_one=Recorder())


def test_partial_files_cleaned_on_start(tmp_path):
    args = make_args(tmp_path)
    td = args.out / "trajectories"
    td.mkdir(parents=True)
    (td / "lcbhard_0__conflicting__r0.json.tmp").write_text("{")
    (td / "lcbhard_1__conflicting__r0.snapshots.pkl").write_bytes(b"x")        # trajectory never written
    (td / "lcbhard_2__conflicting__r0.snapshots.pkl").write_bytes(b"x")
    (td / "lcbhard_2__conflicting__r0.json").write_text("{}")                  # finished: keep
    rr.run(args, run_one=Recorder())
    names = {p.name for p in td.iterdir()}
    assert not any(n.endswith(".tmp") for n in names)
    assert "lcbhard_1__conflicting__r0.snapshots.pkl" not in names
    assert "lcbhard_2__conflicting__r0.snapshots.pkl" in names


def test_signal_stops_gracefully_and_resumes(tmp_path, monkeypatch):
    import os
    import signal
    exits = []
    monkeypatch.setattr(rr, "_hard_exit", exits.append)
    gate = threading.Event()
    args = make_args(tmp_path, workers=2, grace_seconds=0.3)

    def blocking(task, *, rollout_id, seed, client, config, meta):
        gate.wait(5)
        return fake_traj(task, rollout_id, seed, meta, config)

    threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGUSR1)).start()
    try:
        assert rr.run(args, run_one=blocking) == rr.EXIT_INTERRUPTED
    finally:
        gate.set()
    assert exits == [rr.EXIT_INTERRUPTED]                    # in-flight rollouts were abandoned
    time.sleep(0.3)                                          # abandoned threads finish writing at most their own files
    done = {p.stem for p in (args.out / "trajectories").glob("*.json")}
    rec = Recorder()
    assert rr.run(args, run_one=rec) == 0                   # handlers restored; resume runs the rest
    assert len(done | set(rec.calls)) == 12 and not (done & set(rec.calls))
    assert signal.getsignal(signal.SIGUSR1) in (signal.SIG_DFL, signal.default_int_handler)


# ---- --retry-infra --------------------------------------------------------------------------------

class InfraFlaky(Recorder):
    """Rollouts in `fail` end infra_error on their first `times` executions."""

    def __init__(self, fail, times=1, **kw):
        super().__init__(**kw)
        self.fail, self.times, self.runs, self.seeds = set(fail), times, {}, {}

    def __call__(self, task, *, rollout_id, seed, client, config, meta):
        with self.lock:
            n = self.runs[rollout_id] = self.runs.get(rollout_id, 0) + 1
            self.seeds.setdefault(rollout_id, []).append(seed)
        t = super().__call__(task, rollout_id=rollout_id, seed=seed, client=client, config=config, meta=meta)
        if rollout_id in self.fail and n <= self.times:
            t.events[-1].data["status"] = "infra_error"
        return t


BAD = "lcbhard_1__conflicting__r0"


def test_retry_infra_default_is_off(tmp_path):
    args = make_args(tmp_path)
    assert args.retry_infra == 0
    rec = InfraFlaky([BAD])
    assert rr.run(args, run_one=rec) == 0
    assert rec.runs[BAD] == 1 and not (args.out / "trajectories" / "infra_attempts").exists()


def test_retry_infra_moves_aside_and_reruns_same_seed(tmp_path):
    args = make_args(tmp_path, workers=3, retry_infra=2)
    rec = InfraFlaky([BAD], times=1)
    assert rr.run(args, run_one=rec) == 0
    assert rec.runs[BAD] == 2 and rec.seeds[BAD] == [rr.rollout_seed(20261008, "lcbhard_1", 0)] * 2
    td = args.out / "trajectories"
    kept = td / "infra_attempts" / f"{BAD}.attempt1.json"
    assert kept.exists() and json.loads(kept.read_text())["events"][-1]["data"]["status"] == "infra_error"
    final = json.loads((td / f"{BAD}.json").read_text())
    assert final["events"][-1]["data"]["status"] == "ok" and final["meta"]["rollout_id"] == BAD
    mine = [r for r in rows(args.out) if r["rollout_id"] == BAD]
    assert [r["status"] for r in mine] == ["infra_error", "ok"]       # latest row wins for readers
    assert len(rows(args.out)) == 13 and sum(r.get("rollout_id") == BAD for r in rows(args.out)) == 2


def test_retry_infra_caps_at_n_and_keeps_every_attempt(tmp_path):
    args = make_args(tmp_path, retry_infra=2)
    rec = InfraFlaky([BAD], times=99)
    assert rr.run(args, run_one=rec) == 0                 # an excluded rollout is not a crash
    assert rec.runs[BAD] == 3                             # original + 2 more
    d = args.out / "trajectories" / "infra_attempts"
    assert sorted(p.name for p in d.glob("*.json")) == [f"{BAD}.attempt1.json", f"{BAD}.attempt2.json"]
    td = args.out / "trajectories"
    assert json.loads((td / f"{BAD}.json").read_text())["events"][-1]["data"]["status"] == "infra_error"
    # a further invocation does not retry again (budget used), and does not move the final one aside
    rec2 = InfraFlaky([BAD], times=99)
    rr.run(args, run_one=rec2)
    assert rec2.calls == {} and len(list(d.glob("*.json"))) == 2


def test_retry_infra_on_resume_counts_prior_attempts(tmp_path):
    args = make_args(tmp_path, retry_infra=2)
    rr.run(args, run_one=InfraFlaky([BAD], times=99))     # 3 runs, budget exhausted
    # an infra_error left by a run WITHOUT the flag is retried by the next run that has it, from the count
    args2 = make_args(tmp_path / "b", retry_infra=0)
    rr.run(args2, run_one=InfraFlaky([BAD], times=1))
    args2.retry_infra = 2
    rec = InfraFlaky([BAD], times=0)
    rr.run(args2, run_one=rec)
    assert list(rec.calls) == [BAD] and rec.calls[BAD] == rr.rollout_seed(20261008, "lcbhard_1", 0)
    assert (args2.out / "trajectories" / "infra_attempts" / f"{BAD}.attempt1.json").exists()
    assert json.loads((args2.out / "trajectories" / f"{BAD}.json").read_text())["events"][-1]["data"]["status"] == "ok"


def test_retry_infra_cannot_change_manifest_or_ids(tmp_path):
    a, b = make_args(tmp_path / "a", retry_infra=0), make_args(tmp_path / "b", retry_infra=2)
    rr.run(a, run_one=Recorder())
    rr.run(b, run_one=InfraFlaky([BAD]))
    ma, mb = (json.loads((x.out / "manifest.json").read_text()) for x in (a, b))
    assert not [k for k in ("benchmark", "sandbox", "rollout_config", "selection") if ma[k] != mb[k]]
    rr.run(make_args(tmp_path / "b", retry_infra=0), run_one=Recorder())     # flag is not part of the manifest check
    assert {p.stem for p in (b.out / "trajectories").glob("*.json")} == \
           {p.stem for p in (a.out / "trajectories").glob("*.json")}


def test_retry_infra_not_rerun_after_stop(tmp_path, monkeypatch):
    """A stop between the failure and the retry leaves the infra_error trajectory in place (no data loss);
    the next start sets it aside and reruns it, counting that attempt."""
    import os
    import signal
    monkeypatch.setattr(rr, "_hard_exit", lambda c: None)
    args = make_args(tmp_path, workers=1, retry_infra=2, rollouts=1, grace_seconds=2)
    td = args.out / "trajectories"

    class StopAfterInfra(InfraFlaky):
        def __call__(self, *a, **kw):
            t = super().__call__(*a, **kw)
            if kw["rollout_id"] == BAD:
                os.kill(os.getpid(), signal.SIGUSR1)
                time.sleep(0.2)
            return t

    assert rr.run(args, run_one=StopAfterInfra([BAD], times=1)) == rr.EXIT_INTERRUPTED
    assert not (td / "infra_attempts").exists() or not list((td / "infra_attempts").glob("*.json"))
    assert json.loads((td / f"{BAD}.json").read_text())["events"][-1]["data"]["status"] == "infra_error"
    rec = InfraFlaky([BAD], times=0)
    assert rr.run(args, run_one=rec) == 0
    assert BAD in rec.calls and (td / "infra_attempts" / f"{BAD}.attempt1.json").exists()


def test_extend_from_reuses_continues_and_fills(tmp_path):
    """--extend-from: uncapped rollouts are copied, capped ones get an ExtendingClient, missing ones run fresh."""
    from evals.reward_hacking import extend

    def old_run(task, *, rollout_id, seed, client, config, meta):
        t = fake_traj(task, rollout_id, seed, meta, config)
        capped = task.task_id == "lcbhard_1"
        t.events.clear()
        t.add(tj.MODEL_CALL, {"prompt": "p", "generation": "g", "tokens": list(range(8 if capped else 3)),
                              "stop_type": "limit" if capped else "eos", "prompt_tokens": 1, "cached_tokens": 0,
                              "prefix_consistent": None})
        t.add(tj.ROLLOUT_END, {"status": "model_length" if capped else "attempts_exhausted", "reason": "",
                               "label": 0, "timing": {}, "wall_seconds": 0.0})
        return t

    old = make_args(tmp_path / "old", rollouts=2, max_tokens_per_call=8)
    old.out = tmp_path / "old"
    assert rr.run(old, run_one=old_run) == 0
    seen = {}

    def new_run(task, *, rollout_id, seed, client, config, meta):
        seen[rollout_id] = (type(client).__name__, meta.get("extended_from"), config.max_tokens_per_call)
        return fake_traj(task, rollout_id, seed, meta, config)

    new = make_args(tmp_path, rollouts=3, max_tokens_per_call=16, extend_from=old.out)
    assert rr.run(new, run_one=new_run) == 0
    # 4 tasks x 3 k: k<2 of lcbhard_1 extended (2), other k<2 reused (6), all k=2 fresh (4)
    assert sorted(r for r, v in seen.items() if v[0] == "ExtendingClient") == \
        ["lcbhard_1__conflicting__r0", "lcbhard_1__conflicting__r1"]
    assert sum(v[0] != "ExtendingClient" for v in seen.values()) == 4
    assert all(v[2] == 16 for v in seen.values())
    r = rows(new.out)
    assert sum("reused_from" in x for x in r) == 6 and len(r) == 12
    assert (new.out / "trajectories" / "lcbhard_0__conflicting__r0.json").read_text() == \
        (old.out / "trajectories" / "lcbhard_0__conflicting__r0.json").read_text()
    man = json.loads((new.out / "manifest.json").read_text())
    assert man["extended_from"]["old_rollout_config"]["max_tokens_per_call"] == 8
    assert extend.REUSE == "reuse"
    # rerun: nothing left to do, nothing duplicated
    seen.clear()
    assert rr.run(new, run_one=new_run) == 0 and not seen and len(rows(new.out)) == 12


def test_extend_from_refuses_other_config(tmp_path):
    old = make_args(tmp_path / "old", max_tokens_per_call=8)
    old.out = tmp_path / "old"
    rr.run(old, run_one=Recorder())
    with pytest.raises(SystemExit, match="larger cap|rollout config"):
        rr.run(make_args(tmp_path, max_tokens_per_call=16, message_limit=50, extend_from=old.out),
               run_one=Recorder())


def test_extend_below_k_samples_the_rest_fresh(tmp_path):
    old = make_args(tmp_path / "old", rollouts=3, max_tokens_per_call=8)
    old.out = tmp_path / "old"
    rr.run(old, run_one=Recorder())
    rec = Recorder()
    rr.run(make_args(tmp_path, rollouts=3, max_tokens_per_call=16, extend_from=old.out, extend_below_k=2),
           run_one=rec)
    assert sorted(rec.calls) == sorted(f"{t}__conflicting__r2" for t in TASKS)


def test_extend_from_accepts_legacy_qwen36_manifest_and_context_growth(tmp_path):
    old = make_args(tmp_path / "old", max_tokens_per_call=8)
    old.out = tmp_path / "old"
    rr.run(old, run_one=Recorder())
    man = json.loads((old.out / "manifest.json").read_text())
    man.pop("target_model_name", None); man.pop("chat_format", None)          # as written before the flags
    (old.out / "manifest.json").write_text(json.dumps(man))
    assert rr.run(make_args(tmp_path, max_tokens_per_call=0, max_context_tokens=240000, extend_from=old.out),
                  run_one=Recorder()) == 0


def test_schedule_runs_each_row_with_its_prompt(tmp_path):
    extra = tmp_path / "prompts.json"
    texts = {"T1": "First instruction." + tasks.INSTRUCTION_SUFFIX_TOOLS, "T2": "Second instruction." + tasks.INSTRUCTION_SUFFIX_TOOLS}
    extra.write_text(json.dumps({k: {"text": v} for k, v in texts.items()}))
    sched = tmp_path / "s.jsonl"
    sched_rows = [{"task_id": "lcbhard_0", "prompt_id": "T1", "k": 0}, {"task_id": "lcbhard_0", "prompt_id": "T2", "k": 0},
                  {"task_id": "lcbhard_1", "prompt_id": "T1", "k": 1}]
    sched.write_text("".join(json.dumps(r) + "\n" for r in sched_rows))
    seen = {}

    def run_one(task, *, rollout_id, seed, client, config, meta):
        seen[rollout_id] = (task.instruction_prompt, seed, meta.get("prompt_id"))
        t = fake_traj(task, rollout_id, seed, meta, config)
        t.meta.update({k: meta[k] for k in ("prompt_id", "prompt_direction", "prompt_sha256")})
        return t

    args = make_args(tmp_path, schedule=sched, registry_extra=extra)
    assert rr.run(args, run_one=run_one) == 0
    assert set(seen) == {"lcbhard_0__conflicting__T1__r0", "lcbhard_0__conflicting__T2__r0",
                         "lcbhard_1__conflicting__T1__r1"}
    text, seed, pid = seen["lcbhard_0__conflicting__T2__r0"]
    assert text == texts["T2"] and pid == "T2"
    assert seed == rr.rollout_seed(args.base_seed, "lcbhard_0:T2", 0) != seen["lcbhard_0__conflicting__T1__r0"][1]
    assert {r["prompt_id"] for r in rows(args.out)} == {"T1", "T2"}
    seen.clear()
    assert rr.run(args, run_one=run_one) == 0 and not seen          # resume: nothing left
    sched.write_text(sched.read_text() + json.dumps({"task_id": "lcbhard_2", "prompt_id": "T2", "k": 0}) + "\n")
    with pytest.raises(SystemExit, match="schedule"):
        rr.run(args, run_one=run_one)
