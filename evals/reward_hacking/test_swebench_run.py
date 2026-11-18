import json

import pytest

from evals.reward_hacking.scripts import swebench_run as run


def _pre(status="ok", oracle=True):
    return {"status": status, "original_oracle": {"pass": oracle, "kind": "pass" if oracle else "test_failure"}}


def test_build_schedule_filters_on_preflight_and_oracle_and_pin():
    parts = {"train": ["a", "b", "c", "d"], "val": ["e", "f"]}
    pre = {"a": _pre(), "b": _pre("malformed"), "c": _pre(oracle=False), "d": {"status": "ok"}, "e": _pre()}
    pins = {i: {"verified": True} for i in "abcdef"}
    pins["e"] = {"verified": False}
    rows, excl = run.build_schedule(parts, ["train", "val"], [1, 0], pre, pins)
    assert rows == [{"id": "a", "k": 0}, {"id": "a", "k": 1}]
    assert set(excl) == {"b", "c", "d", "e", "f"}
    assert "malformed" in excl["b"] and "oracle" in excl["c"] and "original_oracle" in excl["d"]
    assert "pin" in excl["e"] and "no preflight" in excl["f"]
    rows, _ = run.build_schedule(parts, ["val"], [0], pre, {**pins, "e": {"verified": True}})
    assert rows == [{"id": "e", "k": 0}]


def test_write_schedule_and_load_roundtrip(tmp_path):
    rows = [{"id": "x", "k": 0}, {"id": "y", "k": 0}]
    m = run.write_schedule(tmp_path / "s.jsonl", rows, {"excluded": {}})
    assert run.load_schedule(tmp_path / "s.jsonl") == rows
    assert m["n_rows"] == 2 and len(m["schedule_sha256"]) == 64
    assert json.loads((tmp_path / "s.jsonl.MANIFEST.json").read_text())["schedule_sha256"] == m["schedule_sha256"]
    (tmp_path / "d.jsonl").write_text('{"id":"x","k":0}\n{"id":"x","k":0}\n')
    with pytest.raises(ValueError, match="duplicate"):
        run.load_schedule(tmp_path / "d.jsonl")


def test_sharding_partitions_sorted_by_k_then_id():
    rows = [{"id": i, "k": k} for k in (1, 0) for i in ("c", "a", "b")]
    shards = [run.shard_rows(rows, i, 4) for i in range(4)]
    flat = [(r["k"], r["id"]) for s in shards for r in s]
    assert sorted(flat) == sorted((r["k"], r["id"]) for r in rows) and len(flat) == 6
    assert [(r["k"], r["id"]) for r in shards[0]] == [(0, "a"), (1, "b")]
    assert run.parse_shard("3/8") == (3, 8)
    for bad in ("8/8", "x", "-1/2", "0/0"):
        with pytest.raises(ValueError):
            run.parse_shard(bad)


def test_seeds_stable_and_k_dependent():
    from evals.reward_hacking.scripts import swebench_pilot as pilot
    assert run.stable_seed("a__b-1", 0) == pilot._stable_seed("a__b-1")
    assert run.stable_seed("a__b-1", 1) != run.stable_seed("a__b-1", 0)
    assert run.stable_seed("a__b-1", 1) == run.stable_seed("a__b-1", 1)
    assert run.stable_seed("a__b-1", 1) != run.stable_seed("a__b-2", 1)
    assert 0 <= run.stable_seed("a", 7) < 2**31


def test_plan_items_resume_and_infra_retry(tmp_path):
    rows = [{"id": "a", "k": 0}, {"id": "a", "k": 1}, {"id": "b", "k": 0}]
    fake = {"a__r0": {"instance_id": "a", "kind": "pass", "label": 1, "eval_log": "l"}}
    seen = []

    def recover(path, iid):
        seen.append(path.name)
        return fake.get(path.name)

    todo, done = run.plan_items(rows, tmp_path, recover=recover)
    assert seen == ["a__r0", "a__r1", "b__r0"]
    assert [r["id"] + str(r["k"]) for r in todo] == ["a1", "b0"] and [r["id"] for r, _ in done] == ["a"]
    (tmp_path / "failures-0of2.jsonl").write_text(json.dumps({"instance_id": "b", "k": 0}) + '\ntorn{')
    todo, _ = run.plan_items(rows, tmp_path, recover=recover)  # infra failures retried by default
    assert len(todo) == 2
    todo, _ = run.plan_items(rows, tmp_path, retry_infra=False, recover=recover)
    assert [(r["id"], r["k"]) for r in todo] == [("a", 1)]


def test_manifest_written_once_and_mismatch_refused(tmp_path):
    cfg = run.run_config("f" * 64)
    d1 = run.write_manifest_once(tmp_path / "manifest.json", cfg)
    assert run.write_manifest_once(tmp_path / "manifest.json", cfg) == d1
    with pytest.raises(ValueError, match="mismatch"):
        run.write_manifest_once(tmp_path / "manifest.json", run.run_config("e" * 64))
    assert cfg["extra_body"] == {"top_k": 20, "return_token_ids": True}


def test_append_jsonl_and_records(tmp_path):
    run._append_jsonl(tmp_path / "runs-0of2.jsonl", {"instance_id": "a", "k": 0, "status": "completed"})
    run._append_jsonl(tmp_path / "runs-1of2.jsonl", {"instance_id": "b", "k": 0, "status": "infra_error"})
    assert {r["instance_id"] for r in run._records(tmp_path, "runs")} == {"a", "b"}


def test_paper_matched_settings_enter_config_only_when_set(monkeypatch):
    from evals.reward_hacking.scripts import swebench_run as R
    base = R.run_config("x")
    assert base["max_attempts"] == 3 and "reasoning_effort" not in base
    assert "chat_template_kwargs" not in base["extra_body"]          # earlier default runs keep their config hash
    monkeypatch.setattr(R, "REASONING_EFFORT", "medium")
    monkeypatch.setattr(R, "RUN_MAX_ATTEMPTS", 10)
    monkeypatch.setattr(R, "EXTRA_BODY", dict(R.VLLM_EXTRA_BODY, chat_template_kwargs={"reasoning_effort": "medium"}))
    cfg = R.run_config("x")
    assert cfg["max_attempts"] == 10 and cfg["reasoning_effort"] == "medium"
    assert cfg["extra_body"]["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert cfg["extra_body"]["top_k"] == 20 and cfg["extra_body"]["return_token_ids"] is True
