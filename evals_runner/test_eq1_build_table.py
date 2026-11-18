"""Tests of eq1_build_table.py: run outputs rebuilt from data/eq1 give the same printed table as reproduce/eq1_table.py."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evals.reward_hacking import monitors as m
from evals_runner import eq1_build_table as B
from reproduce import eq1_table

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/eq1"
DIGEST = "sha256:" + "ab" * 32
SHA = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def wjl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_run(tmp: Path) -> dict:
    """Output folders of a GPU run, in the layout the launchers write, holding the numbers of data/eq1."""
    stage2 = D / "generation/stage2_list.json"
    sel = json.loads(stage2.read_text())
    stat = eq1_table.make_stat(sel["positions"])
    gen = tmp / "run/generate_calls_stage2"
    calls = jl(D / "generation/calls.jsonl")
    wjl(gen / "calls.jsonl", calls)
    wjl(gen / "setup.jsonl", [json.loads((D / "generation/setup.json").read_text())])
    (gen / "outputs.jsonl").write_text("{}\n")
    (gen / "run_config.json").write_text(json.dumps({"config_sha256": "0"}))
    a0 = {r["run_key"]: r["elapsed_s"] for r in calls if r["arm"] == "A0"}

    startup = json.loads((D / "monitors/startup.json").read_text())["seconds_launch_to_health_ok"]
    judges, items = tmp / "run/judges", {}
    for name in B.MONITORS:
        rows = jl(D / "monitors" / name / "rows.jsonl")
        for r in rows:
            items[r["item"]] = {"user_sha256": SHA(r["run_key"]), "run_key": r["run_key"]}
            r.update(fitted=False, evidence_shortened=False, user_sha256=SHA(r["run_key"]), original_user_sha256=SHA(r["run_key"]),
                     start_utc="2026-01-01T00:00:01Z")
        cfg = {"outputs_sha256": B.sha(gen / "outputs.jsonl"), "stage2_list_sha256": B.sha(stage2), "instruction": "early",
               "instruction_sha256": SHA(m.EARLY_INSTRUCTION), "histories_sha256": sel["histories_sha256"],
               "positions": [{k: p[k] for k in ("history_id", "task", "weight", "run_key")} for p in sel["positions"]],
               "hardware": {"nvidia_smi_gpu_name_memory_total_mib_driver_power_limit_w": "GPU"}, "code_sha256": "c", "monitor_config": {}}
        d = judges / name
        wjl(d / "rows.jsonl", rows)
        wjl(d / "replies.jsonl", [{"item": r["item"], "attempt": 1, "text": "SCORE: 0"} for r in rows])
        (d / "run_config.json").write_text(json.dumps({"config": cfg, "config_sha256": B.objsha(cfg), "written_utc": "2026-01-01T00:00:00Z"}))
        secs = {r["run_key"]: r["seconds"] for r in rows}
        (d / "summary.json").write_text(json.dumps({"views": {"history_black_box": {
            "tap_s_per_call": stat(secs), "composed": {"with_tap_s_per_call": stat({k: a0[k] + v for k, v in secs.items()})}}}}))
        wjl(d / "sessions.jsonl", [{"server_info_session": {"seconds_launch_to_health_ok": startup[name]}}])
    inputs = judges / "inputs/eq1_judge_saved_calls_inputs.json.gz"
    inputs.parent.mkdir(parents=True)
    with gzip.open(inputs, "wt") as f:
        json.dump({"items": [items[i] for i in range(len(items))]}, f)

    tap = json.loads((D / "tap_work.json").read_text())
    summary = tmp / "run/tap_work/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"weighted_mean_generated_tokens": tap["weighted_mean_generated_tokens"], "metrics": {
        "alone_net_ms": {"weighted_mean_over_positions": tap["tap_work_ms"]},
        "hook_cpu_net_ms": {"weighted_mean_over_positions": tap["hook_calls_ms"]},
        "end_call_ms": {"weighted_mean_over_positions": tap["end_of_call_scoring_ms"]}}}))

    # the first session of each configuration is the one after eviction; the later ones are the Certification column
    sessions, seen, reps = iter(jl(D / "certification/sessions.jsonl")), set(), []
    for r in jl(D / "certification/attestation.jsonl"):
        first = r["config"] not in seen
        seen.add(r["config"])
        overhead = 99.0 if first else next(sessions)
        if not first:
            assert overhead["config"] == r["config"]
            overhead = overhead["certification_s"]
        reps.append({**r, "cache_state": "evicted" if first else "warm", "protocol_overhead_s": overhead})
    cert = tmp / "run/certification/run"
    wjl(cert / "reps.jsonl", reps)
    timings = {"attestation_mode": B.ATTESTATION_MODE, "policy_enforced": True, "policy_digest": DIGEST, "gpu_attestation_verified": True,
               "workload_attested": False, "cpu_attested": False, "full_certification": False, "key_cache_policy": "cold"}
    (cert / "cert_timings.json").write_text(json.dumps(timings))
    return {"args": ["--gen-dir", str(tmp / "run"), "--inputs", str(inputs), "--monitors", str(judges), "--tap-work", str(summary),
                     "--cert-dir", str(cert), "--policy-digest", DIGEST, "--out-dir", str(tmp / "table")],
            "timings": timings, "cert": cert}


def test_builder_prints_the_same_table_as_eq1_table(tmp_path, capsys):
    run = make_run(tmp_path)
    eq1_table.main(["--data", str(D)])
    want = capsys.readouterr().out
    assert B.main(run["args"] + ["--json", str(tmp_path / "rows.json")]) == 0
    assert capsys.readouterr().out == want
    assert json.loads((tmp_path / "rows.json").read_text())[0]["tap"] == "Probe"
    assert (tmp_path / "table/verification.json").exists()


@pytest.mark.parametrize("change, message", [({"attestation_mode": "stub"}, "attestation mode"), ({"policy_digest": "sha256:other"}, "another GPU policy"),
                                             ({"workload_attested": True}, None)])
def test_builder_refuses_other_attestation_modes_and_policies(tmp_path, change, message):
    run = make_run(tmp_path)
    (run["cert"] / "cert_timings.json").write_text(json.dumps({**run["timings"], **change}))
    with pytest.raises(AssertionError, match=message):
        B.main(run["args"])


def test_builder_refuses_a_changed_reply(tmp_path):
    run = make_run(tmp_path)
    replies = Path(run["args"][5]) / "gemma4-e4b/replies.jsonl"
    rows = jl(replies)
    rows[3]["text"] = "no score here"
    wjl(replies, rows)
    with pytest.raises(AssertionError):
        B.main(run["args"])
    shutil.rmtree(tmp_path / "run")
