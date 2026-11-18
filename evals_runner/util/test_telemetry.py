from __future__ import annotations

import json

from evals_runner.util.telemetry import (
    JsonlWriter,
    MachineInfo,
    ModelInfo,
    NullStageRecorder,
    RunInfo,
    RunRecord,
    StageRecorder,
    TapInfo,
    WorkloadInfo,
    canonical_digest,
    read_jsonl,
)


def _machine():
    return MachineInfo("n2d-standard-4", 4, 16.0, None, "us-central1", "SEV", "confidential-space-debug")


def _model():
    return ModelInfo("fake/model", "rev1", "bfloat16")


def _tap():
    return TapInfo("cot", monitor_model_id="fake/model")


def _workload():
    return WorkloadInfo("confidence", n_trials=8, k_max=1, max_new_tokens=64, prompt_template_id="v1")


def _run_info(**overrides):
    base = dict(arm="A2", seed=7, repeat_index=0, order="direct-first", warmup=False)
    base.update(overrides)
    return RunInfo(**base)


def test_config_digest_is_order_independent_and_stable():
    a = {"n_trials": 8, "tap": "cot"}
    b = {"tap": "cot", "n_trials": 8}
    assert canonical_digest(a) == canonical_digest(b)
    assert canonical_digest(a) != canonical_digest({"n_trials": 9, "tap": "cot"})


def test_run_record_fills_timestamp_and_commit_when_absent():
    record = RunRecord(
        run_id="r1", config_digest="deadbeef", machine=_machine(), model=_model(),
        tap=_tap(), workload=_workload(), run=_run_info(),
        outcome={"m": 0.0, "m_fp": 0.0, "aborted": False, "certificate_verified": True},
        totals={"model_load_seconds": 1.0, "peak_rss_bytes": 100, "peak_gpu_memory_bytes": None},
        bytes={"protocol_sent": 0, "protocol_received": 0, "artifact_hashed": 0, "certificate_size": 0},
    )
    assert record.timestamp
    assert record.repo_commit


def test_jsonl_writer_round_trips_run_and_stage_records(tmp_path):
    run_path = tmp_path / "runs.jsonl"
    stage_path = tmp_path / "stages.jsonl"
    run_sink = JsonlWriter(run_path)
    stage_sink = JsonlWriter(stage_path)

    record = RunRecord(
        run_id="r1", config_digest="deadbeef", machine=_machine(), model=_model(),
        tap=_tap(), workload=_workload(), run=_run_info(),
        outcome={"m": 0.0, "m_fp": 0.0, "aborted": False, "certificate_verified": True},
        totals={"model_load_seconds": 1.0, "peak_rss_bytes": 100, "peak_gpu_memory_bytes": None},
        bytes={"protocol_sent": 10, "protocol_received": 20, "artifact_hashed": 30, "certificate_size": 40},
    )
    run_sink.write(record)

    recorder = StageRecorder("r1", stage_sink)
    with recorder.stage("model_inference", trial_index=1):
        pass
    with recorder.stage("scoring", trial_index=1):
        pass

    loaded_runs = read_jsonl(run_path)
    loaded_stages = read_jsonl(stage_path)
    assert len(loaded_runs) == 1
    assert loaded_runs[0]["run_id"] == "r1"
    assert loaded_runs[0]["bytes"]["certificate_size"] == 40
    assert [s["stage"] for s in loaded_stages] == ["model_inference", "scoring"]
    assert all(s["run_id"] == "r1" for s in loaded_stages)
    assert all(s["wall_seconds"] >= 0 for s in loaded_stages)


def test_stage_recorder_without_sink_keeps_records_in_memory():
    recorder = StageRecorder("r1", sink=None)
    with recorder.stage("orchestration"):
        pass
    assert len(recorder.records) == 1
    assert recorder.records[0].stage == "orchestration"


def test_null_stage_recorder_is_a_true_no_op(tmp_path):
    marker = tmp_path / "should-not-exist.jsonl"
    recorder = NullStageRecorder()
    with recorder.stage("model_inference", trial_index=1):
        pass
    assert not marker.exists()


def test_jsonl_lines_are_valid_standalone_json(tmp_path):
    path = tmp_path / "stages.jsonl"
    sink = JsonlWriter(path)
    recorder = StageRecorder("r1", sink)
    for i in range(3):
        with recorder.stage("model_inference", trial_index=i):
            pass
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # each line parses independently
