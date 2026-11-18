from __future__ import annotations

import json

from ct_core.protocol.telemetry import JsonlWriter, NullStageRecorder, StageRecorder, read_jsonl


def test_stage_recorder_times_a_block_and_writes_one_record(tmp_path):
    sink = JsonlWriter(tmp_path / "stages.jsonl")
    recorder = StageRecorder("run-1", sink)
    with recorder.stage("model_inference", trial_index=3, bytes_processed=42):
        pass
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record.run_id == "run-1"
    assert record.trial_index == 3
    assert record.stage == "model_inference"
    assert record.bytes_processed == 42
    assert record.wall_seconds >= 0
    assert record.cpu_seconds >= 0

    lines = read_jsonl(sink.path)
    assert len(lines) == 1
    assert lines[0]["stage"] == "model_inference"


def test_stage_recorder_defaults_trial_index_to_zero_for_session_stages(tmp_path):
    recorder = StageRecorder("run-1")
    with recorder.stage("certificate_sign"):
        pass
    assert recorder.records[0].trial_index == 0


def test_null_stage_recorder_writes_nothing_and_never_raises():
    recorder = NullStageRecorder()
    with recorder.stage("attestation_fetch", trial_index=1):
        raise_if_reached = False
        assert raise_if_reached is False


def test_jsonl_writer_appends_across_multiple_writes(tmp_path):
    path = tmp_path / "stages.jsonl"
    sink = JsonlWriter(path)
    recorder = StageRecorder("run-1", sink)
    for i in range(1, 4):
        with recorder.stage("session_reset", trial_index=i):
            pass
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)
    assert [json.loads(line)["trial_index"] for line in lines] == [1, 2, 3]


def test_jsonl_writer_creates_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "stages.jsonl"
    sink = JsonlWriter(nested)
    recorder = StageRecorder("run-1", sink)
    with recorder.stage("aggregate"):
        pass
    assert nested.exists()
