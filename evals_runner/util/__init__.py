"""Shared utilities for evaluation-runner control-plane code."""

from .comparison import ExperimentConfig, run_comparison, write_certificate, write_result
from .tap_quality import run_tap_quality
from .telemetry import (
    JsonlWriter,
    MachineInfo,
    ModelInfo,
    NullStageRecorder,
    RunInfo,
    RunRecord,
    StageRecorder,
    StageRecord,
    TapInfo,
    WorkloadInfo,
    canonical_digest,
    read_jsonl,
    repo_commit,
)

__all__ = [
    "ExperimentConfig",
    "run_comparison", "run_tap_quality", "write_certificate", "write_result",
    "JsonlWriter", "MachineInfo", "ModelInfo", "NullStageRecorder", "RunInfo",
    "RunRecord", "StageRecorder", "StageRecord", "TapInfo", "WorkloadInfo",
    "canonical_digest", "read_jsonl", "repo_commit",
]
