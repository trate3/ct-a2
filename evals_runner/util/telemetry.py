"""RQ1 cost-of-certification telemetry: JSONL run + stage records.

One run record per (arm, configuration, repeat); one stage record per instrumented
stage. Analysis (curve fitting, confidence intervals, dollar cost) happens entirely in
post-processing against these files, never during the run -- this module only
measures and writes. Every field name below matches the RQ1 experiment specification
so the files can be loaded without a translation layer.

Certification-stage instrumentation lives in `ct_core/certification.py` (the protocol
engine); baseline-stage instrumentation lives in each arm's execution loop
(`evals_runner/rq1_cost_exp.py`). Both write through the same `StageRecorder`/
`JsonlWriter` pair defined here so one JSONL file mixes both without special-casing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import resource
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ct_core.protocol.telemetry import (
    JsonlWriter,
    NullStageRecorder,
    StageRecord,
    StageRecorder,
    read_jsonl,
)

__all__ = [
    "JsonlWriter", "NullStageRecorder", "StageRecord", "StageRecorder", "read_jsonl",
    "MachineInfo", "ModelInfo", "TapInfo", "WorkloadInfo", "RunInfo", "RunRecord",
    "canonical_digest", "repo_commit", "peak_rss_bytes", "peak_gpu_memory_bytes",
    "runtime_info", "GpuUtilSampler", "CERTIFICATION_STAGES", "BASELINE_STAGES",
]

CERTIFICATION_STAGES = (
    "enclave_install", "attestation_fetch", "commit_artifact_hash",
    "env_decrypt_and_check", "session_reset", "tap_outcome_retain",
    "aggregate", "certificate_sign", "certifier_verify",
)
BASELINE_STAGES = ("prompt_build", "model_inference", "tool_call", "orchestration", "scoring")


def canonical_digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def repo_commit(cwd: str | Path | None = None) -> str:
    """Best-effort current commit hash; 'unknown' outside a git checkout.

    `CT_REPO_COMMIT` wins when set: a cloud run executes a copied working tree with no
    .git, so the deploy script stamps the commit (plus a dirty-tree marker) it shipped.
    """
    import os

    if os.environ.get("CT_REPO_COMMIT"):
        return os.environ["CT_REPO_COMMIT"]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
            text=True, timeout=5, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    return rss if sys.platform == "darwin" else rss * 1024


def peak_gpu_memory_bytes() -> int | None:
    """None when no CUDA device is available or torch isn't installed."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


# Optional kernel packages whose absence transformers handles by silently switching to
# a reference PyTorch implementation -- still on the GPU, but far slower. Recorded so
# a slow run can be told apart from a slow machine.
_KERNEL_PACKAGES = ("fla", "causal_conv1d", "flash_attn")


def runtime_info(model=None) -> dict:
    """What the run *actually* executed on, detected rather than supplied by flags.

    `MachineInfo` is what the operator says the machine is; this is what torch saw:
    the device every model parameter lives on, the GPU's own name, and the library
    builds. A record claiming an L4 whose parameters were on the CPU is caught here.
    """
    info: dict = {"kernels": {name: importlib.util.find_spec(name) is not None
                              for name in _KERNEL_PACKAGES}}
    try:
        import torch
    except ImportError:
        return info
    info.update(torch_version=torch.__version__, torch_cuda_build=torch.version.cuda,
                cuda_available=torch.cuda.is_available())
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["gpu_total_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    if model is not None:
        info["param_devices"] = sorted({p.device.type for p in model.parameters()})
        info["param_dtypes"] = sorted({str(p.dtype).replace("torch.", "")
                                       for p in model.parameters()})
        info["model_type"] = getattr(getattr(model, "config", None), "model_type", None)
    return info


class GpuUtilSampler:
    """Samples GPU utilization and memory through one streaming `nvidia-smi` process.

    Peak allocated memory proves the weights reached the GPU; utilization proves the
    GPU did the work. A run whose mean utilization sits near zero was CPU-bound
    somewhere (a CPU-resident layer, a Python loop, disk) however its memory looks.
    No-op, returning None, where `nvidia-smi` is absent (laptops, CI).
    """

    def __init__(self, interval_ms: int = 500):
        self._interval_ms = interval_ms
        self._samples: list[tuple[float, float]] = []
        self._process = None
        self._reader = None

    def __enter__(self):
        binary = shutil.which("nvidia-smi")
        if binary is None:
            return self
        self._process = subprocess.Popen(
            [binary, "--query-gpu=utilization.gpu,memory.used", "--id=0",
             "--format=csv,noheader,nounits", f"-lms={self._interval_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        return self

    def _read(self):
        for line in self._process.stdout:
            try:
                utilization, memory = (float(part) for part in line.split(","))
            except ValueError:
                continue
            self._samples.append((utilization, memory))

    def __exit__(self, *exc):
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._reader.join(timeout=10)
        return False

    def summary(self) -> dict | None:
        if not self._samples:
            return None
        utilization = sorted(sample[0] for sample in self._samples)
        return {
            "samples": len(utilization),
            "interval_ms": self._interval_ms,
            "util_mean_pct": sum(utilization) / len(utilization),
            "util_p50_pct": utilization[len(utilization) // 2],
            "util_max_pct": utilization[-1],
            "mem_used_max_mib": max(sample[1] for sample in self._samples),
        }


@dataclass(frozen=True)
class MachineInfo:
    sku: str
    vcpu: int
    memory_gb: float
    gpu_model: str | None
    region: str
    cc_technology: str  # "none" | "SEV" | "SEV_SNP" | "TDX"
    image_family: str


@dataclass(frozen=True)
class ModelInfo:
    id: str
    revision: str
    dtype: str
    quantization: str | None = None
    params_total: int | None = None
    params_active_per_token: int | None = None


@dataclass(frozen=True)
class TapInfo:
    type: str  # "probe" | "cot" | "none" (the probe-cost baseline; see rq1_cost_exp)
    monitor_model_id: str | None = None
    probe_layers: tuple[int, ...] | None = None
    threshold: float | None = None
    # "trained" | "cost_only" (make_cost_probe: shape-matched, never a detector)
    probe_kind: str | None = None
    # black-box monitor view (evals/deception/taps.py): "full" | "answer"
    monitor_view: str | None = None
    # the probe package's path, e.g. .../roleplaying__lowo.npz (which probe, not just its kind)
    detector: str | None = None


@dataclass(frozen=True)
class WorkloadInfo:
    eval_case: str  # "confidence" | "financial"
    n_trials: int   # trials actually executed, never merely requested
    k_max: int
    max_new_tokens: int
    prompt_template_id: str
    # Distinct test items behind n_trials; < n_trials only when scenarios were cycled.
    unique_trials: int | None = None
    reasoning: str | None = None  # the program's thinking-effort level, if set
    # the program's sampling temperature (0 = greedy); None when it generates nothing
    temperature: float | None = None


@dataclass(frozen=True)
class RunInfo:
    arm: str  # "A0" | "A1" | "A2"
    seed: int
    repeat_index: int
    order: str  # "direct-first" | "certified-first" | "n/a"
    warmup: bool
    attestation: str = "stub"  # certified arm: "stub" (local signer, no TEE) | "gcp" (Confidential Space token)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config_digest: str
    machine: MachineInfo
    model: ModelInfo
    tap: TapInfo
    workload: WorkloadInfo
    run: RunInfo
    outcome: dict       # {m, m_fp, aborted, certificate_verified}
    totals: dict        # {model_load_seconds, peak_rss_bytes, peak_gpu_memory_bytes}
    bytes: dict         # {protocol_sent, protocol_received, artifact_hashed, certificate_size}
    timestamp: str = ""
    repo_commit: str = ""
    runtime: dict = field(default_factory=dict)  # runtime_info() + GPU utilization

    def __post_init__(self):
        if not self.timestamp:
            object.__setattr__(self, "timestamp", datetime.now(timezone.utc).isoformat())
        if not self.repo_commit:
            object.__setattr__(self, "repo_commit", repo_commit())

    def to_json(self) -> dict:
        return {
            "run_id": self.run_id, "timestamp": self.timestamp,
            "repo_commit": self.repo_commit, "config_digest": self.config_digest,
            "machine": asdict(self.machine), "model": asdict(self.model),
            "tap": asdict(self.tap), "workload": asdict(self.workload),
            "run": asdict(self.run), "outcome": dict(self.outcome),
            "totals": dict(self.totals), "bytes": dict(self.bytes),
            "runtime": dict(self.runtime),
        }

