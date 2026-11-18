"""Evaluation-only direct/certified comparison and JSON output."""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ct_core.certification import (
    SessionTelemetry, measure, prepare_components, prepare_environment, run_session,
)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    seed: int
    n_trials: int
    environment: str
    parameters: dict = field(default_factory=dict)

    def as_dict(self):
        if not self.experiment or not self.environment or self.n_trials <= 0:
            raise ValueError("experiment, environment, and positive n_trials are required")
        value = asdict(self)
        json.dumps(value)
        return value


def _timed(call):
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    value = call()
    after = resource.getrusage(resource.RUSAGE_SELF)
    rss = after.ru_maxrss if sys.platform == "darwin" else after.ru_maxrss * 1024
    return value, {
        "wall_seconds": time.perf_counter() - start,
        "cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "process_lifetime_max_rss_bytes": int(rss), "token_counts": None,
    }


def run_comparison(config, *, build_agent_package, build_tap_package,
                   agent_drivers, tap_drivers, build_environment, certified_first=False,
                   deterministic=True):
    """Direct vs certified on the same components.

    `deterministic` agents (greedy decoding) must give identical (m, m_fp) both ways.
    A sampling agent draws each trial's seeds from fresh protocol randomness, so the two
    runs are different samples; they are both recorded and not required to match.
    """
    def direct():
        environment = build_environment()
        spec, packed = prepare_environment(environment)
        if spec["n_trials"] != config.n_trials:
            raise ValueError("environment trial count differs from config")
        result, metrics = _timed(lambda: measure(
            prepare_components(
                build_agent_package(), build_tap_package(), agent_drivers, tap_drivers),
            packed, spec,
            (environment.runtime_driver(),)))
        for private in ("_tests", "_steps", "_trial_predictions", "_trial_labels"):
            result.pop(private)
        return result, metrics, spec

    def certified():
        environment = build_environment()
        traffic = SessionTelemetry()
        cert, metrics = _timed(
            lambda: run_session(
                build_agent_package(), build_tap_package(), environment,
                agent_drivers=agent_drivers, tap_drivers=tap_drivers, telemetry=traffic))
        return cert, metrics, traffic

    if certified_first:
        cert, certified_metrics, traffic = certified()
        direct_result, direct_metrics, spec = direct()
    else:
        direct_result, direct_metrics, spec = direct()
        cert, certified_metrics, traffic = certified()
    if deterministic and direct_result != {"m": cert["m"], "m_fp": cert["m_fp"]}:
        raise ValueError("certification changed the measurement")
    direct_metrics["wire_bytes"] = 0
    certified_metrics["wire_bytes"] = traffic.sent_bytes + traffic.received_bytes
    return {
        "schema_version": "ct-eval/1", "config": config.as_dict(),
        "execution_order": "certified-first" if certified_first else "direct-first",
        "preservation_checked": deterministic,
        "environment_digest": spec["id_E"],
        "direct": {"measurement": direct_result, "metrics": direct_metrics},
        "certified": {"certificate": cert, "metrics": certified_metrics},
        "certified_wire_scope": "certifier-enclave-socketpair",
        "overhead": {
            "wall_seconds": certified_metrics["wall_seconds"] - direct_metrics["wall_seconds"],
            "cpu_seconds": certified_metrics["cpu_seconds"] - direct_metrics["cpu_seconds"],
            "wire_bytes": certified_metrics["wire_bytes"],
        },
    }


def write_result(result: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return destination


def write_certificate(cert: dict, spec: dict, output: str) -> dict:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cert.json").write_text(json.dumps(cert, indent=2) + "\n")
    (destination / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return cert
