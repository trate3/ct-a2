"""Time certification protocol sessions with GPU attestation over the real model files (EQ1 Attestation and Certification columns).

Each session runs the full protocol (`ct_core.certification.run_session`). The agent and tap packages name the real model
files, so `commit_artifact_hash` hashes the real bytes twice; the evaluation inside the session uses the stand-in drivers of
eq1_cert_standin.py and its own time is subtracted. Model loading is the Setup column and is not part of a session.

Configurations (what the commit hashes): probe = agent files + probe artifact; e4b = agent files + Gemma 4 E4B files;
qwen = agent files + the same Qwen files as the tap (hashed once per pass, as the protocol does for a shared checkpoint).

Page cache: with --evict-first, repetition 0 of each configuration starts with its files evicted from the page cache
(posix_fadvise DONTNEED on exactly those files); its cache_state is "evicted". Later repetitions read the same files again.
Without --evict-first the first repetition is "as_found".

Attestation: attestation_fetch (GPU evidence and the NRAS call) and attestation_verify (NRAS keys, token and claims) are
timed by the protocol's stage recorder; their parts come from the attestation record. --attestation nvidia requires the GPU
policy file and its digest in EQ1_GPU_POLICY_FILE and EQ1_GPU_POLICY_DIGEST and checks them before touching the GPU. The NRAS
keys are fetched at every verification.

  python -m evals_runner.eq1_cert_timing run --out DIR --qwen-dir QWEN --e4b-dir E4B [--probe-file P] [--configs probe,e4b,qwen]
         [--repeats 5] [--evict-first] [--attestation nvidia|stub]
  python -m evals_runner.eq1_cert_timing summarize --out DIR       # writes DIR/cert_timings.json from DIR/reps.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTESTATION_MODE = "nvidia-gpu-only-software-signed"
COMMON = ("model_inference", "tool_call", "orchestration", "scoring")            # the evaluation's own work (stand-in here)
NESTED_IN_COMMIT = ("artifact_hash_pre", "component_load", "artifact_hash_post")   # inside commit_artifact_hash: never added to it
OVERLAPPED = ("certifier_prepare_environment",)                                    # runs on the certifier thread while the workload hashes
RUN_WALL = "run_wall"
CODE_FILES = ("ct_core/certification.py", "ct_core/protocol/sealing.py", "tee/extension/nvidia_attestor.py",
              "evals_runner/eq1_cert_timing.py", "evals_runner/eq1_cert_standin.py")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def model_files(directory: str) -> tuple[tuple[str, str], ...]:
    """The files of a model directory, named model/<relative path>."""
    if not directory or not Path(directory).is_dir():
        raise SystemExit(f"model directory not found: {directory}")
    source = Path(directory).resolve()
    files = tuple((f"model/{p.relative_to(source).as_posix()}", str(p)) for p in sorted(source.rglob("*")) if p.is_file())
    if not files:
        raise SystemExit(f"model directory is empty: {directory}")
    return files


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def cached_kib() -> int | None:
    try:
        return next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("Cached:"))
    except (OSError, StopIteration):
        return None


def evict(paths: list[str]) -> None:
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def main_stage_running() -> bool:
    return bool(sh(["pgrep", "-f", r"[p]ython.* -m evals\.tap_live\.generate_calls_stage2"]))


def gpu_busy() -> str:
    return sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])


def require_policy(attestation: str) -> dict:
    """nvidia mode: both policy variables must be set and the file must match the digest, checked before the GPU is used."""
    if attestation != "nvidia":
        return {}
    file, digest = os.environ.get("EQ1_GPU_POLICY_FILE"), os.environ.get("EQ1_GPU_POLICY_DIGEST")
    if not file or not digest:
        raise SystemExit("--attestation nvidia needs EQ1_GPU_POLICY_FILE and EQ1_GPU_POLICY_DIGEST (the preselected GPU policy); not starting")
    from tee.extension.nvidia_attestor import GpuPolicy
    policy = GpuPolicy.load(file, expected_digest=digest)       # raises on a mismatch
    return {"policy_file": file, "policy_digest": policy.digest, "policy_file_sha256": sh(["sha256sum", file]).split()[0]}


def certified_session(agent_package, tap_package, environment, agent_drivers, tap_drivers, stage_recorder, attestation: str = "stub") -> dict:
    """One protocol session in one process, with the stub attestation or a GPU attestation under the policy from the environment."""
    from ct_core.certification import SessionTelemetry, run_session

    traffic = SessionTelemetry()
    signer = verifier = attestation_info = None
    if attestation == "nvidia":
        import tee.extension.nvidia_attestor as nv

        policy_file, policy_digest = os.environ.get("EQ1_GPU_POLICY_FILE"), os.environ.get("EQ1_GPU_POLICY_DIGEST")
        if not policy_file or not policy_digest:
            raise ValueError("--attestation nvidia needs a preselected GPU policy: set EQ1_GPU_POLICY_FILE and EQ1_GPU_POLICY_DIGEST")
        policy = nv.GpuPolicy.load(policy_file, expected_digest=policy_digest)
        with stage_recorder.stage("attestation_fetch"):
            signer = nv.GpuAttestedSoftwareSigner()
        with stage_recorder.stage("attestation_verify"):
            verifier = nv.GpuAttestationVerifier.from_token(signer.token(), signer.verify_key(), policy=policy)
        attestation_info = nv.attestation_record(verifier, signer)
    elif attestation != "stub":
        raise ValueError(f"unknown attestation backend {attestation!r}")
    cert = run_session(agent_package, tap_package, environment, agent_drivers=agent_drivers, tap_drivers=tap_drivers,
                       signer=signer, verifier=verifier, stage_recorder=stage_recorder, telemetry=traffic)
    return {
        "m": cert["m"], "m_fp": cert["m_fp"],
        # with GPU attestation the certifier checks a software signature under the GPU-attested key
        "certificate_verified": None if attestation_info else True,
        **({"software_signature_verified": True} if attestation_info else {}),
        "protocol_sent": traffic.sent_bytes, "protocol_received": traffic.received_bytes,
        "certificate_size": len(json.dumps(cert, sort_keys=True, separators=(",", ":"))),
        **({"attestation_info": attestation_info} if attestation_info else {}),
    }


def packages(config: str, items: list[dict], dirs: dict):
    from ct_core.interfaces import ArtifactPackage
    from evals_runner.eq1_cert_standin import StandInAgentDriver, StandInTapDriver

    agent_files = model_files(dirs["qwen"])
    agent = ArtifactPackage(StandInAgentDriver.DRIVER, {"items": items, "program_model": dirs["qwen"], "device": "cuda", "dtype": "bfloat16"}, agent_files)
    if config == "probe":
        tap = ArtifactPackage(StandInTapDriver.DRIVER, {"kind": "probe", "threshold": 0.0}, (("detector.json", dirs["probe"]),))
    elif config == "e4b":
        tap = ArtifactPackage(StandInTapDriver.DRIVER, {"monitor_model": dirs["e4b"], "device": "cuda", "dtype": "bfloat16"}, model_files(dirs["e4b"]))
    elif config == "qwen":
        tap = ArtifactPackage(StandInTapDriver.DRIVER, {"monitor_model": dirs["qwen"], "device": "cuda", "dtype": "bfloat16"}, agent_files)
    else:
        raise SystemExit(f"unknown configuration {config!r}")
    return agent, tap, (StandInAgentDriver,), (StandInTapDriver,)


def one_run(config, repeat, order_index, cache_state, a, items, dirs, sink_dir) -> dict:
    from ct_core.certification import committed_artifact_bytes
    from ct_core.protocol.telemetry import JsonlWriter, StageRecorder
    from util.dynamic_test_environment import StaticEnvironment

    agent, tap, agent_drivers, tap_drivers = packages(config, items, dirs)
    files = sorted({p for pkg in (agent, tap) for _, p in pkg.artifacts if isinstance(p, str) and Path(p).is_file()})
    if cache_state == "evicted":
        evict(files)
    environment = StaticEnvironment.from_jsonl(a.fixture, a.stand_in_trials, seed=7)
    run_id = str(uuid.uuid4())
    recorder = StageRecorder(run_id, JsonlWriter(sink_dir / "stages.jsonl"))
    cached_before, started = cached_kib(), now_utc()
    with recorder.stage(RUN_WALL):
        outcome = certified_session(agent, tap, environment, agent_drivers, tap_drivers, recorder, attestation=a.attestation)
    cached_after = cached_kib()
    walls: dict[str, float] = {}
    for rec in recorder.records:
        walls[rec.stage] = walls.get(rec.stage, 0.0) + rec.wall_seconds
    hashed = committed_artifact_bytes(agent, tap)
    wall = walls[RUN_WALL]
    common = sum(walls.get(s, 0.0) for s in COMMON)
    att = walls.get("attestation_fetch", 0.0) + walls.get("attestation_verify", 0.0)
    info = outcome.get("attestation_info") or {}
    if a.attestation == "nvidia":
        assert info.get("gpu_attestation_verified") is True and info.get("workload_attested") is False and info.get("full_certification") is False, info
        assert outcome.get("software_signature_verified") is True and outcome["certificate_verified"] is None, "unexpected outcome flags for the GPU attestation mode"
    skip = {RUN_WALL, *COMMON, *NESTED_IN_COMMIT, *OVERLAPPED}
    named = sum(v for k, v in walls.items() if k not in skip)
    return {"run_id": run_id, "config": config, "repeat": repeat, "order_index": order_index, "cache_state": cache_state, "started_utc": started,
            "attestation": a.attestation, "run_wall_s": wall, "stage_wall_s": walls, "common_stand_in_s": common,
            "attestation_fetch_s": walls.get("attestation_fetch"), "attestation_verify_s": walls.get("attestation_verify"), "attestation_s": att,
            "commit_artifact_hash_s": walls.get("commit_artifact_hash"), "hash_pass_pre_s": walls.get("artifact_hash_pre"),
            "hash_pass_post_s": walls.get("artifact_hash_post"), "certifier_prepare_environment_s": walls.get("certifier_prepare_environment"),
            "named_protocol_s": named, "protocol_overhead_s": wall - common, "unattributed_s": wall - common - named,
            "artifact_bytes_hashed_two_passes": hashed, "artifact_files": len(files), "artifact_gib_hashed_two_passes": hashed / 2 ** 30,
            "page_cache_kib_before": cached_before, "page_cache_kib_after": cached_after,
            "m": outcome["m"], "m_fp": outcome["m_fp"], "certificate_verified": outcome["certificate_verified"],
            "software_signature_verified": outcome.get("software_signature_verified"),
            "certificate_size": outcome["certificate_size"], "attestation_record": outcome.get("attestation_info"),
            "attestation_parts_s": {**(info.get("signer_timings") or {}), **{k: v for k, v in (info.get("verifier_timings") or {}).items() if k.endswith("_seconds")}},
            "key_cache_policy": (info.get("verifier_timings") or {}).get("key_cache_policy")}


def run(a) -> int:
    sys.path.insert(0, a.repo_root)
    if not a.skip_main_check and main_stage_running():
        raise SystemExit("the main stage (generate_calls_stage2) is still running; not starting")
    busy = gpu_busy()
    if busy and a.attestation == "nvidia" and not a.allow_busy_gpu:
        raise SystemExit(f"the GPU has compute processes:\n{busy}\nnot starting (--allow-busy-gpu overrides)")
    out = Path(a.out)
    if (out / "reps.jsonl").exists():
        raise SystemExit(f"{out}/reps.jsonl exists; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    dirs = {"qwen": a.qwen_dir, "e4b": a.e4b_dir, "probe": a.probe_file}
    from evals.basic_mocked import FIXTURE
    a.fixture = a.fixture or str(FIXTURE)
    items = [json.loads(l) for l in Path(a.fixture).read_text().splitlines() if l.strip()]
    policy = require_policy(a.attestation)
    configs = [c for c in a.configs.split(",") if c]
    meta = {"started_utc": now_utc(), "argv": sys.argv, "attestation": a.attestation, "attestation_mode": ATTESTATION_MODE if a.attestation == "nvidia" else "stub",
            "policy": policy, "policy_enforced": bool(policy),
            "configs": configs, "repeats": a.repeats, "evict_first": a.evict_first, "dirs": dirs, "stand_in_trials": a.stand_in_trials,
            "env": {k: os.environ.get(k) for k in ("EQ1_GPU_POLICY_FILE", "EQ1_GPU_POLICY_DIGEST")},
            "host": sh(["hostname"]), "kernel": sh(["uname", "-r"]), "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,vbios_version", "--format=csv,noheader"]),
            "conf_compute": sh(["nvidia-smi", "conf-compute", "-q"]), "cpu_count": os.cpu_count(),
            "code_sha256": {f: sh(["sha256sum", str(Path(a.repo_root) / f)]).split()[0] for f in CODE_FILES}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    seen: set[str] = set()
    for rep in range(a.repeats):
        order = configs[rep % len(configs):] + configs[:rep % len(configs)]       # rotate the order so no configuration always follows the same other one
        for idx, config in enumerate(order):
            first = config not in seen
            seen.add(config)
            state = ("evicted" if a.evict_first else "as_found") if first else "warm"
            if not a.skip_main_check and main_stage_running():
                raise SystemExit("the main stage started running; stopping")
            rec = one_run(config, rep, idx, state, a, items, dirs, out)
            with (out / "reps.jsonl").open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            print(f"[cert] rep {rep} {config:5s} {state:8s} run_wall {rec['run_wall_s']:.2f}s  attestation {rec['attestation_s']:.3f}s  "
                  f"hash {rec['commit_artifact_hash_s']:.2f}s ({rec['artifact_gib_hashed_two_passes']:.1f} GiB, 2 passes)  protocol {rec['protocol_overhead_s']:.2f}s", flush=True)
    summarize(a)
    return 0


def spread(values):
    values = [v for v in values if v is not None]
    return None if not values else {"mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def summarize(a) -> int:
    out = Path(a.out)
    meta = json.loads((out / "meta.json").read_text())
    reps = [json.loads(l) for l in (out / "reps.jsonl").read_text().splitlines() if l.strip()]
    configs = {}
    for name in sorted({r["config"] for r in reps}):
        rs = [r for r in reps if r["config"] == name]
        by = lambda state: [r for r in rs if r["cache_state"] == state]  # noqa: E731
        warm, cold = by("warm"), by("evicted")
        pick = lambda key, group: (statistics.fmean([r[key] for r in group]) if group else None)  # noqa: E731
        configs[name] = {
            "n_repetitions": len(rs), "cache_states": {s: len(by(s)) for s in ("warm", "evicted", "as_found")},
            "artifact_gib_hashed": rs[0]["artifact_gib_hashed_two_passes"], "artifact_files": rs[0]["artifact_files"],
            "protocol_overhead_s": {"warm": pick("protocol_overhead_s", warm), "cold": pick("protocol_overhead_s", cold)},
            "commit_artifact_hash_s": {"warm": pick("commit_artifact_hash_s", warm), "cold": pick("commit_artifact_hash_s", cold)},
            "attestation_s": spread([r["attestation_s"] for r in rs]),
            "named_protocol_s_warm_mean": pick("named_protocol_s", warm), "unattributed_s_warm_mean": pick("unattributed_s", warm),
            "other_named_stages_s_warm_mean": {k: statistics.fmean([r["stage_wall_s"].get(k, 0.0) for r in warm]) for k in sorted(
                {k for r in warm for k in r["stage_wall_s"]} - set(COMMON) - set(NESTED_IN_COMMIT) - {RUN_WALL, "commit_artifact_hash"})} if warm else None,
            "all_run_wall_s": [r["run_wall_s"] for r in rs]}
    att_all = [r["attestation_s"] for r in reps]
    policies = {r["key_cache_policy"] for r in reps}
    assert len(policies) == 1, f"repetitions mix key-cache policies: {policies}"
    nv_mode = meta["attestation"] == "nvidia"
    part_names = sorted({k for r in reps for k in r["attestation_parts_s"]})
    summary = {"kind": "cert_timings", "attestation_mode": meta["attestation_mode"], "policy_enforced": meta["policy_enforced"],
               "policy_digest": (meta["policy"] or {}).get("policy_digest"), "policy_file_sha256": (meta["policy"] or {}).get("policy_file_sha256"),
               "gpu_attestation_verified": nv_mode, "workload_attested": False, "cpu_attested": False, "full_certification": False,
               "scope": ("GPU attestation of a software-held signing key" if nv_mode else "stub attestation"),
               "key_cache_policy": policies.pop(),
               "attestation_definition": "mean over repetitions of the per-repetition SUM attestation_fetch + attestation_verify (outer wall times, network included)",
               "attestation_parts_s": {k: spread([r["attestation_parts_s"].get(k) for r in reps]) for k in part_names},
               "measurements_seen": sorted({(r["attestation_record"] or {}).get("measurement") for r in reps} - {None}),
               "n_repetitions_pooled": len(reps), "attestation_pooled_mean_s": statistics.fmean(att_all), "attestation_pooled": spread(att_all),
               "attestation_fetch_pooled": spread([r["attestation_fetch_s"] for r in reps]), "attestation_verify_pooled": spread([r["attestation_verify_s"] for r in reps]),
               "configs": configs, "host": meta["host"], "gpu": meta["gpu"],
               "composition_note": "the Certification column is the mean protocol_overhead_s of the repetitions after the first of each configuration",
               "definition": "protocol_overhead_s = run_wall of one session with a stand-in evaluation minus the stand-in's own work (model_inference, tool_call, "
                             "orchestration, scoring); it contains attestation, the two artifact hash passes, signing and verification, the sealed-environment "
                             "exchange and any unattributed time. Model loading and the evaluation itself are separate columns."}
    (out / "cert_timings.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("attestation_mode", "policy_enforced", "n_repetitions_pooled", "attestation_pooled_mean_s")}, indent=1))
    for name, c in configs.items():
        print(f"  {name}: protocol {c['protocol_overhead_s']['warm']} s (first read {c['protocol_overhead_s']['cold']} s); hash {c['commit_artifact_hash_s']['warm']} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--repo-root", default=str(ROOT))
    r.add_argument("--configs", default="probe,e4b,qwen")
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--attestation", choices=("nvidia", "stub"), default="nvidia")
    r.add_argument("--evict-first", action="store_true", help="evict the model files from the page cache before the first run of each configuration")
    r.add_argument("--qwen-dir", default=os.environ.get("MODEL_DIR"), help="Qwen3.8-27B snapshot directory [MODEL_DIR]")
    r.add_argument("--e4b-dir", default=os.environ.get("E4B_DIR"), help="Gemma 4 E4B snapshot directory [E4B_DIR]")
    r.add_argument("--probe-file", default=str(ROOT / "data/eq1/probe/tap_p2_logistic_mean.json"))
    r.add_argument("--fixture", default=None)
    r.add_argument("--stand-in-trials", type=int, default=4)
    r.add_argument("--skip-main-check", action="store_true", help="tests only")
    r.add_argument("--allow-busy-gpu", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else summarize(a)


if __name__ == "__main__":
    raise SystemExit(main())
