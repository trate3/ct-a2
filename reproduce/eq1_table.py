"""Recompute the EQ1 phase table (time for the phases of certified-tap evaluation) from data/eq1 and print it.

Design (fixed before timing): 10 development LiveCodeBench tasks, prompts at 25%, 50% and 75% of each recorded run
(an earlier call when the 48,000-token prompt cap applies), 30 weighted positions, 24 distinct prompts. Each distinct
prompt is generated once without the tap (A0) and once with the mean probe (A1), same seed, natural stop.

  Setup          probe: agent loading plus probe loading; monitors: agent loading plus monitor server launch-to-health time
  No tap         weighted mean A0 generation time per call
  With tap       probe: weighted mean A1 time; monitors: A0 time plus the monitor's judging time of the saved A0 response
  Tap            probe: isolated tap work per call (data/eq1/tap_work.json); monitors: weighted mean judging time
  Attestation    mean GPU attestation time (fetch plus verify) over all certification sessions
  Certification  mean protocol session time of the row's configuration (data/eq1/certification/sessions.jsonl)

Means use the 30 position weights; the intervals in --json are 95% task-bootstrap intervals (2,000 draws, seed 0).

Usage: python reproduce/eq1_table.py [--data data/eq1] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# monitor folder, row label, certification configuration
MONITORS = [("gemma4-e4b", "CoT: Gemma-4-E4B (none)", "e4b"), ("gemma4-e4b-think", "CoT: Gemma-4-E4B (enabled)", "e4b"),
            ("qwen3.8-27b-4k", "CoT: Qwen3.8-27B (none)", "qwen"), ("qwen3.8-27b-think", "CoT: Qwen3.8-27B (medium)", "qwen")]


def read(p):
    return json.loads(Path(p).read_text())


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def make_stat(positions):
    tasks = sorted({p["task"] for p in positions})

    def stat(values):
        """Weighted mean over the positions and a task-clustered bootstrap interval."""
        sums = np.array([sum(p["weight"] * values[p["run_key"]] for p in positions if p["task"] == t) for t in tasks])
        ws = np.array([sum(p["weight"] for p in positions if p["task"] == t) for t in tasks])
        ix = np.random.default_rng(0).integers(0, len(tasks), size=(2000, len(tasks)))
        draws = sums[ix].sum(axis=1) / ws[ix].sum(axis=1)
        return {"estimate": float(sums.sum() / ws.sum()), "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}
    return stat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "eq1")
    ap.add_argument("--json", type=Path, default=None, help="also write the rows with intervals to this file")
    args = ap.parse_args(argv)
    D = args.data

    positions = read(D / "generation" / "stage2_list.json")["positions"]
    stat = make_stat(positions)
    calls = {(r["run_key"], r["arm"]): r for r in jl(D / "generation" / "calls.jsonl")}
    assert len(calls) == 48, len(calls)
    for (key, arm), r in calls.items():
        assert r["status"] == "ok" and r["stopped_naturally"], (key, arm)
        assert r["token_hash"] == calls[key, "A1" if arm == "A0" else "A0"]["token_hash"], "A0/A1 token streams differ"
    a0 = {k: r["elapsed_s"] for (k, arm), r in calls.items() if arm == "A0"}
    a1 = {k: r["elapsed_s"] for (k, arm), r in calls.items() if arm == "A1"}
    setup = read(D / "generation" / "setup.json")
    no_tap = stat(a0)

    att = [r["attestation_fetch_s"] + r["attestation_verify_s"] for r in jl(D / "certification" / "attestation.jsonl")]
    attestation_s = statistics.fmean(att)
    sessions = jl(D / "certification" / "sessions.jsonl")
    cert = {c: statistics.fmean(r["certification_s"] for r in sessions if r["config"] == c) for c in {r["config"] for r in sessions}}

    tap_work = read(D / "tap_work.json")["tap_work_ms"]
    rows = [{"agent": "Qwen3.8-27B (xhigh)", "tap": "Probe", "setup_s": setup["model_load_s"] + setup["artifact_load_s"],
             "no_tap_s": no_tap, "with_tap_s": stat(a1),
             "tap_s": {"estimate": tap_work["estimate"] / 1000, "ci95": [v / 1000 for v in tap_work["ci95"]]},
             "attestation_s": attestation_s, "certification_s": cert["probe"]}]
    startup = read(D / "monitors" / "startup.json")["seconds_launch_to_health_ok"]
    for name, label, config in MONITORS:
        raw = jl(D / "monitors" / name / "rows.jsonl")
        assert len(raw) == 24 and len({r["run_key"] for r in raw}) == 24, name
        assert all(r["valid_score"] and r["finish_reason"] == "stop" and r["attempt"] == 1 and r["http_status"] == 200 for r in raw), name
        secs = {r["run_key"]: r["seconds"] for r in raw}
        rows.append({"agent": "Qwen3.8-27B (xhigh)", "tap": label, "setup_s": setup["model_load_s"] + startup[name],
                     "no_tap_s": no_tap, "with_tap_s": stat({k: a0[k] + v for k, v in secs.items()}), "tap_s": stat(secs),
                     "attestation_s": attestation_s, "certification_s": cert[config]})

    head = ["Agent", "Tap", "Setup (s)", "No tap (s)", "With tap (s)", "Tap (s/inf.)", "Attestation (s)", "Certification (s)"]
    table = [head] + [[r["agent"], r["tap"], f"{r['setup_s']:.2f}", f"{r['no_tap_s']['estimate']:.2f}", f"{r['with_tap_s']['estimate']:.2f}",
                       f"{r['tap_s']['estimate']:.3f}", f"{r['attestation_s']:.2f}", f"{r['certification_s']:.2f}"] for r in rows]
    width = [max(len(line[i]) for line in table) for i in range(len(head))]
    print(f"EQ1: {len(positions)} positions, {len(a0)} distinct prompts, {len({p['task'] for p in positions})} tasks")
    for j, line in enumerate(table):
        print("  ".join(c.ljust(w) if i < 2 else c.rjust(w) for i, (c, w) in enumerate(zip(line, width))))
        if j == 0:
            print("  ".join("-" * w for w in width))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
