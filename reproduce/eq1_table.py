"""Regenerate the EQ1 phase table (time for the phases of certified-tap evaluation) on CPU.

Reads the raw per-call export of the live generation run and the per-request judge
records, recomputes every cell, and prints it next to the reported number.

Design (fixed before timing): 10 development LiveCodeBench tasks, prompts at 25%, 50%
and 75% of each recorded run, a 48,000-token prompt cap with fallback to an earlier
call (30 weighted positions, 24 distinct prompts). Each distinct prompt is generated
once without the tap (A0) and once with the mean probe (A1), same seed, natural stop.
The probe row's Tap cell is the isolated tap execution time per call, from a separate
component microbenchmark that replays the tap's own work on synthetic hidden states at the
main-call shapes (data/eq1/profile_components). It is isolated work, not added latency, and
it is not a bound on the added latency. The paired A1 - A0 latency difference is reported
next to it as supporting evidence; it is unresolved around zero.
LLM-monitor rows add each monitor's separately measured judging time of the saved A0
response to the A0 generation time. Means use the 30 position weights; intervals are
95% task-bootstrap intervals (2,000 draws, seed 0, whole tasks resampled).

Usage: python reproduce/eq1_table.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.reward_hacking import monitors as m  # noqa: E402
from evals_runner.eq1_monitor_panel import reply_valid  # noqa: E402
from evals.tap_live.profile_components import summarize as profile_summarize  # noqa: E402

D = ROOT / "data" / "eq1"
GEN = D / "generation"
PAPER = json.loads((ROOT / "reproduce" / "paper_numbers.json").read_text())["eq1"]
MONITORS = [("rtx", D / "judges_rtx", ["gemma4-e4b", "gemma4-e4b-think", "qwen3.8-27b-4k", "qwen3.8-27b-think"]),
            ("second_machine", D / "judges_second_machine" / "e4b_qwen", ["gemma4-e4b", "gemma4-e4b-think", "qwen3.8-27b-4k", "qwen3.8-27b-think"]),
            ("second_machine", D / "judges_second_machine" / "glm", ["glm5.3-flash-nvfp4", "glm5.3-flash-nvfp4-think"])]
SHORT = {"gemma4-e4b": "E4B, off", "gemma4-e4b-think": "E4B, on", "qwen3.8-27b-4k": "Qwen, off",
         "qwen3.8-27b-think": "Qwen, on", "glm5.3-flash-nvfp4": "GLM, off", "glm5.3-flash-nvfp4-think": "GLM, on"}


def read(p):
    return json.loads(Path(p).read_text())


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


sel = read(GEN / "stage2_list.json")
positions = sel["positions"]
tasks = sorted({p["task"] for p in positions})


def stat(values):
    """Weighted mean over the 30 positions and a task-clustered bootstrap interval."""
    sums = np.array([sum(p["weight"] * values[p["run_key"]] for p in positions if p["task"] == t) for t in tasks])
    ws = np.array([sum(p["weight"] for p in positions if p["task"] == t) for t in tasks])
    ix = np.random.default_rng(0).integers(0, len(tasks), size=(2000, len(tasks)))
    draws = sums[ix].sum(axis=1) / ws[ix].sum(axis=1)
    return {"estimate": float(sums.sum() / ws.sum()), "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {label:44s} {got:14.6f}   paper {want:14.6f}   {'ok' if ok else 'DIFFERS'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results_regenerated" / "eq1")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ok = True

    calls = {(r["run_key"], r["arm"]): r for r in jl(GEN / "generate_calls_stage2" / "calls.jsonl")}
    assert len(calls) == 48, len(calls)
    for (key, arm), r in calls.items():
        assert r["status"] == "ok" and r["stopped_naturally"], (key, arm)
        assert r["token_hash"] == calls[key, "A1" if arm == "A0" else "A0"]["token_hash"], "A0/A1 token streams differ"
    a0 = {k: r["elapsed_s"] for (k, arm), r in calls.items() if arm == "A0"}
    a1 = {k: r["elapsed_s"] for (k, arm), r in calls.items() if arm == "A1"}
    setup = jl(GEN / "generate_calls_stage2" / "setup.jsonl")[0]
    no_tap, with_tap = stat(a0), stat(a1)
    diff = stat({k: (a1[k] - a0[k]) * 1000 for k in a0})
    print(f"Generation: {len(positions)} positions, {len(a0)} distinct prompts, {len(tasks)} tasks; "
          f"all 24 A0/A1 pairs stopped naturally with identical token streams")
    ok &= check("No tap, mean s per response", no_tap["estimate"], PAPER["no_tap_s"], 5e-6)
    ok &= check("With mean probe, mean s per response", with_tap["estimate"], PAPER["with_probe_s"], 5e-6)
    ok &= check("Paired A1 - A0 latency difference, ms (supporting)", diff["estimate"], PAPER["probe_added_ms"][0], 5e-6)
    ok &= check("  95% interval low", diff["ci95"][0], PAPER["probe_added_ms"][1], 5e-6)
    ok &= check("  95% interval high", diff["ci95"][1], PAPER["probe_added_ms"][2], 5e-6)
    probe_setup = setup["model_load_s"] + setup["artifact_load_s"]
    ok &= check("Setup: model + probe loading, s", probe_setup, PAPER["probe_setup_s"], 5e-6)
    # isolated tap execution time: recompute the profile tables from its raw passes (copied, so nothing shipped is touched)
    import shutil
    prof_dir = args.out / "profile_components"
    shutil.rmtree(prof_dir, ignore_errors=True)
    shutil.copytree(D / "profile_components", prof_dir)
    prof = profile_summarize(prof_dir)
    iso = prof["metrics"]["alone_net_ms"]["weighted_mean_over_positions"]
    print("Probe Tap cell: isolated tap execution time per call (synthetic-state microbenchmark; not added latency)")
    ok &= check("Isolated tap work, ms per call", iso["estimate"], PAPER["probe_isolated_tap_ms"][0], 5e-6)
    ok &= check("  95% task interval low", iso["ci95"][0], PAPER["probe_isolated_tap_ms"][1], 5e-6)
    ok &= check("  95% task interval high", iso["ci95"][1], PAPER["probe_isolated_tap_ms"][2], 5e-6)
    ratio = iso["estimate"] / prof["weighted_mean_generated_tokens"] * 1000
    ok &= check("  ratio of weighted means, us per token", ratio, PAPER["probe_isolated_tap_us_per_token_ratio"], 5e-4)
    w = {p["run_key"]: 0.0 for p in positions}
    for p in positions:
        w[p["run_key"]] += p["weight"]
    med = statistics.median([a0[p["run_key"]] for p in positions])
    ok &= check("Median A0 time over the 30 positions, s", med, PAPER["table_rounded"]["median_agent_s"], 0.005)

    rows = [{"tap": "Mean probe", "hardware": "RTX PRO 6000", "setup_s": probe_setup, "no_tap_s": no_tap["estimate"],
             "with_tap_s": with_tap["estimate"], "tap_ms": None, "paired_latency_ms": diff}]
    print("\nLLM monitors (judging time of the saved A0 response, added to A0 time)")
    for hw, base, names in MONITORS:
        for name in names:
            d = base / name
            cfg = read(d / "run_config.json")["config"]
            assert cfg["instruction"] == "early"
            assert cfg["instruction_sha256"] == hashlib.sha256(m.EARLY_INSTRUCTION.encode()).hexdigest()
            assert cfg["outputs_sha256"] == sha(GEN / "generate_calls_stage2" / "outputs.jsonl")
            raw = jl(d / "rows.jsonl")
            replies = {(r["item"], r["attempt"]): r["text"] for r in jl(d / "replies.jsonl")}
            assert len(raw) == 24
            secs = {}
            for r in raw:
                assert r["valid_score"] and r["finish_reason"] == "stop" and r["attempt"] == 1
                good, why = reply_valid(replies[r["item"], r["attempt"]], r["finish_reason"],
                                        m.MonitorConfig(**cfg["monitor_config"]))
                assert good and not why
                secs[r["run_key"]] = r["seconds"]
            mon = stat({k: v * 1000 for k, v in secs.items()})
            composed = stat({k: a0[k] + v for k, v in secs.items()})
            startup = jl(d / "sessions.jsonl")[0]["server_info_session"]["seconds_launch_to_health_ok"]
            key = f"{hw}:{name}"
            want = PAPER["table_rounded"]["tap_ms"][key]
            got = [mon["estimate"], *mon["ci95"]]
            same = all(abs(round(g) - x) <= 1 for g, x in zip(got, want))
            s_ok = abs(round(setup["model_load_s"] + startup, 2) - PAPER["table_rounded"]["setup"][key]) <= 0.011
            ok &= same and s_ok
            print(f"  {hw:14s} {SHORT[name]:10s} added {got[0]:9.1f} ms [{got[1]:8.1f}, {got[2]:8.1f}]  paper {want}"
                  f"   setup {setup['model_load_s'] + startup:8.2f} s (paper {PAPER['table_rounded']['setup'][key]})"
                  f"   {'ok' if same and s_ok else 'DIFFERS'}   24/24 valid")
            rows.append({"tap": SHORT[name], "hardware": "RTX PRO 6000" if hw == "rtx" else "second machine",
                         "setup_s": setup["model_load_s"] + startup, "no_tap_s": no_tap["estimate"],
                         "with_tap_s": composed["estimate"], "tap_ms": mon})

    rows[0]["tap_ms"] = {"estimate": iso["estimate"], "ci95": iso["ci95"],
                         "definition": "isolated tap execution time per call (synthetic-state microbenchmark)"}
    lines = [r"\begin{tabular}{lllrrrrrr}", r"\toprule",
             r"Model & Task & Tap & Setup (s) & No tap (s) & With tap (s) & Tap (ms/inf.) & Attestation (s) & Total certification (s) \\",
             r"\midrule"]
    last = None
    for j, r in enumerate(rows):
        if j and r["hardware"] != last:
            lines += [r"\midrule", r"\multicolumn{9}{l}{LLM monitors on " + r["hardware"] + r"} \\"]
            last = r["hardware"]
        digs = 2 if j == 0 else 0
        t = r["tap_ms"]
        cellv = f"{t['estimate']:,.{digs}f} [{t['ci95'][0]:,.{digs}f}, {t['ci95'][1]:,.{digs}f}]"
        lines.append(f"{'Qwen3.8-27B' if j == 0 else ''} & {'LiveCodeBench' if j == 0 else ''} & {r['tap']} & "
                     f"{r['setup_s']:,.2f} & {r['no_tap_s']:,.2f} & {r['with_tap_s']:,.2f} & {cellv} & -- & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (args.out / "eq1_phases.tex").write_text("\n".join(lines))
    (args.out / "eq1_rows.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"\nAttestation and Total certification: not measured (pending the TEE run); shown as '--'.")
    print(f"wrote {args.out / 'eq1_phases.tex'} and eq1_rows.json")
    print("ALL EQ1 CELLS MATCH" if ok else "SOME EQ1 CELLS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
