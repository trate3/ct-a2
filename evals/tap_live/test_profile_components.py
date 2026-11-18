"""CPU-only tests of `profile_components`: schedule, weights, shapes and the weighted aggregation. No GPU, no torch.

Run from the repository root:  python3 -m pytest evals/tap_live/test_profile_components.py --noconftest
(`--noconftest` because the folder's conftest builds a tiny torch model that these tests do not need.)
Tests that read the saved main stage are skipped when its files are not in the checkout.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from evals.tap_live import profile_components as pc

ROOT = Path(__file__).resolve().parents[2]
STAGE2_LIST = ROOT / "results/tap_live/run2_main/run2/stage2_list.json"
CALLS = ROOT / "results/tap_live/run2_main/run2/generate_calls_stage2/calls.jsonl"
have_main = pytest.mark.skipif(not (STAGE2_LIST.exists() and CALLS.exists()), reason="saved main stage not in this checkout")


# ---- buffer growth schedule (follows the hook in live_tap.py) -------------------------------------------------
def test_growth_first_call_grows_from_16_to_4096():
    g = pc.growth_schedule([681])
    assert g[0]["rows_at_start"] == 16
    assert g[0]["growths"] == [{"token_index": 16, "rows_before": 16, "rows_after": 4096}]
    assert g[0]["rows_at_end"] == 4096


def test_growth_doubles_and_is_never_shrunk():
    g = pc.growth_schedule([681, 6896, 879, 23266, 100])
    assert [x["rows_at_end"] for x in g] == [4096, 8192, 8192, 32768, 32768]
    assert [e["rows_after"] for e in g[3]["growths"]] == [16384, 32768]       # one call can grow twice
    assert g[2]["growths"] == [] and g[4]["growths"] == []                    # shorter calls reuse the buffer
    assert g[3]["rows_at_start"] == 8192


def test_growth_boundary_exact_fit_does_not_grow():
    # the hook grows when token index i >= rows: n == rows rows fits, n == rows + 1 grows
    assert pc.growth_schedule([4096])[0]["growths"][0]["token_index"] == 16           # 16 -> 4096 at i = 16
    g = pc.growth_schedule([4096, 4096, 4097])
    assert g[1]["growths"] == []
    assert g[2]["growths"] == [{"token_index": 4096, "rows_before": 4096, "rows_after": 8192}]


def test_scored_tokens_leaves_out_the_stop_token_state():
    assert pc.scored_tokens(681) == 680
    assert pc.scored_tokens(681, last_is_stop=False) == 681


# ---- the saved main stage ------------------------------------------------------------------------------------
@have_main
def test_schedule_shape_and_weights_from_the_saved_main_stage():
    s = pc.load_schedule(STAGE2_LIST, CALLS)
    assert len(s["runs"]) == 24 and len(s["positions"]) == 30
    assert abs(sum(p["weight"] for p in s["positions"]) - 1.0) < 1e-12
    assert all(abs(p["weight"] - 1 / 30) < 1e-12 for p in s["positions"])
    assert len({p["task"] for p in s["positions"]}) == 10
    assert [r["history_id"] for r in s["runs"]][0] == "lcbhard_101@0.25" and s["runs"][0]["n_generated"] == 681
    assert sum(r["n_generated"] for r in s["runs"]) == 76898
    assert min(r["n_generated"] for r in s["runs"]) == 222 and max(r["n_generated"] for r in s["runs"]) == 23266
    runs_by_key = {r["run_key"]: r for r in s["runs"]}
    assert all(p["run_key"] in runs_by_key for p in s["positions"])
    # 6 positions reuse a representative
    assert len(s["positions"]) - len(s["runs"]) == 6


@have_main
def test_weighted_mean_generated_tokens_matches_the_main_table():
    s = pc.load_schedule(STAGE2_LIST, CALLS)
    pv = pc.position_values({r["run_key"]: float(r["n_generated"]) for r in s["runs"]}, s["positions"])
    mean = sum(p["weight"] * p["value"] for p in pv)
    assert round(mean, 1) == 5625.7                                          # main_complete_call: weighted_mean_generated_tokens_a0


@have_main
def test_raw_score_s_field_reproduces_the_known_weighted_mean():
    """The letter's number: raw main-call score_s, weighted by the 30 positions, averages 0.490074 ms, median 0.119949 ms."""
    s = pc.load_schedule(STAGE2_LIST, CALLS)
    pv = pc.position_values({r["run_key"]: 1e3 * r["main_stage_score_s"] for r in s["runs"]}, s["positions"])
    assert round(sum(p["weight"] * p["value"] for p in pv), 6) == 0.490074
    vals = [p["value"] for p in pv]
    assert round(statistics.median(vals), 6) == 0.119949
    assert round(min(vals), 6) == 0.108405 and round(max(vals), 6) == 2.259373


@have_main
def test_growth_events_of_the_main_stage_sequence():
    s = pc.load_schedule(STAGE2_LIST, CALLS)
    g = pc.growth_schedule([r["n_generated"] for r in s["runs"]])
    assert g[0]["rows_at_start"] == 16 and g[0]["rows_at_end"] == 4096          # first real call grows the warm-up buffer
    assert g[-1]["rows_at_end"] == 32768                                          # the 23,266-token call forces 32,768 rows
    # no call after the buffer reached its maximum grows it again
    last = max(i for i, x in enumerate(g) if x["growths"])
    assert all(not x["growths"] for x in g[last + 1:])
    # the rows at the end of one call are the rows at the start of the next
    assert all(g[i]["rows_at_end"] == g[i + 1]["rows_at_start"] for i in range(len(g) - 1))


# ---- aggregation (synthetic) ---------------------------------------------------------------------------------
def _positions(n_tasks=3, per_task=2):
    pos, runs = [], []
    for t in range(n_tasks):
        for d in range(per_task):
            k = f"r{t}_{d}"
            pos.append({"history_id": f"t{t}@{d}", "task": f"t{t}", "weight": 1.0 / (n_tasks * per_task), "run_key": k})
    return pos


def test_position_values_reuse_the_representative():
    pos = _positions()
    values = {p["run_key"]: float(i) for i, p in enumerate(pos)}
    pos[1]["run_key"] = pos[0]["run_key"]                                       # position 1 shares position 0's run
    pv = pc.position_values(values, pos)
    assert pv[0]["value"] == pv[1]["value"] == 0.0


def test_weighted_mean_and_bootstrap_conventions():
    pos = _positions()
    per_run = {p["run_key"]: float(i + 1) for i, p in enumerate(pos)}
    r = pc.weighted_mean_ci(pc.position_values(per_run, pos))
    assert r["estimate"] == pytest.approx(sum(per_run.values()) / 6)
    lo, hi = r["ci95"]
    assert lo <= r["estimate"] <= hi
    # constant values: the interval collapses on the constant
    c = pc.weighted_mean_ci(pc.position_values({p["run_key"]: 2.5 for p in pos}, pos))
    assert c["estimate"] == pytest.approx(2.5) and c["ci95"][0] == pytest.approx(2.5) and c["ci95"][1] == pytest.approx(2.5)
    # identical to the existing tables' helper (tasks sorted, 2000 draws, seed 0)
    from evals.tap_live import summarize_decode as sd
    parts = {}
    for p in pc.position_values(per_run, pos):
        s_, n_ = parts.get(p["task"], (0.0, 0.0))
        parts[p["task"]] = (s_ + p["weight"] * p["value"], n_ + p["weight"])
    assert sd.task_bootstrap(parts)["ci95"] == r["ci95"]


def test_unequal_weights_are_respected():
    items = [{"task": "a", "weight": 0.75, "value": 4.0}, {"task": "b", "weight": 0.25, "value": 8.0}]
    assert pc.weighted_mean_ci(items)["estimate"] == pytest.approx(5.0)


def test_per_run_values_skip_warmups_and_none():
    rows = [{"run_key": "x", "warmup": True, "v": 100.0}, {"run_key": "x", "warmup": False, "v": 1.0},
            {"run_key": "x", "warmup": False, "v": 3.0}, {"run_key": "x", "warmup": False, "v": 11.0},
            {"run_key": "y", "warmup": False, "v": None}]
    assert pc.per_run_values(rows, "v", "mean") == {"x": 5.0}
    assert pc.per_run_values(rows, "v", "median") == {"x": 3.0}


def test_paired_rows_subtract_the_baseline_of_the_same_repeat_and_call():
    rows = [{"rep": 0, "run_index": 0, "alone_ms": 5.0}, {"rep": 1, "run_index": 0, "alone_ms": 7.0}]
    base = [{"rep": 0, "run_index": 0, "loop_ms": 2.0}, {"rep": 1, "run_index": 0, "loop_ms": 6.0}]
    out = pc.paired_rows(rows, base, {"net": ("alone_ms", "loop_ms")})
    assert [r["net"] for r in out] == [3.0, 1.0]


def test_compose_adds_and_subtracts_per_run():
    assert pc.compose([(1, {"a": 1.0, "b": 2.0}), (1, {"a": 0.5, "b": 0.5}), (-1, {"a": 0.25, "b": 0.0})]) == {"a": 1.25, "b": 2.5}


def _synthetic_raw(sched, reps=4):
    raw = {k: [] for k in ("0a", "A", "0b", "B", "C", "P")}
    for rep in range(-1, reps):
        for k, r in enumerate(sched["runs"]):
            key = {"rep": rep, "warmup": rep < 0, "run_index": k, "run_key": r["run_key"], "n_generated": r["n_generated"]}
            n = r["n_generated"]
            raw["0a"].append({**key, "loop_ms": 0.002 * n})
            raw["A"].append({**key, "alone_ms": 0.002 * n + 0.01 * n + 0.5, "score_s_ms": 0.3})
            raw["0b"].append({**key, "hook_cpu_ms": 0.0025 * n, "growth_cpu_ms": 0.0, "growth_wait_ms": 0.01})
            raw["B"].append({**key, "begin_ms": 0.02, "hook_cpu_ms": 0.0125 * n, "growth_cpu_ms": 0.1, "growth_wait_ms": 0.51, "drain_ms": 0.05,
                             "end_call_ms": 0.6, "score_head_ms": 0.4, "n_growth": 1, "split_scoring_ms": 0.1, "split_pooling_ms": 0.1,
                             "split_readback_ms": 0.1})
            raw["C"].append({**key, "gpu_capture_raw_ms": 0.004 * n, "gpu_growth_span_ms": 1.0, "n_hooks_no_growth": n - 1, "n_growth": 1,
                             "max_single_hook_event_ms": 0.01, "enqueue_loop_ms": 1.0})
            raw["P"].append({**key, "device_total_ms": 0.001 * n + 0.2, "step_kernel_count": n, "step_kernel_ms": 0.001 * n, "other_device_ms": 0.2})
    return raw


def test_build_tables_on_synthetic_rows_gives_positive_consistent_totals():
    sched = {"runs": [{"run_key": f"r{t}_{d}", "n_generated": 100 * (1 + t + d)} for t in range(3) for d in range(2)], "positions": _positions()}
    t = pc.build_tables(_synthetic_raw(sched), sched, floor_ms=0.001)
    m = t["metrics"]
    # alone_net = 0.01 * n + 0.5 for every repeat; weighted over the six equal-weight positions
    ns = [100, 200, 200, 300, 300, 400]
    expect = sum(0.01 * n + 0.5 for n in ns) / 6
    assert m["alone_net_ms"]["weighted_mean_over_positions"]["estimate"] == pytest.approx(expect)
    # pass B partition: begin + hook_net + growth_cpu_net + growth_wait_net + drain + end_call
    exp_b = sum(0.02 + 0.01 * n + 0.1 + 0.50 + 0.05 + 0.6 for n in ns) / 6
    assert m["partition_total_ms"]["weighted_mean_over_positions"]["estimate"] == pytest.approx(exp_b)
    # end bookkeeping = end_call - score_head
    assert m["end_bookkeeping_ms"]["weighted_mean_over_positions"]["estimate"] == pytest.approx(0.2)
    # device-bound total: begin + (gpu capture raw - floor * hooks) + growth cpu + growth wait + end_call
    exp_d = sum(0.02 + (0.004 * n - 0.001 * (n - 1)) + 0.1 + 0.50 + 0.6 for n in ns) / 6
    assert m["device_bound_total_ms"]["weighted_mean_over_positions"]["estimate"] == pytest.approx(exp_d)
    # kernel-based device view: begin + CUPTI step kernels + growth cpu + growth wait + end_call
    exp_k = sum(0.02 + 0.001 * n + 0.1 + 0.50 + 0.6 for n in ns) / 6
    assert m["device_kernel_total_ms"]["weighted_mean_over_positions"]["estimate"] == pytest.approx(exp_k)
    assert m["device_kernel_total_ms"]["weighted_mean_over_positions"]["estimate"] < m["device_bound_total_ms"]["weighted_mean_over_positions"]["estimate"] < m["alone_net_ms"]["weighted_mean_over_positions"]["estimate"] + 1
    for name in ("alone_net_ms", "partition_total_ms", "device_bound_total_ms", "device_kernel_total_ms"):
        assert m[name]["weighted_mean_over_positions"]["estimate"] > 0
    # warm-up repeats are excluded
    assert t["n_rows"]["A"] == 6 * 5
    assert t["weighted_mean_generated_tokens"] == pytest.approx(sum(ns) / 6)


def test_summarize_writes_tables_from_raw_files(tmp_path):
    sched = {"runs": [{"run_key": f"r{t}_{d}", "n_generated": 100 * (1 + t + d), "index": 0} for t in range(3) for d in range(2)], "positions": _positions()}
    (tmp_path / "schedule.json").write_text(json.dumps(sched))
    for k, rows in _synthetic_raw(sched).items():
        (tmp_path / f"raw_pass_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "timer_overhead.json").write_text(json.dumps({"event_floor_pair_around_standin_call_no_hook_ms": {"mean": 0.001}}))
    t = pc.summarize(tmp_path)
    assert (tmp_path / "summary.json").exists() and (tmp_path / "per_position.csv").exists()
    assert t["event_floor_ms_used"] == 0.001
    assert len((tmp_path / "per_position.csv").read_text().strip().splitlines()) == 7
    assert "device_kernel_total_ms_mean" in (tmp_path / "per_position.csv").read_text()
