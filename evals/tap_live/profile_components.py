"""Component microbenchmark of the Qwen mean-probe tap: what the tap itself executes per agent call.

It replays the tap's work at the main-call shapes without model generation: the decoder block's output is replaced
by a stand-in bf16 tensor [1, 1, d] (values drawn from the probe's own mean/sigma). It measures the tap's own work per
call, not end-to-end latency.

What is replayed (no change to any tap code, `live_tap.LiveProbeTap` is used as it is):
  * the 24 distinct main-stage calls (stage-2 list, in the order the main stage ran them), each with the
    n_generated of the saved main stage. Every forward pass of a call fires the tap's real forward hook on a
    stand-in decoder block (real `tap.attach`, real hook, so the module-hook dispatch is included); at the end the
    real `begin_call` / `end_call` run with the same arguments as `decode_core.decode_run` (call-end scoring,
    scope "call", float32 buffer, bf16 hidden states, layer 48, the exact probe artifact);
  * the buffer reuse and growth schedule of the main stage: the buffer persists across calls, starts with the 16 rows
    the 16-token warm-up call left, grows by doubling (to at least 4096 rows) when a call is longer than it holds,
    `empty_cache()` after each call, as the main stage did. Each repeat replays the whole 24-call sequence from that
    starting state, so each call sees the same buffer size and the same growth events as in the main stage;
  * the 30 weighted positions (weight 1/30 each, 10 tasks) take their prompt group's representative call.

Passes (each repeat runs each pass over the whole sequence; the order of A and B alternates with the repeat):
  0a/0b  baseline: the same loops with the hook detached (stand-in block call only). Not tap work. It is subtracted,
         repeat by repeat, from A and B so that the tap's own cost is reported net of the stand-in block call and
         of the loop timers.
  A      "tap alone, one clock": synchronize, perf_counter, begin_call, one hook call per generated token (growth
         included), end_call (scoring, read-back, decision), synchronize, perf_counter. No timer inside the call.
         One interval, so CPU enqueue work and GPU work are never added.
  B      the same work split on the CPU clock into consecutive, non-overlapping intervals: begin_call; hook calls
         without growth; hook calls that grow the buffer (the call plus an immediate synchronization that waits for
         the zeros / old-buffer copy on the device); a synchronization that drains whatever the hooks left on the
         device; end_call (its own `score_s` is the `_score_head` part, the rest is end-of-call bookkeeping). After
         end_call a replica of `_score_head` is run once on the same buffer to split scoring / pooling / read-back
         (a synchronization after each part; used only for the split, checked against the real score).
  C      device time of the per-step capture: a short sleep kernel is queued before every hook call so the CPU is
         ahead of the device, and CUDA events wrap the whole hook call (growth, zeros, old-buffer copy, the row copy
         and the float32 conversion). The empty event pair (floor) is measured the same way and reported; raw and
         floor-corrected sums are both given. Growth calls are listed apart (their span also contains allocator CPU time).
  P      device kernel time from the CUDA profiler (CUPTI) over begin_call and all hook calls of each call: the time the
         tap's kernels really run (the per-step float32 conversion copy, plus the growth zeros / old-buffer copy), without
         event or launch-gap effects. The events of pass C measure a larger span because they include the launch gap
         around each tiny kernel; both are reported.

Timer overhead is measured and reported (perf_counter pair, event pair, synchronization on an idle device, a minimal
read-back, the stand-in block call with and without the hook).

  python -m evals.tap_live.profile_components --stage2-list L.json --calls calls.jsonl --artifact PROBE.json --out OUT
  python -m evals.tap_live.profile_components --out OUT --summarize-only        # recompute the tables from the raw files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

LAYER = 48
WARMUP_CALL_ROWS = 16          # the 16-token warm-up call of the main stage left a 16-row buffer
INITIAL_ROWS = 4096            # live_tap.INITIAL_ROWS (checked against the module when the GPU part runs)
N_FAKE_LAYERS = 64
SLEEP_US = 80.0                # sleep kernel queued before every hook call in pass C


# ----------------------------------------------------------------------------------------------------------------
# schedule, weights, aggregation: plain Python, no GPU
# ----------------------------------------------------------------------------------------------------------------
def load_schedule(stage2_list: Path, calls: Path) -> dict:
    """Runs in main-stage order (n_generated, prompt tokens, max_new_tokens from the saved A1 calls, equal to A0's) and the
    30 weighted positions. Raises if anything needed is missing or inconsistent."""
    sel = json.loads(Path(stage2_list).read_text())
    rows = [json.loads(x) for x in Path(calls).read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        if r.get("status") != "ok":
            raise ValueError(f"call row not ok: {r.get('history_id')} {r.get('arm')}")
        by[(r["run_key"], r["arm"])] = r
    runs = []
    for k, run in enumerate(sel["runs"]):
        a0, a1 = by[(run["run_key"], "A0")], by[(run["run_key"], "A1")]
        if a0["n_generated"] != a1["n_generated"]:
            raise ValueError(f"A0 and A1 lengths differ for {run['history_id']}")
        if not a1["stopped_naturally"]:
            raise ValueError(f"{run['history_id']} did not stop naturally")
        runs.append({"index": k, "run_key": run["run_key"], "history_id": run["history_id"], "prompt_tokens": a1["prompt_tokens"],
                     "n_generated": int(a1["n_generated"]), "max_new_tokens": int(a1["max_new_tokens"]),
                     "stopped_naturally": True, "main_stage_score_s": a1.get("tap_score_s")})
    positions = [{"history_id": p["history_id"], "task": p["task"], "weight": float(p["weight"]), "run_key": p["run_key"]}
                 for p in sel["positions"]]
    keys = {r["run_key"] for r in runs}
    if any(p["run_key"] not in keys for p in positions):
        raise ValueError("a position points to a run that is not in the list")
    if abs(sum(p["weight"] for p in positions) - 1.0) > 1e-9:
        raise ValueError("position weights do not sum to 1")
    return {"runs": runs, "positions": positions, "n_positions": len(positions)}


def scored_tokens(n_generated: int, last_is_stop: bool = True) -> int:
    """Tokens scored by end_call: the state that produced the stop token is left out (as live_tap does)."""
    return n_generated - 1 if last_is_stop else n_generated


def growth_schedule(n_list: list[int], start_rows: int = WARMUP_CALL_ROWS, initial_rows: int = INITIAL_ROWS) -> list[dict]:
    """Buffer rows at the start of each call and the growth events inside it, following live_tap's hook exactly:
    growth happens when token index i >= rows, new rows = max(2 * rows, initial_rows); the buffer is never shrunk."""
    rows, out = start_rows, []
    for n in n_list:
        start, events = rows, []
        for i in range(n):
            if i >= rows:
                new = max(2 * rows, initial_rows)
                events.append({"token_index": i, "rows_before": rows, "rows_after": new})
                rows = new
        out.append({"rows_at_start": start, "growths": events, "rows_at_end": rows})
    return out


def position_values(per_run: dict[str, float], positions: list[dict]) -> list[dict]:
    """Every position takes the value of its group's representative run."""
    return [{"history_id": p["history_id"], "task": p["task"], "weight": p["weight"], "value": per_run[p["run_key"]]} for p in positions]


def weighted_mean_ci(items: list[dict]) -> dict:
    """Weighted mean over positions with the 95 % percentile interval of the task bootstrap of the existing tables
    (`summarize_decode.task_bootstrap`: 2,000 draws, seed 0, a task's positions kept together)."""
    from evals.tap_live import summarize_decode as sd
    parts = defaultdict(lambda: [0.0, 0.0])
    for it in items:
        parts[it["task"]][0] += it["weight"] * it["value"]
        parts[it["task"]][1] += it["weight"]
    b = sd.task_bootstrap({k: tuple(v) for k, v in parts.items()})
    return {"estimate": b["estimate"], "ci95": b["ci95"]}


def stats(v: list[float]) -> dict:
    s = sorted(v)
    return {"n": len(s), "median": statistics.median(s), "mean": statistics.fmean(s), "min": s[0], "max": s[-1],
            "p05": s[int(0.05 * (len(s) - 1))], "p95": s[int(math.ceil(0.95 * (len(s) - 1)))]}


def per_run_values(rows: list[dict], field: str, stat: str = "mean") -> dict[str, float]:
    """{run_key: statistic over repeats of `field`} for the measured (non-warm-up) rows."""
    by = defaultdict(list)
    for r in rows:
        if not r.get("warmup") and r.get(field) is not None:
            by[r["run_key"]].append(r[field])
    return {k: stats(v)[stat] for k, v in by.items()}


def paired_rows(rows: list[dict], base: list[dict], pairs: dict[str, str]) -> list[dict]:
    """Rows with the extra fields {new: row[field] - base_row[field]} where base_row is the baseline row of the same
    repeat and call (pass 0a for A, pass 0b for B)."""
    bk = {(r["rep"], r["run_index"]): r for r in base}
    out = []
    for r in rows:
        b = bk.get((r["rep"], r["run_index"]))
        if b is None:
            continue
        x = dict(r)
        for new, (field, bfield) in pairs.items():
            x[new] = r[field] - b[bfield]
        out.append(x)
    return out


def compose(per_run_terms: list[tuple[float, dict[str, float]]]) -> dict[str, float]:
    """Sum of weighted per-run values: [(+1/-1, {run_key: value}), ...]."""
    keys = set.intersection(*(set(t.keys()) for _, t in per_run_terms))
    return {k: sum(s * t[k] for s, t in per_run_terms) for k in keys}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def build_tables(raw: dict[str, list[dict]], sched: dict, floor_ms: float | None) -> dict:
    """All derived tables from the raw rows (milliseconds per call). `raw` keys: "0a", "A", "0b", "B", "C"."""
    positions = sched["positions"]
    A = paired_rows(raw["A"], raw["0a"], {"alone_net_ms": ("alone_ms", "loop_ms")})
    B = paired_rows(raw["B"], raw["0b"], {"hook_cpu_net_ms": ("hook_cpu_ms", "hook_cpu_ms"),
                                          "growth_cpu_net_ms": ("growth_cpu_ms", "growth_cpu_ms"),
                                          "growth_wait_net_ms": ("growth_wait_ms", "growth_wait_ms")})
    for r in B:
        r["partition_total_ms"] = (r["begin_ms"] + r["hook_cpu_net_ms"] + r["growth_cpu_net_ms"] + r["growth_wait_net_ms"]
                                   + r["drain_ms"] + r["end_call_ms"])
        r["end_bookkeeping_ms"] = r["end_call_ms"] - r["score_head_ms"]
        r["split_parts_sum_ms"] = r["split_scoring_ms"] + r["split_pooling_ms"] + r["split_readback_ms"]
    C = [dict(r) for r in raw.get("C", [])]
    P = [dict(r) for r in raw.get("P", [])]
    for r in C:
        r["gpu_capture_net_ms"] = r["gpu_capture_raw_ms"] - (floor_ms or 0.0) * r["n_hooks_no_growth"]
    metrics = {  # name -> (rows, field, description)
        "alone_net_ms": (A, "alone_net_ms", "HEADLINE: all tap operations of a call run alone, one clock, net of the stand-in block call"),
        "alone_raw_ms": (A, "alone_ms", "same interval before subtracting the baseline loop (includes the stand-in block call)"),
        "begin_ms": (B, "begin_ms", "begin_call (per-call reset; no allocation after the first call of the run)"),
        "hook_cpu_net_ms": (B, "hook_cpu_net_ms", "hook calls without growth: CPU time (Python bookkeeping, indexing, launch), net of the stand-in call"),
        "growth_cpu_net_ms": (B, "growth_cpu_net_ms", "hook calls that grow the buffer: CPU time (allocation, zeros and copy launches)"),
        "growth_wait_net_ms": (B, "growth_wait_net_ms", "wait for the growth zeros / old-buffer copy on the device"),
        "drain_ms": (B, "drain_ms", "device work left over by the hook calls after the CPU finished enqueueing"),
        "end_call_ms": (B, "end_call_ms", "end_call: whole call-end scoring path (scoring, pooling, read-back, decision, bookkeeping)"),
        "score_head_ms": (B, "score_head_ms", "of which _score_head (the live score_s): scoring, pooling and read-back"),
        "end_bookkeeping_ms": (B, "end_bookkeeping_ms", "of which end-of-call bookkeeping outside _score_head"),
        "split_scoring_ms": (B, "split_scoring_ms", "replica split: normalise, dot, bias (enqueue + execution)"),
        "split_pooling_ms": (B, "split_pooling_ms", "replica split: sum, mean, roll-up, stack"),
        "split_readback_ms": (B, "split_readback_ms", "replica split: read-back to the host"),
        "partition_total_ms": (B, "partition_total_ms", "sum of the consecutive CPU-clock intervals of pass B"),
        "gpu_capture_raw_ms": (C, "gpu_capture_raw_ms", "device time of the hook calls (events, raw), no-growth hooks"),
        "gpu_capture_net_ms": (C, "gpu_capture_net_ms", "same minus the empty-event-pair floor"),
        "gpu_growth_span_ms": (C, "gpu_growth_span_ms", "event span of the growth hook calls (includes allocator CPU time)"),
        "kernel_step_ms": (P, "step_kernel_ms", "device kernel time of the per-step capture copy (CUPTI profiler)"),
        "kernel_growth_ms": (P, "other_device_ms", "device kernel time of the buffer growth (zeros, old-buffer copy; CUPTI)"),
        "kernel_total_ms": (P, "device_total_ms", "device kernel time of begin_call and all hook calls (CUPTI)"),
    }
    metrics = {k: v for k, v in metrics.items() if v[0]}
    per_run = {name: {m: per_run_values(rows, f, m) for m in ("mean", "median")} for name, (rows, f, _) in metrics.items()}
    # device-bound total: capture on the device (hook CPU time hidden behind other device work), growth, begin, end_call
    per_run["device_bound_total_ms"] = {"mean": compose([(1, per_run["begin_ms"]["mean"]), (1, per_run["gpu_capture_net_ms"]["mean"]),
                                                          (1, per_run["growth_cpu_net_ms"]["mean"]), (1, per_run["growth_wait_net_ms"]["mean"]),
                                                          (1, per_run["end_call_ms"]["mean"])])}
    metrics["device_bound_total_ms"] = (None, None, "COMPANION (device view, upper): begin + event span of capture (floor-corrected) + growth + end_call (hook CPU time not added)")
    if P:
        per_run["device_kernel_total_ms"] = {"mean": compose([(1, per_run["begin_ms"]["mean"]), (1, per_run["kernel_step_ms"]["mean"]),
                                                               (1, per_run["growth_cpu_net_ms"]["mean"]), (1, per_run["growth_wait_net_ms"]["mean"]),
                                                               (1, per_run["end_call_ms"]["mean"])])}
        metrics["device_kernel_total_ms"] = (None, None, "COMPANION (device view, lower): begin + kernel time of the per-step copy (CUPTI) + growth + end_call (hook CPU time not added)")
    out = {"metrics": {}, "per_position": {}}
    for name in per_run:
        d = {"description": metrics[name][2]}
        for stat in per_run[name]:
            pv = position_values(per_run[name][stat], positions)
            d[f"weighted_{stat}_over_positions"] = weighted_mean_ci(pv)
        out["metrics"][name] = d
        out["per_position"][name] = {stat: {p["history_id"]: p["value"] for p in position_values(per_run[name][stat], positions)}
                                     for stat in per_run[name]}
    # the same position-weighted mean, with the position tokens, so the scale is visible
    tok = {r["run_key"]: float(r["n_generated"]) for r in sched["runs"]}
    out["weighted_mean_generated_tokens"] = weighted_mean_ci(position_values(tok, positions))["estimate"]
    # within-position spread over repeats for the headline (so noise of the measurement is visible)
    spread = defaultdict(list)
    for r in A:
        if not r.get("warmup"):
            spread[r["run_key"]].append(r["alone_net_ms"])
    out["headline_repeat_spread_by_run"] = {k: stats(v) for k, v in spread.items()}
    out["n_rows"] = {k: len(v) for k, v in raw.items()}
    return out


# ----------------------------------------------------------------------------------------------------------------
# GPU part (torch imported here, never at module import)
# ----------------------------------------------------------------------------------------------------------------
def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _env() -> dict:
    import torch
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "host": platform.node()}
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.total",
                                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:     # noqa: BLE001
        info["nvidia_smi"] = f"unavailable: {exc}"
    return info


def make_fake_model(torch, d: int, mu, sigma, device):
    """A stand-in model: `model.layers` of 64 trivial blocks (so the real attach finds layer 48) whose block returns a
    pre-made bf16 [1, 1, d] tensor. Only block L-1 is ever called."""
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    pool = [(mu.cpu() + sigma.cpu() * torch.randn(d, generator=g)).to(torch.bfloat16).to(device).view(1, 1, d).contiguous() for _ in range(64)]

    class Block(torch.nn.Module):
        def __init__(self, pool=None):
            super().__init__()
            self.pool = pool

        def forward(self, i):
            return self.pool[i & 63]

    class Cfg:
        num_hidden_layers = N_FAKE_LAYERS

        def get_text_config(self):
            return self

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(pool if k == LAYER - 1 else None) for k in range(N_FAKE_LAYERS)])

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.model = Inner()

    return Fake()


class Bench:
    def __init__(self, artifact: Path, sched: dict, device: str = "cuda"):
        import torch
        from evals.tap_live import live_tap
        assert live_tap.INITIAL_ROWS == INITIAL_ROWS, "live_tap.INITIAL_ROWS changed; the growth schedule would be wrong"
        self.torch, self.live_tap = torch, live_tap
        self.device = torch.device(device)
        self.artifact = Path(artifact)
        self.sched = sched
        self.runs = sched["runs"]
        self.growth = growth_schedule([r["n_generated"] for r in self.runs])
        self.gsets = [frozenset(e["token_index"] for e in g["growths"]) for g in self.growth]
        self.sync = lambda: torch.cuda.synchronize(self.device)
        self.tap = None
        self.fake = None
        self.new_tap()

    # -- state
    def new_tap(self):
        """A fresh tap in the state the main stage's warm-up call left: a 16-row buffer, nothing else."""
        torch = self.torch
        if self.tap is not None:
            self.tap.detach()
        self.tap = None
        self.tap = self.live_tap.LiveProbeTap(self.artifact, device=self.device, scope="call", time_capture=False, name="A1")
        head = next(iter(self.tap.heads.values()))
        self.head = head
        if self.fake is None:
            self.fake = make_fake_model(torch, head.d, head.mu, head.sigma, self.device)
        self.layer = self.fake.model.layers[LAYER - 1]
        # the warm-up call of the main stage (16 tokens, up to 16 new tokens): leaves the 16-row buffer
        with torch.no_grad():
            self.tap.attach(self.fake)
            self.tap.begin_episode()
            self.tap.begin_call(WARMUP_CALL_ROWS, 0)
            for i in range(WARMUP_CALL_ROWS):
                self.layer(i)
            self.tap.end_call(n_generated=WARMUP_CALL_ROWS, last_is_stop=False, call_index=0, record={})
            self.tap.detach()
        torch.cuda.empty_cache()
        assert self.tap.bufs[LAYER].shape[0] == WARMUP_CALL_ROWS

    def check_buffer(self, k: int):
        rows = self.tap.bufs[LAYER].shape[0]
        if rows != self.growth[k]["rows_at_end"]:
            raise RuntimeError(f"call {k}: buffer has {rows} rows, schedule says {self.growth[k]['rows_at_end']}")

    # -- passes
    def baseline_a(self, k, run):
        """0a: same loop as A without the hook."""
        n, pc, layer, sync = run["n_generated"], time.perf_counter, self.layer, self.sync
        self.tap.detach()
        sync()
        t0 = pc()
        for i in range(n):
            layer(i)
        sync()
        return {"loop_ms": (pc() - t0) * 1e3}

    def baseline_b(self, k, run):
        """0b: same loop as B (per-call timers, growth-call branch) without the hook."""
        n, pc, layer, sync, gset = run["n_generated"], time.perf_counter, self.layer, self.sync, self.gsets[k]
        self.tap.detach()
        sync()
        cpu_ng = cpu_g = wait = 0.0
        for i in range(n):
            a = pc()
            layer(i)
            b = pc()
            if i in gset:
                sync()
                c = pc()
                cpu_g += b - a
                wait += c - b
            else:
                cpu_ng += b - a
        return {"hook_cpu_ms": cpu_ng * 1e3, "growth_cpu_ms": cpu_g * 1e3, "growth_wait_ms": wait * 1e3}

    def pass_a(self, k, run):
        tap, n, pc, layer, sync = self.tap, run["n_generated"], time.perf_counter, self.layer, self.sync
        tap.attach(self.fake)
        tap.begin_episode()
        rec = {}
        sync()
        t0 = pc()
        tap.begin_call(run["max_new_tokens"], 0)
        for i in range(n):
            layer(i)
        tap.end_call(n_generated=n, last_is_stop=True, call_index=0, record=rec)
        sync()
        t1 = pc()
        tap.detach()
        self.check_buffer(k)
        return {"alone_ms": (t1 - t0) * 1e3, "score_s_ms": rec["score_s"] * 1e3}

    def pass_b(self, k, run):
        torch, tap, n, pc, layer, sync, gset = self.torch, self.tap, run["n_generated"], time.perf_counter, self.layer, self.sync, self.gsets[k]
        tap.attach(self.fake)
        tap.begin_episode()
        rec = {}
        sync()
        t = pc()
        tap.begin_call(run["max_new_tokens"], 0)
        begin = pc() - t
        cpu_ng = cpu_g = wait = 0.0
        for i in range(n):
            a = pc()
            layer(i)
            b = pc()
            if i in gset:
                sync()
                c = pc()
                cpu_g += b - a
                wait += c - b
            else:
                cpu_ng += b - a
        t = pc()
        sync()
        drain = pc() - t
        t = pc()
        tap.end_call(n_generated=n, last_is_stop=True, call_index=0, record=rec)
        end_call = pc() - t
        real_score = tap.decisions[-1]["score_call"]
        m = scored_tokens(n)
        split = self.split_score(m, real_score)
        tap.detach()
        self.check_buffer(k)
        return {"begin_ms": begin * 1e3, "hook_cpu_ms": cpu_ng * 1e3, "growth_cpu_ms": cpu_g * 1e3, "growth_wait_ms": wait * 1e3,
                "drain_ms": drain * 1e3, "end_call_ms": end_call * 1e3, "score_head_ms": rec["score_s"] * 1e3,
                "n_growth": len(gset), **split}

    def split_score(self, m: int, real_score: float) -> dict:
        """Replica of `_score_head` for the token-mean head on the same buffer, split in three parts, each ended by a
        synchronization. Used only for the split; its score is compared with the real one."""
        torch, tap, head, pc, sync = self.torch, self.tap, self.head, time.perf_counter, self.sync
        art = head.art
        assert art.pooling == "token_mean"
        head.reset_episode()
        with torch.no_grad():
            sync()
            t0 = pc()
            s = tap._token_scores(head, tap.bufs[art.layer][:m])
            sync()
            t1 = pc()
            first0 = s[0] if art.ema_init == "first" else torch.zeros((), dtype=s.dtype, device=s.device)    # as _score_head does
            tot = s.sum()
            call_t = tot / m
            head.r_sum = tot
            head.r_n = m
            roll_t = head.r_sum / head.r_n
            vals = torch.stack([call_t, roll_t])
            sync()
            t2 = pc()
            out = vals.tolist()
            t3 = pc()
        head.reset_episode()
        if abs(out[0] - real_score) > 1e-4 * max(1.0, abs(real_score)):
            raise RuntimeError(f"replica score {out[0]} differs from the real score {real_score}")
        return {"split_scoring_ms": (t1 - t0) * 1e3, "split_pooling_ms": (t2 - t1) * 1e3, "split_readback_ms": (t3 - t2) * 1e3}

    def pass_c(self, k, run, pairs, sleep_cycles):
        torch, tap, n, layer, sync, gset = self.torch, self.tap, run["n_generated"], self.layer, self.sync, self.gsets[k]
        slp = torch.cuda._sleep
        tap.attach(self.fake)
        tap.begin_episode()
        tap.begin_call(run["max_new_tokens"], 0)
        t0 = time.perf_counter()
        for i in range(n):
            slp(sleep_cycles)
            a, z = pairs[i]
            a.record()
            layer(i)
            z.record()
        enqueue_ms = (time.perf_counter() - t0) * 1e3
        sync()
        ng = g = 0.0
        worst = 0.0
        for i in range(n):
            e = pairs[i][0].elapsed_time(pairs[i][1])
            if i in gset:
                g += e
            else:
                ng += e
                worst = max(worst, e)
        tap.end_call(n_generated=n, last_is_stop=True, call_index=0, record={})
        tap.detach()
        self.check_buffer(k)
        return {"gpu_capture_raw_ms": ng, "gpu_growth_span_ms": g, "n_hooks_no_growth": n - len(gset), "n_growth": len(gset),
                "max_single_hook_event_ms": worst, "enqueue_loop_ms": enqueue_ms}

    def make_pairs(self, n):
        torch = self.torch
        return [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]

    def calibrate_sleep(self) -> int:
        torch = self.torch
        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        cyc = 1_000_000
        for _ in range(3):
            self.sync()
            a.record()
            torch.cuda._sleep(cyc)
            z.record()
            self.sync()
        ms = a.elapsed_time(z)
        return int(cyc * SLEEP_US / (ms * 1e3))

    # -- timer overhead
    def timer_overhead(self, sleep_cycles: int) -> dict:
        torch, pc, sync, layer = self.torch, time.perf_counter, self.sync, self.layer
        out = {"sleep_us_target": SLEEP_US, "sleep_cycles": sleep_cycles}
        n = 1_000_000
        t = pc()
        for _ in range(n):
            pc()
        out["perf_counter_call_ns"] = (pc() - t) / n * 1e9
        t = pc()
        for _ in range(n):
            a = pc()
            pc() - a
        out["perf_counter_pair_with_subtraction_ns"] = (pc() - t) / n * 1e9
        sync()
        ts = []
        for _ in range(2000):
            t = pc()
            sync()
            ts.append(pc() - t)
        out["synchronize_idle_us"] = {"median": statistics.median(ts) * 1e6, "mean": statistics.fmean(ts) * 1e6}
        x = torch.zeros(2, device=self.device)
        ts = []
        for _ in range(2000):
            t = pc()
            x.tolist()
            ts.append(pc() - t)
        out["readback_tolist_2_elements_idle_us"] = {"median": statistics.median(ts) * 1e6, "mean": statistics.fmean(ts) * 1e6}
        self.tap.detach()
        n = 20000
        pairs = self.make_pairs(n)
        sync()
        t = pc()
        for a, z in pairs:
            a.record()
            z.record()
        out["event_pair_record_cpu_us"] = (pc() - t) / n * 1e6
        sync()
        slp = torch.cuda._sleep
        for variant in ("empty_pair", "pair_around_standin_call_no_hook"):
            sync()
            for i in range(n):
                slp(sleep_cycles)
                a, z = pairs[i]
                a.record()
                if variant != "empty_pair":
                    layer(i)
                z.record()
            sync()
            e = sorted(a.elapsed_time(z) for a, z in pairs)
            out[f"event_floor_{variant}_ms"] = {"mean": statistics.fmean(e), "median": e[len(e) // 2], "p99": e[int(0.99 * (len(e) - 1))], "max": e[-1], "n": n}
        # stand-in block call with and without the hook (per call, CPU)
        m = 30000
        sync()
        t = pc()
        for i in range(m):
            layer(i)
        base = (pc() - t) / m
        self.tap.attach(self.fake)
        self.tap.begin_episode()
        self.tap.begin_call(m, 0)
        sync()
        t = pc()
        for i in range(m):
            layer(i)
        hooked = (pc() - t) / m
        sync()
        self.tap.abort_call()
        self.tap.detach()
        out["standin_block_call_no_hook_us"] = base * 1e6
        out["standin_block_call_with_hook_us_incl_growth_amortised"] = hooked * 1e6
        self.new_tap()
        return out

    # -- device kernel time (CUPTI profiler)
    def pass_p(self, k, run):
        torch, tap, n, layer, sync = self.torch, self.tap, run["n_generated"], self.layer, self.sync
        from torch.profiler import ProfilerActivity, profile
        tap.attach(self.fake)
        tap.begin_episode()
        sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            tap.begin_call(run["max_new_tokens"], 0)
            for i in range(n):
                layer(i)
            sync()
        kern = defaultdict(lambda: [0, 0.0])
        for e in prof.events():
            dt = getattr(e, "device_time", None)
            if dt is None:
                dt = getattr(e, "cuda_time", 0.0)
            dev = getattr(e, "device_type", None)
            if dt and dev is not None and "CUDA" in str(dev):
                kern[e.name][0] += 1
                kern[e.name][1] += dt
        tap.end_call(n_generated=n, last_is_stop=True, call_index=0, record={})
        tap.detach()
        self.check_buffer(k)
        table = sorted(({"name": n_, "count": c, "total_us": t_, "mean_us": t_ / c} for n_, (c, t_) in kern.items()), key=lambda x: -x["total_us"])
        step = max(table, key=lambda x: x["count"])               # the per-step copy kernel: one launch per hook call
        total = sum(x["total_us"] for x in table)
        return {"device_total_ms": total / 1e3, "step_kernel_count": step["count"], "step_kernel_ms": step["total_us"] / 1e3,
                "step_kernel_mean_us": step["mean_us"], "other_device_ms": (total - step["total_us"]) / 1e3, "step_kernel_name": step["name"],
                "top_kernels": table[:8]}


def run(args) -> int:
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sched = load_schedule(args.stage2_list, args.calls)
    (out / "schedule.json").write_text(json.dumps({**sched, "growth": growth_schedule([r["n_generated"] for r in sched["runs"]])}, indent=1) + "\n")
    inputs = {"stage2_list": [str(args.stage2_list), sha256_file(args.stage2_list)], "calls": [str(args.calls), sha256_file(args.calls)],
              "artifact": [str(args.artifact), sha256_file(args.artifact)]}
    bench = Bench(args.artifact, sched)
    inputs["live_tap.py"] = sha256_file(Path(bench.live_tap.__file__))
    inputs["profile_components.py"] = sha256_file(Path(__file__))
    art = bench.tap.heads[next(iter(bench.tap.heads))].describe()
    inputs["artifact_description"] = art
    (out / "inputs.json").write_text(json.dumps(inputs, indent=1) + "\n")
    (out / "environment.json").write_text(json.dumps({**_env(), "reps": args.reps, "warmups": args.warmups, "gpu_reps": args.gpu_reps,
                                                      "gpu_warmups": args.gpu_warmups, "sleep_us": SLEEP_US}, indent=1) + "\n")
    runs = sched["runs"]
    cyc = bench.calibrate_sleep()
    print(f"sleep cycles for {SLEEP_US} us: {cyc}", flush=True)
    files = {k: open(out / f"raw_pass_{k}.jsonl", "w") for k in ("0a", "A", "0b", "B", "C", "P")}

    def emit(kind, rep, warm, k, row):
        run_ = runs[k]
        files[kind].write(json.dumps({"pass": kind, "rep": rep, "warmup": warm, "run_index": k, "run_key": run_["run_key"],
                                      "history_id": run_["history_id"], "n_generated": run_["n_generated"], **row}) + "\n")

    def sequence(fn, kind, rep, warm, **kw):
        bench.new_tap()
        for k, run_ in enumerate(runs):
            with torch.no_grad():
                row = fn(k, run_, **kw)
            torch.cuda.empty_cache()
            emit(kind, rep, warm, k, row)
        bench.tap.detach()

    t_start = time.perf_counter()
    for rep in range(-args.warmups, args.reps):
        warm = rep < 0
        order = [("0a", bench.baseline_a), ("A", bench.pass_a), ("0b", bench.baseline_b), ("B", bench.pass_b)]
        if rep % 2:
            order = [order[2], order[3], order[0], order[1]]
        for kind, fn in order:
            sequence(fn, kind, rep, warm)
        if rep % 20 == 0:
            for f in files.values():
                f.flush()
            print(f"rep {rep} done, {time.perf_counter() - t_start:.0f} s", flush=True)
    pairs = bench.make_pairs(max(r["n_generated"] for r in runs))
    for rep in range(-args.gpu_warmups, args.gpu_reps):
        sequence(bench.pass_c, "C", rep, rep < 0, pairs=pairs, sleep_cycles=cyc)
        print(f"C rep {rep} done, {time.perf_counter() - t_start:.0f} s", flush=True)
    for rep in range(-1, args.p_reps):
        sequence(bench.pass_p, "P", rep, rep < 0)
        print(f"P rep {rep} done, {time.perf_counter() - t_start:.0f} s", flush=True)
    for f in files.values():
        f.close()
    timer = bench.timer_overhead(cyc)
    (out / "timer_overhead.json").write_text(json.dumps(timer, indent=1) + "\n")
    print(f"measurement finished in {time.perf_counter() - t_start:.0f} s", flush=True)
    summarize(out)
    return 0


def summarize(out: Path) -> dict:
    out = Path(out)
    sched = json.loads((out / "schedule.json").read_text())
    raw = {k: read_jsonl(out / f"raw_pass_{k}.jsonl") for k in ("0a", "A", "0b", "B", "C", "P")}
    timer = json.loads((out / "timer_overhead.json").read_text()) if (out / "timer_overhead.json").exists() else {}
    floor = (timer.get("event_floor_pair_around_standin_call_no_hook_ms") or {}).get("mean")
    tables = build_tables(raw, sched, floor)
    tables["event_floor_ms_used"] = floor
    (out / "summary.json").write_text(json.dumps(tables, indent=1) + "\n")
    # per-position table (csv): median and mean over repeats of the main quantities
    names = ["alone_net_ms", "begin_ms", "hook_cpu_net_ms", "growth_cpu_net_ms", "growth_wait_net_ms", "drain_ms", "end_call_ms", "score_head_ms",
             "partition_total_ms", "gpu_capture_net_ms", "kernel_step_ms", "device_bound_total_ms", "device_kernel_total_ms"]
    names = [n for n in names if n in tables["per_position"]]
    mean_only = {"device_bound_total_ms", "device_kernel_total_ms"}
    lines = ["history_id,task,weight,n_generated," + ",".join(f"{n}_mean" if n in mean_only else f"{n}_median,{n}_mean" for n in names)]
    ntok = {r["run_key"]: r["n_generated"] for r in sched["runs"]}
    for p in sched["positions"]:
        cells = []
        for n in names:
            pp = tables["per_position"][n]
            if n not in mean_only:
                cells.append(f"{pp['median'][p['history_id']]:.6f}")
            cells.append(f"{pp['mean'][p['history_id']]:.6f}")
        lines.append(f"{p['history_id']},{p['task']},{p['weight']:.6f},{ntok[p['run_key']]}," + ",".join(cells))
    (out / "per_position.csv").write_text("\n".join(lines) + "\n")
    return tables


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2-list", type=Path)
    ap.add_argument("--calls", type=Path)
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmups", type=int, default=20)
    ap.add_argument("--gpu-reps", type=int, default=20, help="repeats of the event pass C")
    ap.add_argument("--gpu-warmups", type=int, default=3)
    ap.add_argument("--p-reps", type=int, default=5, help="repeats of the profiler pass P")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)
    if args.summarize_only:
        summarize(args.out)
        return 0
    if not (args.stage2_list and args.calls and args.artifact):
        ap.error("--stage2-list, --calls and --artifact are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
