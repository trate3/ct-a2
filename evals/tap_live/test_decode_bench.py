"""CPU tests of the focused decode measurement (tiny random Qwen3_5 model, tiny tokenizer, synthetic trajectories)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from evals.reward_hacking import activations as act
from evals.reward_hacking import tap as tapmod
from evals.tap_live import decode_bench as db
from evals.tap_live import decode_core as dc
from evals.tap_live import live_tap as lt
from evals.tap_live import select_histories as sh
from evals.tap_live import select_tasks, summarize_decode as sd
from evals.tap_live import tiny_model as tm

LAYER = 2
WORDS = "the user wants me to implement a function in func.py and pass the tests "


def write_traj(d: Path, task: str, k: int, n_calls: int, base_words: int, grow: int, tag: str = "") -> Path:
    events = [{"seq": 0, "kind": "user_message", "call_index": None, "data": {"text": "x"}, "t": 0.0},
              {"seq": 1, "kind": "evaluation", "call_index": None, "data": None, "t": 0.0}]       # labels are never read
    for i in range(n_calls):
        events.append({"seq": 2 + i, "kind": "model_call", "call_index": i, "t": 0.0,
                       "data": {"prompt": tag + WORDS * (base_words + i * grow), "tokens": [1], "prompt_tokens": 7}})
    p = d / f"{task}__oneoff__T1__r{k}.json"
    p.write_text(json.dumps({"meta": {"task_id": task}, "events": events}))
    return p


@pytest.fixture(scope="module")
def traj_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("trajs") / "trajectories"
    d.mkdir()
    for t in select_tasks.select()["task_ids"]:
        write_traj(d, t, 0, 9, 2, 1)
        write_traj(d, t, 1, 4, 5, 0, tag="other episode ")
    return d.parent


@pytest.fixture(scope="module")
def histories(traj_dir, tok, tmp_path_factory):
    doc = sh.select(traj_dir, tok, cap=10_000)
    p = tmp_path_factory.mktemp("h") / "histories.json"
    p.write_text(json.dumps(doc))
    return p, doc


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    p = tmp_path_factory.mktemp("a") / "mean.json"
    tm.random_artifact(LAYER, "token_mean", seed=3, threshold=0.0).save(p)
    return p


# ---- selection ---------------------------------------------------------------------------------------------------------

def test_selection_rule_ids_depths_and_smallest_k(traj_dir, tok):
    doc = sh.select(traj_dir, tok, cap=10_000)
    assert doc["tasks"] == select_tasks.select()["task_ids"] and len(doc["tasks"]) == 10
    assert len(doc["histories"]) == 30 and doc["n_ok"] == 30
    for h in doc["histories"]:
        assert h["call_index"] == int(h["depth_fraction"] * 8) and h["call_index"] in (2, 4, 6)       # floor(f * (n - 1)), n = 9
        assert h["source_file"].endswith("__r0.json") and not h["prompt"].startswith("other episode")   # smallest k, not r1
        assert h["n_calls"] == 9 and not h["fallback_to_earlier_call"]
        assert h["prompt_sha256"] and h["source_sha256"] and h["history_id"] == f"{h['task']}@{h['depth_fraction']}"
    assert doc["splits_json"]["sha256"] and doc["splits_oneoff_v2_json"]["sha256"]
    assert doc["prompt_cap_tokens"] == 10_000 and sh.PROMPT_CAP == 48000 and "floor" not in doc["rule"] or "call index" in doc["rule"]


def test_earlier_call_fallback_and_unavailable(traj_dir, tok):
    lens = [len(tok.encode(WORDS * (2 + i), add_special_tokens=False)) for i in range(9)]
    cap = lens[5]                      # calls 0..5 fit, 6 does not
    doc = sh.select(traj_dir, tok, cap=cap, task_ids=select_tasks.select()["task_ids"][:1])
    by = {h["depth_fraction"]: h for h in doc["histories"]}
    assert by[0.75]["call_index"] == 5 and by[0.75]["fallback_to_earlier_call"] and by[0.75]["nominal_call_index"] == 6
    assert by[0.75]["prompt_tokens"] == lens[5] and by[0.5]["call_index"] == 4 and not by[0.5]["fallback_to_earlier_call"]
    tiny = sh.select(traj_dir, tok, cap=lens[0] - 1, task_ids=select_tasks.select()["task_ids"][:1])
    assert all(h["status"] == "unavailable" for h in tiny["histories"]) and tiny["n_unavailable"] == 3
    gone = sh.select(traj_dir, tok, cap=cap, task_ids=["lcbhard_none"])
    assert all(h["status"] == "unavailable" and h["reason"] == "no trajectory file" for h in gone["histories"])


def test_selection_cli(traj_dir, tok, tmp_path, monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: tok))
    sh.main(["--trajectories", str(traj_dir), "--tokenizer", "x", "--out", str(tmp_path / "h.json"), "--cap", "100000"])
    assert json.loads((tmp_path / "h.json").read_text())["n_ok"] == 30


# ---- identical cache start, stop handling, tap semantics ---------------------------------------------------------------

def _first(histories, tok):
    h = next(x for x in histories[1]["histories"] if x["status"] == "ok")
    return h, tok.encode(h["prompt"], add_special_tokens=False)


def test_both_arms_start_from_a_byte_identical_cache(model, tok, fmt, histories):
    h, ids = _first(histories, tok)
    dev = torch.device("cpu")
    base = dc.prefill_base(model, ids[:-1], dev)
    fp = dc.cache_fingerprint(base)
    last = torch.tensor([ids[-1:]])
    sp = fmt.sampling
    c0, _ = dc.copy_cache(base, dev)
    c1, _ = dc.copy_cache(base, dev)
    assert dc.caches_equal(base, c0) and dc.caches_equal(c0, c1) and dc.cache_fingerprint(c0) == dc.cache_fingerprint(c1) == fp
    r0 = dc.decode_run(model, c0, last, stop_ids=fmt.stop_token_ids, sampling=sp, seed=5, max_new_tokens=10, device=dev)
    assert not dc.caches_equal(base, c0) and dc.cache_fingerprint(base) == fp        # the continuation mutates the copy only
    c2, _ = dc.copy_cache(base, dev)                                                 # the second arm's copy: fresh, not A0's
    assert dc.cache_fingerprint(c2) == fp and dc.cache_fingerprint(c2) != dc.cache_fingerprint(c0)
    tap = lt.LiveProbeTap(tm.random_artifact(LAYER, "token_mean", seed=3), device=dev, time_capture=False)
    tap.attach(model)
    try:
        r1 = dc.decode_run(model, c1, last, stop_ids=fmt.stop_token_ids, sampling=sp, seed=5, max_new_tokens=10, device=dev, tap=tap)
    finally:
        tap.detach()
    assert r1["gen_ids"] == r0["gen_ids"]                # same seed, same start state, read-only tap


def force(ids_by_step, stop):
    def proc(logits, gen_ids, call_index):
        want = ids_by_step[len(gen_ids)] if len(gen_ids) < len(ids_by_step) else stop
        out = torch.full_like(logits, float("-inf"))
        out[want] = 0.0
        return out
    return proc


def test_natural_stop_is_honoured_and_excluded_from_pooling(model, tok, fmt, histories):
    h, ids = _first(histories, tok)
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    body = tok.encode(" the user wants me", add_special_tokens=False)
    dev = torch.device("cpu")
    base = dc.prefill_base(model, ids[:-1], dev)
    art = tm.random_artifact(LAYER, "token_mean", seed=3)
    tap = lt.LiveProbeTap(art, device=dev, time_capture=False)
    tap.attach(model)
    try:
        c, _ = dc.copy_cache(base, dev)
        r = dc.decode_run(model, c, torch.tensor([ids[-1:]]), stop_ids=fmt.stop_token_ids, sampling=fmt.sampling, seed=1,
                          max_new_tokens=256, device=dev, tap=tap, processor=force(body, stop))
    finally:
        tap.detach()
    assert r["stopped_naturally"] and r["gen_ids"] == body + [stop] and len(r["gen_ids"]) < 256
    d = r["decisions"][0]
    assert d["n_tokens"] == len(body)                                  # eligible tokens: the stop token's state is excluded
    states = tap.bufs[LAYER][:len(body) + 1].numpy()
    ch = np.array([act.VISIBLE] * len(body) + [act.END], np.int8)
    off = tapmod.StreamingTap(art).update(states.astype(np.float64), ch, 0)
    assert d["score"] == pytest.approx(off, abs=1e-5)
    # a run that hits the budget pools every generated token
    c2, _ = dc.copy_cache(base, dev)
    tap.attach(model)
    try:
        r2 = dc.decode_run(model, c2, torch.tensor([ids[-1:]]), stop_ids=fmt.stop_token_ids, sampling=fmt.sampling, seed=1,
                           max_new_tokens=5, device=dev, tap=tap, processor=force([body[0]] * 5, stop))
    finally:
        tap.detach()
    assert not r2["stopped_naturally"] and len(r2["gen_ids"]) == 5 and r2["decisions"][0]["n_tokens"] == 5


# ---- the benchmark: rows, order, provenance, resume ---------------------------------------------------------------------

def argv(histories, artifact, out, extra=()):
    return ["--model", "tiny", "--histories", str(histories[0]), "--artifact", str(artifact), "--out", str(out), "--device", "cpu",
            "--dtype", "float32", "--repeats", "2", "--max-new-tokens", "12", "--limit-histories", "3", *extra]


@pytest.fixture(scope="module")
def bench_out(model, tok, fmt, histories, artifact, tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    assert db.main(argv(histories, artifact, out), inject={"model": model, "tokenizer": tok, "fmt": fmt}) == 0
    return out


def test_bench_rows_order_and_provenance(bench_out, histories):
    rows = [json.loads(x) for x in (bench_out / "decode.jsonl").read_text().splitlines()]
    assert len(rows) == 3 * 2 * 2 and all(r["status"] == "ok" for r in rows)
    cfg = json.loads((bench_out / "run_config.json").read_text())
    assert {r["config_sha256"] for r in rows} == {cfg["config_sha256"]} and db.config_hash(cfg["config"]) == cfg["config_sha256"]
    c = cfg["config"]
    assert set(c["code_sha256"]) >= {"decode_bench.py", "live_tap.py"} and c["histories_sha256"] == db.sha256_file(histories[0])
    assert c["repeats"] == 2 and c["max_new_tokens"] == 12 and c["sampling"]["top_k"] == 20 and c["artifact"]["heads"]
    assert next(iter(c["artifact"]["heads"].values()))["sha256"] and "torch" in c["environment"] and (bench_out / "environment.json").exists()
    for r in rows:                                          # the same start state for both arms; order alternates with the repeat
        assert r["cache_equal_to_base"] is True and r["order_position"] == db.arm_order(r["repeat"]).index(r["arm"])
        assert r["prompt_token_count_matches_selection"] and r["cache_copy_s"] > 0 and r["session"]
        assert (r["decisions"] is None) == (r["arm"] == "A0")
    firsts = [r["arm"] for r in rows if r["order_position"] == 0]
    assert firsts.count("A0") == firsts.count("A1") == 3
    for h in {r["history_id"] for r in rows}:               # same seed, same start: identical streams in both arms
        for rep in (0, 1):
            pair = [r for r in rows if r["history_id"] == h and r["repeat"] == rep]
            assert len({p["seed"] for p in pair}) == 1 and len({p["token_hash"] for p in pair}) == 1
    a1 = next(r for r in rows if r["arm"] == "A1")
    assert a1["decisions"][0]["n_tokens"] in (a1["n_generated"], a1["n_generated"] - 1) and a1["tap_score_s"] > 0
    setup = [json.loads(x) for x in (bench_out / "setup.jsonl").read_text().splitlines()]
    assert setup[0]["model_load_s"] >= 0 and setup[0]["artifact_load_s"] > 0 and setup[0]["warmup_s"] > 0


def test_resume_is_refused_when_the_configuration_changes(model, tok, fmt, histories, artifact, bench_out, tmp_path):
    inject = {"model": model, "tokenizer": tok, "fmt": fmt}
    out = tmp_path / "r"
    db.main(argv(histories, artifact, out, ["--limit-histories", "1"]), inject=inject)
    cfg_bytes, env_bytes = (out / "run_config.json").read_bytes(), (out / "environment.json").read_bytes()
    n = len((out / "decode.jsonl").read_text().splitlines())
    assert db.main(argv(histories, artifact, out, ["--limit-histories", "1"]), inject=inject) == 0       # same config: resumes, nothing to do
    assert len((out / "decode.jsonl").read_text().splitlines()) == n
    with pytest.raises(SystemExit, match="refusing to resume"):
        db.main(argv(histories, artifact, out, ["--limit-histories", "1", "--max-new-tokens", "13"]), inject=inject)
    with pytest.raises(SystemExit, match="refusing to resume"):
        db.main(argv(histories, artifact, out, ["--limit-histories", "1", "--repeats", "3"]), inject=inject)
    other = tmp_path / "other.json"
    tm.random_artifact(LAYER, "token_mean", seed=99).save(other)
    with pytest.raises(SystemExit, match="refusing to resume"):
        db.main(argv(histories, other, out, ["--limit-histories", "1"]), inject=inject)
    assert (out / "run_config.json").read_bytes() == cfg_bytes and (out / "environment.json").read_bytes() == env_bytes
    # an interrupted run resumes: drop the last row and run again
    lines = (out / "decode.jsonl").read_text().splitlines()
    (out / "decode.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    db.main(argv(histories, artifact, out, ["--limit-histories", "1"]), inject=inject)
    assert len((out / "decode.jsonl").read_text().splitlines()) == n


def test_a_failing_run_is_a_row_not_a_gap(model, tok, fmt, histories, artifact, tmp_path):
    calls = {"n": 0}

    def boom(logits, gen_ids, ci):
        calls["n"] += 1
        if calls["n"] == 20:
            raise RuntimeError("injected failure")
        return logits
    out = tmp_path / "fail"
    db.main(argv(histories, artifact, out, ["--limit-histories", "1", "--no-warmup"]),
            inject={"model": model, "tokenizer": tok, "fmt": fmt, "logits_processor": boom})
    rows = [json.loads(x) for x in (out / "decode.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and [r["status"] for r in rows].count("error") == 1
    bad = next(r for r in rows if r["status"] == "error")
    assert "injected failure" in bad["error"] and bad["elapsed_s"] is None
    assert not any(act_hook for act_hook in act.find_decoder_layers(model)[LAYER - 1]._forward_hooks.values() if "LiveProbeTap" in getattr(act_hook, "__qualname__", ""))
    s = sd.summarize(out)
    assert s["counts"]["A0"]["failed"] + s["counts"]["A1"]["failed"] == 1 and s["pairs"]["with_a_failed_arm"] == 1
    assert s["pairs"]["with_an_elapsed_time_missing"] == 1 and s["all_attempts"]["n_pairs"] == 1


def test_summary_of_a_real_tiny_run(bench_out):
    s = sd.summarize(bench_out)
    assert s["counts"]["A0"]["scheduled"] == 6 and s["counts"]["A1"]["completed"] == 6 and s["counts"]["A0"]["missing"] == 0
    assert s["all_attempts"]["n_pairs"] == 6 and s["clean"]["n_pairs"] == 6 and not s["identical_streams"]["empty"]
    assert s["capture_s"] is None
    sd.main([str(bench_out)])
    assert (bench_out / "decode_summary.md").exists()


# ---- summary statistics -------------------------------------------------------------------------------------------------------

def synth(tmp_path, rows, histories=("a@0.25", "b@0.25", "c@0.5", "d@0.5"), repeats=2):
    cfg = {"config_sha256": "x" * 64, "config": {"history_ids": list(histories), "repeats": repeats, "max_new_tokens": 256,
                                                  "histories_n_unavailable": 2}}
    (tmp_path / "run_config.json").write_text(json.dumps(cfg))
    (tmp_path / "decode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return sd.summarize(tmp_path)


def mk(h, rep, arm, el, status="ok", tok="t", gen=100):
    return {"config_sha256": "x" * 64, "history_id": h, "task": h.split("@")[0], "depth": float(h.split("@")[1]), "repeat": rep,
            "arm": arm, "status": status, "elapsed_s": el, "n_generated": gen, "stopped_naturally": False, "token_hash": tok,
            "cache_copy_s": 0.01, "decisions": [{"alarm": False, "n_tokens": gen, "score": 0.1}] if arm == "A1" else None,
            "tap_score_s": 0.001 if arm == "A1" else None}


def test_summary_subsets_interval_and_missing_arms(tmp_path):
    hs = ["a@0.25", "b@0.25", "c@0.5", "d@0.5"]
    rows = []
    for i, h in enumerate(hs):
        for rep in (0, 1):
            rows += [mk(h, rep, "A0", 10.0), mk(h, rep, "A1", 10.0 + (0.5 if i % 2 == 0 else -0.5), tok="u" if (h, rep) == ("a@0.25", 0) else "t")]
    rows = [r for r in rows if not (r["history_id"] == "d@0.5" and r["repeat"] == 1 and r["arm"] == "A1")]            # a missing arm
    next(r for r in rows if r["history_id"] == "b@0.25" and r["repeat"] == 0 and r["arm"] == "A1").update(status="error", elapsed_s=None)
    next(r for r in rows if r["history_id"] == "c@0.5" and r["repeat"] == 1 and r["arm"] == "A1").update(status="cache_mismatch")
    s = synth(tmp_path, rows)
    assert s["counts"]["A1"] == {**s["counts"]["A1"], "scheduled": 8, "attempted": 7, "completed": 5, "failed": 2, "missing": 1}
    assert s["pairs"]["both_arms_attempted"] == 7 and s["pairs"]["missing_an_arm"] == 1 and s["pairs"]["with_a_failed_arm"] == 2
    assert s["all_attempts"]["n_pairs"] == 6 and s["clean"]["n_pairs"] == 5 and s["identical_streams"]["n_pairs"] == 4
    assert s["identical_streams"]["n_diverging_clean_pairs"] == 1
    allr, clean = s["all_attempts"], s["clean"]
    assert allr["estimand_overall"]["n_histories"] == 4 and clean["estimand_overall"]["n_histories"] == 4
    # per-history mean of paired relative overheads, then the mean over histories
    assert allr["estimand_overall"]["per_history"]["a@0.25"] == pytest.approx(0.05)
    assert allr["estimand_overall"]["mean_relative_overhead"] == pytest.approx(0.0)
    e = allr["estimand_overall"]
    assert e["verdict"] == "unresolved at the achieved precision" and e["ci95"][0] < 0 < e["ci95"][1]
    assert e["half_width_pct_points"] == pytest.approx(100 * (e["ci95"][1] - e["ci95"][0]) / 2)
    assert set(allr["per_depth"]) == {"0.25", "0.5"}
    assert len(allr["paired_diff_s_values"]) == 6 and allr["paired_diff_s"]["max"] == pytest.approx(0.5)
    md = sd.to_markdown(s)
    assert "unresolved at the achieved precision" in md and "Capture time: null" in md
    for bad in ("zero overhead", "speedup", "<1%"):
        assert bad not in md
    assert sd._verdict(*sd.task_bootstrap({"a": (0.1, 1), "b": (0.12, 1), "c": (0.11, 1), "d": (0.13, 1)})["ci95"]) == "interval above zero"
    assert sd.sum_or_none([1.0, None]) is None and sd.sum_or_none([]) is None and sd.sum_or_none([1.0, 2.0]) == 3.0


def test_task_bootstrap_is_reproducible_and_wider_with_noise():
    a = sd.task_bootstrap({str(i): (0.02, 1) for i in range(10)})
    assert a["ci95"] == pytest.approx([0.02, 0.02]) and a["bootstrap"]["unit"].startswith("task")
    parts = {str(i): ((0.1 if i % 2 else -0.1), 1) for i in range(10)}
    b = sd.task_bootstrap(parts)
    assert b == sd.task_bootstrap(parts) and (b["ci95"][1] - b["ci95"][0]) > 0.05


def test_resampling_keeps_the_depths_of_a_task_together(tmp_path):
    # task X: three depths all +10 %, task Y: three depths all +30 %. Resampling TASKS gives only the means 0.1, 0.2, 0.3
    # (P(all X) = 1/4 > 2.5 %, so the lower limit is exactly 0.1); resampling the six histories would not reach it.
    pairs = []
    for t, f in (("X", 0.1), ("Y", 0.3)):
        for d in (0.25, 0.5, 0.75):
            for rep in (0, 1):
                pairs.append((mk(f"{t}@{d}", rep, "A0", 10.0), mk(f"{t}@{d}", rep, "A1", 10.0 * (1 + f))))
    e = sd.rel_estimand(pairs)
    assert e["n_tasks"] == 2 and e["n_histories"] == 6 and e["n_pairs"] == 12
    assert e["mean_relative_overhead"] == pytest.approx(0.2) and e["ci95"][0] == pytest.approx(0.1) and e["ci95"][1] == pytest.approx(0.3)
    assert "task" in e["bootstrap"]["unit"] and "fraction" in e["quantity"]
    ab = sd.abs_estimand(pairs, "pair")
    assert "milliseconds" in ab["quantity"] and ab["mean_paired_difference_ms"] == pytest.approx(2000.0) and "ci95_ms" in ab


def test_normalized_cost_per_100_tokens(tmp_path):
    p = [(mk("a@0.25", 0, "A0", 10.0, gen=200), mk("a@0.25", 0, "A1", 10.4, gen=200)),
         (mk("b@0.25", 0, "A0", 10.0, gen=0), mk("b@0.25", 0, "A1", 10.1, gen=0)),                 # zero denominator
         (mk("c@0.5", 0, "A0", 10.0, gen=100, tok="p"), mk("c@0.5", 0, "A1", 10.9, gen=100, tok="q")),   # diverging: excluded
         (mk("d@0.5", 0, "A0", 20.0, gen=400), mk("d@0.5", 0, "A1", 20.4, gen=400))]
    n = sd.normalized_per_100_tokens(p)
    assert n["n_identical_pairs"] == 3 and n["n_zero_denominator_excluded"] == 1 and n["n_used"] == 2
    assert n["stats_ms_per_100_tokens"]["values"] if "values" in n["stats_ms_per_100_tokens"] else True
    assert n["pair_weighted_mean_ms_per_100_tokens"] == pytest.approx((400 * 100 / 200 + 400 * 100 / 400) / 2)
    assert "fixed per call" in n["note"] and "100 *" in n["formula"]
    s = synth(tmp_path, [r for a, b in p for r in (a, b)], histories=("a@0.25", "b@0.25", "c@0.5", "d@0.5"), repeats=1)
    assert s["normalized_cost_per_100_tokens"]["n_used"] == 2 and "per 100 generated tokens" in sd.to_markdown(s)


def test_summary_states_task_history_and_fallback_coverage(tmp_path):
    rows = [mk("a@0.25", 0, "A0", 10.0), mk("a@0.25", 0, "A1", 10.1)]
    cfg = {"config_sha256": "x" * 64, "config": {"history_ids": ["a@0.25", "a@0.5", "b@0.5"], "repeats": 1, "max_new_tokens": 256,
           "histories_summary": {"n_tasks": 2, "n_histories": 3, "n_fallback_to_earlier_call": 1, "fallback_history_ids": ["a@0.5"]}}}
    (tmp_path / "run_config.json").write_text(json.dumps(cfg))
    (tmp_path / "decode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = sd.summarize(tmp_path)
    assert s["unique_tasks"] == 2 and s["unique_histories"] == 3 and s["n_earlier_call_fallbacks"] == 1 and s["earlier_call_fallbacks"] == ["a@0.5"]
    md = sd.to_markdown(s)
    assert "2 unique tasks" in md and "a@0.5" in md and "resampling is the TASK" in md


def test_empty_identical_subset_is_stated(tmp_path):
    rows = []
    for h in ("a@0.25", "b@0.5"):
        rows += [mk(h, 0, "A0", 10.0, tok="p"), mk(h, 0, "A1", 11.0, tok="q")]
    s = synth(tmp_path, rows, histories=("a@0.25", "b@0.5"), repeats=1)
    assert s["identical_streams"]["empty"] and s["identical_streams"]["n_diverging_clean_pairs"] == 2
    assert "This subset is empty" in sd.to_markdown(s)
