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
from evals.tap_live import correctness_trial as ct
from evals.tap_live import decode_bench as db
from evals.tap_live import decode_core as dc
from evals.tap_live import export_phase_table as ex
from evals.tap_live import length_check as lc
from evals.tap_live import live_tap as lt
from evals.tap_live import profile_pass as pp
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


def test_correctness_trial_against_the_offline_scorer(model, tok, fmt, histories, artifact, tmp_path):
    inject = {"model": model, "tokenizer": tok, "fmt": fmt}
    args = ["--model", "tiny", "--histories", str(histories[0]), "--artifact", str(artifact), "--device", "cpu", "--dtype", "float32",
            "--out", str(tmp_path / "c.json"), "--max-new-tokens", "24", "--prefill-chunk", "16"]
    assert ct.main(args, inject=inject) == 0
    res = json.loads((tmp_path / "c.json").read_text())
    assert len(res["histories"]) == 2 and res["declared_tolerance_abs_score"] == 1e-4 and res["all_declared_checks_pass"]
    for x in res["histories"]:
        assert x["a1_arithmetic_on_tap_states"]["pass"] and x["a2_full_forward_pass"]["pass"]
        assert x["a2_full_forward_pass"]["max_abs_state_diff_vs_tap_buffer"] < 1e-4
        assert x["a1_arithmetic_on_tap_states"]["tap"]["eligible_tokens"] == x["a1_arithmetic_on_tap_states"]["offline"]["eligible_tokens"]
        assert x["b_cached_vs_fresh"]["identical_token_streams"] and x["b_cached_vs_fresh"]["max_abs_first_logits_diff"] < 1e-4
        assert x["c_A0_vs_A1"]["identical_token_streams"]
        assert x["d_chunked_vs_single_prefill"]["identical_token_streams"] and x["d_chunked_vs_single_prefill"]["max_abs_first_logits_diff"] < 1e-4


def test_correctness_trial_with_a_forced_stop(model, tok, fmt, histories, artifact, tmp_path):
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    body = tok.encode(" the user wants me", add_special_tokens=False)
    inject = {"model": model, "tokenizer": tok, "fmt": fmt, "logits_processor": force(body, stop)}
    args = ["--model", "tiny", "--histories", str(histories[0]), "--artifact", str(artifact), "--device", "cpu", "--dtype", "float32",
            "--out", str(tmp_path / "c.json"), "--n-histories", "1"]
    ct.main(args, inject=inject)
    x = json.loads((tmp_path / "c.json").read_text())["histories"][0]
    assert x["a2_full_forward_pass"]["tap"]["eligible_tokens"] == len(body) == x["a2_full_forward_pass"]["offline"]["eligible_tokens"]
    assert x["a1_arithmetic_on_tap_states"]["pass"] and x["a2_full_forward_pass"]["pass"]


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
    assert set(c["code_sha256"]) >= {"decode_bench.py", "live_tap.py", "podman_shim/docker"} and c["histories_sha256"] == db.sha256_file(histories[0])
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


# ---- separate passes ------------------------------------------------------------------------------------------------------

def test_profile_pass_and_length_check_run(model, tok, fmt, histories, artifact, tmp_path, monkeypatch):
    inject = {"model": model, "tokenizer": tok, "fmt": fmt}
    common = ["--model", "tiny", "--histories", str(histories[0]), "--artifact", str(artifact), "--device", "cpu", "--dtype", "float32"]
    pp.main([*common, "--out", str(tmp_path / "p"), "--n-histories", "2", "--repeats", "2", "--max-new-tokens", "8"], inject=inject)
    s = json.loads((tmp_path / "p" / "profile_summary.json").read_text())
    assert set(s["arms"]) == {"A0", "A1"} and s["arms"]["A1"]["capture_s"] > 0 and s["arms"]["A1"]["score_wall_s"] > 0
    assert s["arms"]["A0"]["decode_step_ms"]["n"] > 0 and "not an overhead estimate" in s["note"]
    monkeypatch.setattr(lt, "INITIAL_ROWS", 4)
    lc.main([*common, "--out", str(tmp_path / "l"), "--n-histories", "2", "--repeats", "2", "--budgets", "6,12,20"], inject=inject)
    ls = json.loads((tmp_path / "l" / "length_summary.json").read_text())
    assert set(k for k in ls if k.isdigit()) == {"6", "12", "20"}
    assert ls["20"]["A1"]["max_generated"] <= 20 and ls["20"]["tap_buffer_rows_after"] >= ls["20"]["A1"]["max_generated"]
    assert ls["12"]["tap_buffer_grew_in_runs"] + ls["20"]["tap_buffer_grew_in_runs"] > 0
    assert ls["12"]["identical_streams_A0_A1"] == 4


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


# ---- phase-table export ------------------------------------------------------------------------------------------------------

TEMPLATE = Path(__file__).resolve().parents[2] / "review_exchange" / "eq1_phases_live_table.tex"


def synth_decode_dir(d: Path, budget=256):
    d.mkdir()
    heads = {"token_mean": {"sha256": "ab" * 32, "layer": 48, "pooling": "token_mean", "threshold": 0.52}}
    cfg = {"config_sha256": "y" * 64, "config": {"history_ids": ["a@0.25", "b@0.5"], "repeats": 2, "max_new_tokens": budget,
                                                  "artifact": {"heads": heads}}}
    (d / "run_config.json").write_text(json.dumps(cfg))
    rows = []
    for h in ("a@0.25", "b@0.5"):
        for rep in (0, 1):
            rows += [{**mk(h, rep, "A0", 10.0, gen=200), "budget": budget}, {**mk(h, rep, "A1", 10.05, gen=200), "budget": budget}]
    (d / "decode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "setup.jsonl").write_text(json.dumps({"session": "s1", "model_load_s": 100.0, "artifact_load_s": 0.5, "warmup_s": 3.0}) + "\n")


@pytest.mark.skipif(not TEMPLATE.exists(), reason="review_exchange table skeleton not present")
def test_phase_table_export_keeps_the_skeleton_and_fills_only_the_row(tmp_path):
    synth_decode_dir(tmp_path / "dec")
    ldir = tmp_path / "len"
    ldir.mkdir()
    lrows = []
    for b in (256, 1024, 4096):
        for h in ("a@0.25", "b@0.5"):
            lrows += [{**mk(h, 0, "A0", 20.0 + b / 1000), "budget": b}, {**mk(h, 0, "A1", 20.1 + b / 1000), "budget": b}]
    (ldir / "length.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lrows))
    ex.main(["--decode-dir", str(tmp_path / "dec"), "--length-dir", str(ldir), "--out-dir", str(tmp_path / "o"), "--template", str(TEMPLATE)])
    doc = json.loads((tmp_path / "o" / "eq1_phase_row.json").read_text())
    r0 = doc["rows"][0]
    assert [r["block_budget_tokens"] for r in doc["rows"]] == [256, 1024, 4096]              # not merged
    assert r0["setup"]["setup_s"] == pytest.approx(100.5) and r0["setup"]["warmup_s_reported_separately"] == 3.0
    assert r0["all_attempt"]["no_tap_s_mean"] == pytest.approx(10.0) and r0["all_attempt"]["with_tap_s_mean"] == pytest.approx(10.05)
    assert r0["all_attempt"]["tap_ms_per_call_mean_paired"] == pytest.approx(50.0) and r0["clean"]["n_pairs"] == 4
    assert r0["all_attempt"]["mean_generated_per_call"] == {"A0": 200, "A1": 200} and r0["block_budget_tokens"] == 256
    assert r0["attestation_s"] is None and r0["total_certification_s"] is None and r0["tap"] == "Probe: mean"
    assert doc["rows"][1]["setup"] is None and "unit" in doc and "generation call" in doc["unit"]
    assert "mean elapsed seconds per timed call" in doc["aggregation"] and "caption_note" in r0
    tex = (tmp_path / "o" / "eq1_phase_table.tex").read_text()
    assert "\n" + doc["caption_latex"] in tex
    tex = tex.replace("\n" + doc["caption_latex"], "", 1)                 # the caption additions are the only other change
    orig = TEMPLATE.read_text()
    head = orig[:orig.index("\\midrule") + len("\\midrule\n")]
    tail = orig[orig.index("\n%", len(head)) + 1:]            # from the first comment line after the data rows
    assert tex.startswith(head) and tex.endswith(tail)                                      # skeleton verbatim
    body = tex[len(head):-len(tail)]
    rows = [x for x in body.split("\\\\\n") if x.strip()]
    assert len(rows) == 3 and all(x.rstrip().endswith("-- & --") or "-- & --" in x for x in rows)
    assert "100.50" in rows[0] and "10.000" in rows[0] and "10.050" in rows[0] and "+50.00" in rows[0] and "256-token budget" in rows[0]
    assert "1{,}024-token budget" in rows[1] and "21.024" in rows[1] and "& -- & 21.024" in " ".join(rows[1].split())
    assert "Probe: mean" in rows[0] and "Qwen3.8-27B & \\shortstack[l]{LiveCodeBench" in rows[0]
    assert (tmp_path / "o" / "eq1_phase_rows.tex").read_text().count("Qwen3.8-27B") == 3


@pytest.mark.skipif(not TEMPLATE.exists(), reason="review_exchange table skeleton not present")
def test_phase_table_from_a_real_tiny_run(bench_out, tmp_path):
    ex.main(["--decode-dir", str(bench_out), "--out-dir", str(tmp_path / "o")] + (["--template", str(TEMPLATE)] if TEMPLATE.exists() else []))
    doc = json.loads((tmp_path / "o" / "eq1_phase_row.json").read_text())
    r = doc["rows"][0]
    assert r["all_attempt"]["n_pairs"] == 6 and r["setup"]["setup_s"] > 0 and r["tap_artifacts"]
    assert re.search(r"Probe: mean", (tmp_path / "o" / "eq1_phase_rows.tex").read_text())


def test_phase_row_uses_one_weighting_with_unequal_history_counts(tmp_path):
    d = tmp_path / "dec"
    synth_decode_dir(d)
    rows = [json.loads(x) for x in (d / "decode.jsonl").read_text().splitlines()]
    # history a@0.25 loses its repeat 1 (failed A1) and history c@0.5 (another task) is added with a larger overhead
    rows = [r for r in rows if not (r["history_id"] == "a@0.25" and r["repeat"] == 1 and r["arm"] == "A1")]
    for rep in (0, 1, 2):
        rows += [{**mk("c@0.5", rep, "A0", 12.0, gen=150), "budget": 256}, {**mk("c@0.5", rep, "A1", 12.4, gen=150), "budget": 256}]
    cfg = json.loads((d / "run_config.json").read_text())
    cfg["config"]["history_ids"] = ["a@0.25", "b@0.5", "c@0.5"]
    cfg["config"]["histories_summary"] = {"n_tasks": 3, "n_histories": 3, "n_fallback_to_earlier_call": 0, "fallback_history_ids": []}
    (d / "run_config.json").write_text(json.dumps(cfg))
    (d / "decode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    doc, _ = ex.build(d)
    a = doc["rows"][0]["all_attempt"]
    assert a["n_pairs"] == 6 and a["n_histories"] == 3 and a["n_tasks"] == 3
    # the displayed difference is the with-tap mean minus the no-tap mean of the same pairs, and lies inside the interval
    assert a["tap_ms_per_call_mean_paired"] == pytest.approx(1000 * (a["with_tap_s_mean"] - a["no_tap_s_mean"]))
    est = a["estimand_pair_weighted_abs_ms"]
    assert est["mean_paired_difference_ms"] == pytest.approx(a["tap_ms_per_call_mean_paired"]) and est["weighting"].startswith("pair-weighted")
    assert a["ci95_ms"][0] <= a["tap_ms_per_call_mean_paired"] <= a["ci95_ms"][1] and a["ci95_ms"] == est["ci95_ms"]
    # the equal-weight-per-history estimand differs when the counts are unequal, and is labelled as another estimand
    hw = a["estimand_history_weighted_abs_ms"]
    assert hw["weighting"].startswith("equal weight per history") and hw["mean_paired_difference_ms"] != pytest.approx(est["mean_paired_difference_ms"])
    assert "absolute" in est["quantity"] and "fraction" in a["relative_overhead_fraction"]["quantity"]
    assert doc["rows"][0]["selection_coverage"]["unique_tasks"] == 3 and a["normalized_cost_per_100_tokens"]["n_used"] == 6
    assert "bootstrap" in est and "task" in est["bootstrap"]["unit"]


@pytest.mark.skipif(not TEMPLATE.exists(), reason="review_exchange table skeleton not present")
def test_caption_note_and_length_rows_carry_range_not_interval(tmp_path):
    synth_decode_dir(tmp_path / "dec")
    ldir = tmp_path / "len"
    ldir.mkdir()
    lrows = []
    for h in ("a@0.25", "b@0.5", "c@0.75"):
        for rep in (0, 1, 2):
            lrows += [{**mk(h, rep, "A0", 20.0), "budget": 1024}, {**mk(h, rep, "A1", 20.0 + 0.01 * (rep + 1)), "budget": 1024}]
    (ldir / "length.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lrows))
    doc, cells = ex.build(tmp_path / "dec", ldir)
    note = doc["rows"][0]["caption_note"]
    for frag in ("generation block", "not a complete agent turn", "historical prefix is excluded".replace("historical prefix", "prefill of the historical prefix"),
                 "natural stops", "up to 256 newly generated tokens", "mean 200 generated tokens", "resamples whole tasks"):
        assert frag in note
    L = doc["rows"][1]["all_attempt"]
    assert L["ci95_ms"] is None and "no population interval" in L["interval_note"] and L["n_tasks"] == 3
    assert L["paired_diff_ms_raw_range"] == [pytest.approx(10.0), pytest.approx(30.0)] and "estimand_pair_weighted_abs_ms" not in L
    line = ex.latex_row(*cells[1])
    assert "range [+10.00, +30.00]" in line and "$^{\\dagger}$" in line and "3 tasks" in line and "natural stops" in ex.latex_row(*cells[0]) and "mean 200 tok." in ex.latex_row(*cells[0])


def test_length_check_summary_gives_the_raw_paired_range():
    rows = []
    for h in ("a@0.25", "b@0.5"):
        for rep in (0, 1):
            for arm, el in (("A0", 5.0), ("A1", 5.0 + 0.01 * (rep + 1))):
                rows.append({"history_id": h, "budget": 256, "repeat": rep, "arm": arm, "status": "ok", "elapsed_s": el, "n_generated": 256,
                             "stopped_naturally": False, "token_hash": "t", "buffer_rows_after": 256})
    s = lc.summarize(rows)["256"]
    assert s["paired_diff_ms"]["n_pairs"] == 4 and s["paired_diff_ms"]["min"] == pytest.approx(10.0) and s["paired_diff_ms"]["max"] == pytest.approx(20.0)
    assert s["task_coverage"]["unique_tasks"] == 2 and "population interval" in s["task_coverage"]["note"]


def test_significant_figures_keep_small_uncertainties_visible():
    assert sd.fmt_sig(0.0020643691) == "0.0021" and sd.fmt_sig(0.0000123) == "0.000012" and sd.fmt_sig(0.5) == "0.50" and sd.fmt_sig(12.3456) == "12.35"
    assert sd.fmt_sig(-0.0637, True) == "-0.064" and sd.fmt_sig(0.14, True) == "+0.14" and sd.fmt_sig(0) == "0" and sd.fmt_sig(None) == "-"
    assert sd._pct(0.000020643691) == "+0.0021%" and sd._pct(-0.2) == "-20.00%"
    pairs = [(mk(f"{t}@0.25", 0, "A0", 10.0), mk(f"{t}@0.25", 0, "A1", 10.0 + d)) for t, d in (("a", 0.0001), ("b", 0.0002), ("c", 0.0003))]
    e = sd.rel_estimand(pairs)
    line = sd._estimand_lines("x", e)[0]
    assert "percentage points" in line and "0.00 percentage" not in line and re.search(r"half-width 0\.00\d+ percentage points", line)
    assert ex.cell_num(-0.0637) == "-0.06" and ex.cell_num(0.1371) == "+0.14" and ex.cell_num(-0.2832) == "-0.28" and ex.cell_num(0.0021) == "+0.0021"


@pytest.mark.skipif(not TEMPLATE.exists(), reason="review_exchange table skeleton not present")
def test_main_row_shows_estimate_and_asymmetric_interval_with_an_unresolved_note(tmp_path):
    d = tmp_path / "dec"
    synth_decode_dir(d)
    rows = [json.loads(x) for x in (d / "decode.jsonl").read_text().splitlines()]
    # tasks a and b: A1 slightly slower or faster, so the task bootstrap interval is asymmetric about a small negative estimate
    for r in rows:
        if r["arm"] == "A1":
            r["elapsed_s"] = 10.0 + (-0.0004 if r["task"] == "a" else 0.0001)
    (d / "decode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ex.main(["--decode-dir", str(d), "--out-dir", str(tmp_path / "o"), "--template", str(TEMPLATE)])
    doc = json.loads((tmp_path / "o" / "eq1_phase_row.json").read_text())
    a = doc["rows"][0]["all_attempt"]
    lo, hi = a["ci95_ms"]
    assert lo < 0 < hi and a["interval_statement"].startswith("interval includes zero (unresolved)") and "not a speedup finding" in a["interval_statement"]
    assert "interval includes zero (unresolved)" in doc["rows"][0]["caption_note"].replace("The interval includes zero (unresolved)", "interval includes zero (unresolved)")
    assert "unresolved at" not in doc["rows"][0]["caption_note"] and "±" not in doc["rows"][0]["caption_note"]
    tex = (tmp_path / "o" / "eq1_phase_rows.tex").read_text()
    est = ex.cell_num(a["tap_ms_per_call_mean_paired"])
    assert f"{{{est}\\\\{{\\scriptsize [{ex.cell_num(lo)}, {ex.cell_num(hi)}]}}}}" in tex.replace("\\\\", "\\\\") or f"[{ex.cell_num(lo)}, {ex.cell_num(hi)}]" in tex


def test_length_rows_count_tasks_from_the_history_ids_and_carry_a_dagger(tmp_path):
    synth_decode_dir(tmp_path / "dec")
    ldir = tmp_path / "len"
    ldir.mkdir()
    lrows = []
    for h in ("lcbhard_101@0.25", "lcbhard_55@0.75", "lcbhard_99@0.75"):
        for rep in (0, 1):
            lrows += [{**mk(h, rep, "A0", 20.0), "budget": 1024}, {**mk(h, rep, "A1", 20.0 + 0.01 * (rep + 1)), "budget": 1024}]
    (ldir / "length.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lrows))
    doc, cells = ex.build(tmp_path / "dec", ldir)
    L = doc["rows"][1]["all_attempt"]
    assert L["n_tasks"] == 3 and L["n_histories"] == 3 and "not an interval; few tasks" in L["interval_note"]
    assert "3 unique task(s)" in doc["rows"][0]["caption_note"] and "$^{\\dagger}$" in doc["rows"][0]["caption_note"]
    assert "range [" in ex.latex_row(*cells[1]) and "$^{\\dagger}$" in ex.latex_row(*cells[1]) and "$^{\\dagger}$" not in ex.latex_row(*cells[0])
    rows = [{"history_id": h, "budget": 256, "repeat": 0, "arm": arm, "status": "ok", "elapsed_s": 5.0, "n_generated": 10, "stopped_naturally": False,
             "token_hash": "t", "buffer_rows_after": 256} for h in ("lcbhard_101@0.25", "lcbhard_55@0.75", "lcbhard_99@0.75") for arm in ("A0", "A1")]
    cov = lc.summarize(rows)["256"]["task_coverage"]
    assert cov["unique_tasks"] == 3 and cov["task_ids"] == ["lcbhard_101", "lcbhard_55", "lcbhard_99"]


@pytest.mark.skipif(not TEMPLATE.exists(), reason="review_exchange table skeleton not present")
def test_caption_additions_reach_the_exported_tex_and_it_compiles(tmp_path):
    import shutil
    import subprocess
    import tempfile
    synth_decode_dir(tmp_path / "dec")
    rows = [json.loads(x) for x in (tmp_path / "dec" / "decode.jsonl").read_text().splitlines()]
    for r in rows:                                           # an interval that includes zero
        if r["arm"] == "A1":
            r["elapsed_s"] = 10.0 + (-0.0004 if r["task"] == "a" else 0.0001)
    (tmp_path / "dec" / "decode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ldir = tmp_path / "len"
    ldir.mkdir()
    lrows = []
    for h in ("lcbhard_101@0.25", "lcbhard_55@0.75", "lcbhard_99@0.75"):
        for rep in (0, 1):
            lrows += [{**mk(h, rep, "A0", 20.0), "budget": 1024}, {**mk(h, rep, "A1", 20.0 + 0.01 * (rep + 1)), "budget": 1024}]
    (ldir / "length.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lrows))
    ex.main(["--decode-dir", str(tmp_path / "dec"), "--length-dir", str(ldir), "--out-dir", str(tmp_path / "o"), "--template", str(TEMPLATE)])
    tex = (tmp_path / "o" / "eq1_phase_table.tex").read_text()
    doc = json.loads((tmp_path / "o" / "eq1_phase_row.json").read_text())
    cap = tex[tex.index("\\caption{"):tex.index("\\label{")]
    orig = TEMPLATE.read_text()
    owner_cap = orig[orig.index("\\caption{") + 9:orig.index("\n\\label")].rstrip("}")
    assert owner_cap in cap and "\\label{tab:eq1-phases}" in tex                                      # owner wording and label kept
    for frag in ("generation blocks, not complete agent turns", "the prefill of the historical prefix excluded", "95\\% bootstrap interval that resamples whole tasks",
                 "observed paired range$^{\\dagger}$", "Raw paired range, not an interval (few tasks)", "the interval includes zero (unresolved)",
                 "is not evidence of a speedup", "natural stops kept", "3 unique tasks", "Attestation and total certification were not measured"):
        assert frag in cap, frag
    assert cap.count("{") == cap.count("}") and doc["caption_latex"] in cap
    pdflatex, base = shutil.which("pdflatex"), Path(tempfile.gettempdir())
    if pdflatex is None or not base.is_dir():
        pytest.skip("pdflatex or the job tmp directory is not available")
    work = Path(tempfile.mkdtemp(dir=base))
    try:
        (work / "t.tex").write_text("\\documentclass{article}\n\\usepackage{booktabs,amsmath}\n\\begin{document}\n" + tex + "\n\\end{document}\n")
        r = subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "t.tex"], cwd=work, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and (work / "t.pdf").exists(), r.stdout[-1500:]
    finally:
        shutil.rmtree(work, ignore_errors=True)
