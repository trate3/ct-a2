"""CPU tests of the complete-call generation run (tiny random Qwen3_5 model, synthetic histories)."""
from __future__ import annotations

import json

import pytest
import torch

from evals.tap_live import decode_bench as db
from evals.tap_live import generate_calls as gc
from evals.tap_live import summarize_calls as sc
from evals.tap_live.test_decode_bench import artifact, force, histories, mk, traj_dir  # noqa: F401  (fixtures and helpers)

LAYER = 2


def argv(histories, artifact, out, extra=()):
    return ["--model", "tiny", "--histories", str(histories[0]), "--artifact", str(artifact), "--out", str(out), "--device", "cpu", "--dtype", "float32",
            "--repeats", "2", "--cap", "10", "--limit-histories", "2", *extra]


def inject(model, tok, fmt, processor=None):
    return {"model": model, "tokenizer": tok, "fmt": fmt, **({"logits_processor": processor} if processor else {})}


def rows_of(out):
    return [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]


def test_cap_hits_and_natural_stops_are_recorded_and_outputs_saved(model, tok, fmt, histories, artifact, tmp_path):
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    body = tok.encode(" the user wants me", add_special_tokens=False)
    out = tmp_path / "o"

    def keyed(logits, gen, ci):
        return force(body, stop)(logits, gen, ci) if keyed.natural else logits
    keyed.natural = True
    assert gc.main(argv(histories, artifact, out, ["--limit-histories", "1", "--cap", "40"]), inject=inject(model, tok, fmt, keyed)) == 0
    rows = rows_of(out)
    assert len(rows) == 4 and all(r["status"] == "ok" and r["finish_reason"] == "natural" and r["n_generated"] == len(body) + 1 for r in rows)
    outs = [json.loads(x) for x in (out / "outputs.jsonl").read_text().splitlines()]
    assert len(outs) == 1                                    # the A0 first repeat only
    o = outs[0]
    assert o["arm"] == "A0" and o["repeat"] == 0 and o["finish_reason"] == "natural" and o["n_tokens"] == len(body) + 1
    assert o["token_ids"] == body + [stop] and o["stop_token_id"] == stop
    assert o["text_with_special_tokens"].endswith("<|im_end|>") and not o["text_stop_token_removed"].endswith("<|im_end|>")
    assert o["text_with_special_tokens"] == o["text_stop_token_removed"] + "<|im_end|>" and o["prompt_sha256"]
    # a never-stopping model hits the cap
    out2 = tmp_path / "o2"
    keyed.natural = False
    gc.main(argv(histories, artifact, out2, ["--limit-histories", "1", "--cap", "7"]), inject=inject(model, tok, fmt, lambda l, g, c: torch.where(
        torch.arange(l.shape[0]) == stop, torch.full_like(l, -1e9), l)))
    r2 = rows_of(out2)
    assert all(r["finish_reason"] == "cap" and r["hit_cap"] and r["n_generated"] == 7 and not r["stopped_naturally"] for r in r2)
    o2 = json.loads((out2 / "outputs.jsonl").read_text().splitlines()[0])
    assert o2["finish_reason"] == "cap" and o2["stop_token_id"] is None and o2["text_with_special_tokens"] == o2["text_stop_token_removed"]
    s = sc.summarize(out2)
    assert s["counts"]["A0"]["cap_hits"] == 2 and s["counts"]["A0"]["natural_stops"] == 0 and "mean_seconds_per_call_natural_only" not in s["counts"]["A0"]
    assert s["identical_natural_stop_only"]["empty"]


def test_run_rows_order_arms_seeds_and_provenance(model, tok, fmt, histories, artifact, tmp_path):
    out = tmp_path / "o"
    assert gc.main(argv(histories, artifact, out), inject=inject(model, tok, fmt)) == 0
    rows = rows_of(out)
    assert len(rows) == 2 * 2 * 2 and all(r["status"] == "ok" and r["cache_equal_to_base"] for r in rows)
    cfg = json.loads((out / "run_config.json").read_text())
    c = cfg["config"]
    assert c["kind"] == "generate_calls" and c["cap"] == 10 and c["repeats"] == 2 and "cap_behaviour" in c and c["histories_sha256"] == db.sha256_file(histories[0])
    assert "generate_calls.py" in c["code_sha256"] and "summarize_calls.py" in c["code_sha256"] and c["artifact"]["heads"] and c["sampling"]["top_k"] == 20
    assert {r["config_sha256"] for r in rows} == {cfg["config_sha256"]}
    for r in rows:
        assert r["order_position"] == db.arm_order(r["repeat"]).index(r["arm"]) and r["seed"] == gc.seed_call(r["history_id"], r["repeat"])
    for h in {r["history_id"] for r in rows}:
        for rep in (0, 1):
            pair = [r for r in rows if r["history_id"] == h and r["repeat"] == rep]
            assert len({p["seed"] for p in pair}) == 1 and len({p["token_hash"] for p in pair}) == 1
    assert len((out / "outputs.jsonl").read_text().splitlines()) == 2 and (out / "setup.jsonl").exists()
    a1 = next(r for r in rows if r["arm"] == "A1")
    assert a1["decisions"][0]["n_tokens"] in (a1["n_generated"], a1["n_generated"] - 1)


def test_a0_only_schedule(model, tok, fmt, histories, artifact, tmp_path):
    out = tmp_path / "o"
    gc.main(argv(histories, artifact, out, ["--arms", "A0"]), inject=inject(model, tok, fmt))
    assert {r["arm"] for r in rows_of(out)} == {"A0"} and len(rows_of(out)) == 4
    s = sc.summarize(out)
    assert "pairs" not in s and s["counts"]["A0"]["completed"] == 4
    with pytest.raises(SystemExit):
        gc.main(argv(histories, artifact, tmp_path / "bad", ["--arms", "A1"]), inject=inject(model, tok, fmt))


def test_resume_refuses_a_changed_config_and_repairs_an_interrupted_run(model, tok, fmt, histories, artifact, tmp_path):
    out = tmp_path / "o"
    inj = inject(model, tok, fmt)
    gc.main(argv(histories, artifact, out, ["--limit-histories", "1"]), inject=inj)
    n = len(rows_of(out))
    cfg_bytes = (out / "run_config.json").read_bytes()
    assert gc.main(argv(histories, artifact, out, ["--limit-histories", "1"]), inject=inj) == 0 and len(rows_of(out)) == n
    for extra in (["--cap", "11"], ["--repeats", "3"]):
        with pytest.raises(SystemExit, match="refusing to resume"):
            gc.main(argv(histories, artifact, out, ["--limit-histories", "1", *extra]), inject=inj)
    assert (out / "run_config.json").read_bytes() == cfg_bytes
    # crash after the output line but before the A0 first-repeat row: resume reruns the call and does not duplicate the output
    lines = (out / "calls.jsonl").read_text().splitlines()
    first = next(i for i, x in enumerate(lines) if json.loads(x)["arm"] == "A0" and json.loads(x)["repeat"] == 0)
    (out / "calls.jsonl").write_text("\n".join(l for i, l in enumerate(lines) if i != first) + "\n")
    gc.main(argv(histories, artifact, out, ["--limit-histories", "1"]), inject=inj)
    assert len(rows_of(out)) == n and len((out / "outputs.jsonl").read_text().splitlines()) == 1


def test_a_failed_call_is_a_row_and_summary_counts_it(model, tok, fmt, histories, artifact, tmp_path):
    state = {"n": 0}

    def boom(logits, gen, ci):
        state["n"] += 1
        if state["n"] == 15:
            raise RuntimeError("injected failure")
        return logits
    out = tmp_path / "o"
    gc.main(argv(histories, artifact, out, ["--limit-histories", "1", "--no-warmup"]), inject=inject(model, tok, fmt, boom))
    rows = rows_of(out)
    assert len(rows) == 4 and [r["status"] for r in rows].count("error") == 1
    s = sc.summarize(out)
    assert sum(c["failed"] for c in s["counts"].values()) == 1 and s["pairs"]["with_a_failed_arm"] == 1


def test_summary_tables_cover_the_requested_quantities(tmp_path):
    hs = [f"{t}@{d}" for t in "abc" for d in (0.25, 0.5)]
    cfg = {"config_sha256": "z" * 64, "config": {"history_ids": hs, "repeats": 2, "cap": 4096, "arms": ["A0", "A1"],
                                                  "histories_summary": {"fallback_history_ids": ["a@0.5"]}}}
    (tmp_path / "run_config.json").write_text(json.dumps(cfg))
    rows = []
    for i, h in enumerate(hs):
        for rep in (0, 1):
            nat = i % 2 == 0
            base = {"finish_reason": "natural" if nat else "cap", "stopped_naturally": nat, "hit_cap": not nat, "cap": 4096}
            rows += [{**mk(h, rep, "A0", 50.0 + i, gen=1800 if nat else 4096), **base}, {**mk(h, rep, "A1", 50.0 + i + (0.01 if i % 3 else -0.01), gen=1800 if nat else 4096), **base}]
    (tmp_path / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = sc.summarize(tmp_path)
    a0 = s["counts"]["A0"]
    assert a0["scheduled"] == 12 and a0["attempted"] == 12 and a0["natural_stops"] == 6 and a0["cap_hits"] == 6
    assert a0["mean_generated_all"] == pytest.approx((1800 + 4096) / 2) and a0["mean_generated_natural_only"] == 1800
    assert a0["mean_seconds_per_call_all"] > 0 and s["unique_tasks"] == 3 and s["unique_histories"] == 6
    assert s["all_attempts"]["n_pairs"] == 12 and s["identical_natural_stop_only"]["n_pairs"] == 6
    e = s["all_attempts"]["estimand_overall"]
    assert e["n_tasks"] == 3 and "task" in e["bootstrap"]["unit"] and s["all_attempts"]["absolute_ms_per_call_pair_weighted"]["ci95_ms"]
    md = sc.to_markdown(s)
    assert "3 unique tasks" in md and "cap hits" in md and "natural stops" in md and "whole tasks" in md
    sc.main([str(tmp_path)])
    assert (tmp_path / "calls_summary.md").exists()
