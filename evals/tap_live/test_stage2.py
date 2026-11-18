"""CPU tests of the stage-2 natural-completion pipeline (tiny random Qwen3_5 model, synthetic stage-1 data)."""
from __future__ import annotations

import json

import pytest
import torch

from evals.tap_live import decode_bench as db
from evals.tap_live import generate_calls as gc
from evals.tap_live import generate_calls_stage2 as g2
from evals.tap_live import main_complete_call as mc
from evals.tap_live import select_stage2 as ss
from evals.tap_live import verify_stage2 as vs
from evals.tap_live.test_decode_bench import artifact, histories, traj_dir  # noqa: F401  (fixtures)


# ---- selection rule -----------------------------------------------------------------------------------------------------------

def hrec(hid, sha, tokens=100):
    t, d = hid.split("@")
    return {"status": "ok", "history_id": hid, "task": t, "depth_fraction": float(d), "call_index": 3, "prompt_sha256": sha, "prompt_tokens": tokens}


def crow(hid, rep, arm, fin="natural", status="ok", seed=1, h="h", el=10.0, n=100):
    return {"history_id": hid, "task": hid.split("@")[0], "depth": float(hid.split("@")[1]), "repeat": rep, "arm": arm, "status": status,
            "finish_reason": fin if status == "ok" else None, "seed": seed, "token_hash": h, "elapsed_s": el if status == "ok" else None, "n_generated": n}


SEEDS = {"t3@0.25": 7, "t4@0.25": 7, "t5@0.25": 8}   # prompt p4 is shared by t3 (the representative, seed 7), t4 (seed 7) and t5 (seed 8)


def synth_stage1(tmp_path):
    d = tmp_path / "s1"
    d.mkdir()
    hs = [hrec("t1@0.25", "p1"), hrec("t1@0.5", "p2"), hrec("t2@0.25", "p3"), hrec("t3@0.25", "p4"), hrec("t4@0.25", "p4"), hrec("t5@0.25", "p4"),
          hrec("t6@0.25", "p6"), hrec("t7@0.25", "p7"), hrec("t8@0.25", "p7")]          # manifest order = this order; 9 positions, 6 distinct prompts
    rows = []
    for h in hs:
        hid = h["history_id"]
        s = SEEDS.get(hid, 1)
        rows += [crow(hid, 0, "A0", seed=s), crow(hid, 0, "A1", seed=s), crow(hid, 1, "A0", seed=s), crow(hid, 1, "A1", seed=s)]

    def put(hid, rep, arm, **kw):
        r = next(r for r in rows if (r["history_id"], r["repeat"], r["arm"]) == (hid, rep, arm))
        r.update(kw)
    put("t1@0.5", 0, "A1", finish_reason="cap")                              # representative of p2: a cap hit in A1 only triggers
    put("t2@0.25", 1, "A0", finish_reason="cap")                              # a cap hit in repeat 1 only does not
    put("t3@0.25", 0, "A0", status="error", finish_reason=None, elapsed_s=None)     # representative of p4 failed: triggers
    put("t4@0.25", 0, "A0", finish_reason="cap")                              # non-representative of p4 (already triggered)
    put("t6@0.25", 0, "A1", token_hash="different")                           # representative of p6: natural, but the A0/A1 streams differ: reported
    put("t8@0.25", 0, "A0", finish_reason="cap")                              # NON-representative of p7 hit the cap: irrelevant for the main result
    (d / "run_config.json").write_text(json.dumps({"config_sha256": "c" * 64, "config": {"arms": ["A0", "A1"], "cap": 4096, "repeats": 2}}))
    (d / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    hp = tmp_path / "histories.json"
    hp.write_text(json.dumps({"histories": hs}))
    return d, hp


def test_capped_only_representative_mode_triggers_only_on_the_representative(tmp_path):
    d, hp = synth_stage1(tmp_path)
    r = ss.select(d, hp, "representative")
    assert r["mode"] == "representative" and r["primary"] and r["n_positions"] == 9 and all(p["weight"] == pytest.approx(1 / 9) for p in r["positions"])
    pos = {p["history_id"]: p for p in r["positions"]}
    assert len(pos) == 9 and r["summary"]["distinct_prompts"] == 6                              # every position is mapped, weights preserved
    assert {p["representative"] for p in r["positions"]} == {"t1@0.25", "t1@0.5", "t2@0.25", "t3@0.25", "t6@0.25", "t7@0.25"}
    # the representative is the first position of the group in manifest order
    assert pos["t4@0.25"]["representative"] == pos["t5@0.25"]["representative"] == "t3@0.25" and pos["t8@0.25"]["representative"] == "t7@0.25"
    # trigger: only the representative's repeat-0 outcome; groups p2 and p4 need stage 2
    assert {k for k, p in pos.items() if p["stage"] == "stage2"} == {"t1@0.5", "t3@0.25", "t4@0.25", "t5@0.25"}
    assert pos["t5@0.25"]["stage"] == "stage2"          # t5 itself completed naturally, but its representative failed: it reuses the representative's rerun
    assert pos["t8@0.25"]["stage"] == "stage1_r0" and pos["t2@0.25"]["stage"] == "stage1_r0"   # a non-representative cap hit / a repeat-1 cap hit do not trigger
    assert "A1: ended at the cap" in pos["t1@0.5"]["reasons"][0] and "failed (error)" in pos["t3@0.25"]["reasons"][0]
    runs = {x["run_key"]: x for x in r["runs"]}
    assert set(runs) == {"p2", "p4"} and runs["p4"]["history_id"] == "t3@0.25" and runs["p4"]["seed"] == 7 and runs["p4"]["positions"] == ["t3@0.25", "t4@0.25", "t5@0.25"]
    assert pos["t5@0.25"]["seed"] == 7 and pos["t5@0.25"]["own_seed"] == 8                      # the group realization uses the representative's seed
    # representative_seed_reuse marks the non-representative labels, and is documented as not a mismatch
    assert {k for k, p in pos.items() if p["representative_seed_reuse"]} == {"t4@0.25", "t5@0.25", "t8@0.25"} and "never denotes a mismatch" in r["representative_seed_reuse_note"]
    s = r["summary"]
    assert s["representatives_to_rerun"] == 2 and s["positions_to_rerun"] == 4 and s["positions_reusing_a_representative_seed"] == 3
    c = r["non_rerun_check"]
    assert c["n"] == 4 and c["all_both_natural"] and c["not_identical"] == ["t6@0.25"]           # the representative pairs of the non-rerun groups
    sens = r["sensitivity_evidence"]["own_outcome_differs_from_representative"]
    assert {x["history_id"] for x in sens} == {"t5@0.25", "t8@0.25"}                              # kept as sensitivity evidence only
    assert pos["t8@0.25"]["own_repeat0_sensitivity"]["A0"]["finish_reason"] == "cap"
    assert len(r["groups"]) == 6 and {g["prompt_sha256"]: g["n_positions"] for g in r["groups"]}["p4"] == 3
    assert r["rule_sha256"] == ss.RULE_SHA256 and r["stage1"]["run_config_sha256"] == "c" * 64 and r["histories_sha256"]
    e = r["expected_tokens"]
    assert e["representatives_to_rerun"] == 2 and e["stage2_tokens_lower_bound_per_arm"] == 2 * 4096
    assert e["stage1_tokens_of_representative_pairs_that_completed"] == {"A0": 400, "A1": 400}      # 4 completed representative pairs x 100 tokens
    ss.main(["--stage1", str(d), "--histories", str(hp), "--out", str(tmp_path / "l.json"), "--mode", "representative"])
    assert json.loads((tmp_path / "l.json").read_text())["summary"] == s


def test_default_mode_runs_every_representative_and_does_not_depend_on_stage_1(tmp_path, capsys):
    d, hp = synth_stage1(tmp_path)
    r = ss.select(None, hp)                                          # no stage-1 directory needed
    assert r["mode"] == "all_representatives" and r["primary"] and r["rule_sha256"] == ss.RULE_ALL_SHA256 and r["n_positions"] == 9
    assert [g["representative"] for g in r["groups"]] == ["t1@0.25", "t1@0.5", "t2@0.25", "t3@0.25", "t6@0.25", "t7@0.25"]         # manifest order
    assert [g["representative_index"] for g in r["groups"]] == [0, 1, 2, 3, 4, 5]
    assert [g["arm_order"] for g in r["groups"]] == [["A0", "A1"], ["A1", "A0"]] * 3                                        # alternating by index
    assert len(r["runs"]) == 6 and [x["representative_index"] for x in r["runs"]] == [0, 1, 2, 3, 4, 5] and r["summary"]["representatives_to_run"] == 6
    assert all(p["stage"] == "stage2" and p["weight"] == pytest.approx(1 / 9) for p in r["positions"]) and sum(g["weight"] for g in r["groups"]) == pytest.approx(1.0)
    g4 = next(g for g in r["groups"] if g["prompt_sha256"] == "p4")
    assert g4["positions"] == ["t3@0.25", "t4@0.25", "t5@0.25"] and g4["weight"] == pytest.approx(3 / 9)
    assert g4["seed"] == gc.seed_call("t3@0.25", 0)                  # the representative's stage-1 repeat-0 seed, computed
    pos = {p["history_id"]: p for p in r["positions"]}
    assert pos["t5@0.25"]["seed"] == g4["seed"] != pos["t5@0.25"]["own_seed"] and pos["t5@0.25"]["representative_seed_reuse"] and not pos["t3@0.25"]["representative_seed_reuse"]
    # the list is identical whatever stage 1 holds: byte-identical files with and without it, and after stage 1 changes
    ss.main(["--histories", str(hp), "--out", str(tmp_path / "a.json")])
    d2 = tmp_path / "s1b"
    d2.mkdir()
    (d2 / "run_config.json").write_text((d / "run_config.json").read_text())
    (d2 / "calls.jsonl").write_text("")
    ss.main(["--histories", str(hp), "--out", str(tmp_path / "b.json"), "--stage1", str(d2), "--consistency-out", str(tmp_path / "c.json")])
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    # the consistency file (separate) records, for each group, whether its stage-1 repeat-0 pair exists and ended naturally
    ss.main(["--histories", str(hp), "--out", str(tmp_path / "e.json"), "--stage1", str(d), "--consistency-out", str(tmp_path / "c2.json")])
    cons = json.loads((tmp_path / "c2.json").read_text())
    by = {c["representative"]: c for c in cons["groups"]}
    assert (tmp_path / "e.json").read_bytes() == (tmp_path / "a.json").read_bytes()
    assert by["t1@0.25"]["pair_exists"] and by["t1@0.25"]["pair_natural"] and by["t1@0.5"]["pair_exists"] and not by["t1@0.5"]["pair_natural"]
    assert not by["t3@0.25"]["pair_natural"] and by["t3@0.25"]["seed_matches"] is False                     # (synthetic seeds differ from the computed ones)
    assert cons["n_pairs_exist"] == 6 and json.loads((tmp_path / "c.json").read_text())["n_pairs_exist"] == 0
    assert "6 representatives to run" in capsys.readouterr().out


def test_recorded_lengths_give_the_expected_token_total_and_the_cli_prints_it(tmp_path, capsys):
    d, hp = synth_stage1(tmp_path)
    lens = tmp_path / "lens.json"
    lens.write_text(json.dumps({"t1@0.5": 9000, "t3@0.25": 3000}))
    ss.main(["--stage1", str(d), "--histories", str(hp), "--out", str(tmp_path / "l.json"), "--recorded-lengths", str(lens), "--tok-per-s", "20", "--mode", "representative"])
    out = capsys.readouterr().out
    e = json.loads((tmp_path / "l.json").read_text())["expected_tokens"]
    assert e["stage2_tokens_per_arm"] == 12000 and e["stage2_tokens_both_arms"] == 24000 and e["stage2_hours_both_arms"] == pytest.approx(24000 / 20 / 3600)
    assert "6 distinct prompts" in out and "2 representatives to rerun" in out and "from recorded lengths: 12000 tokens per arm" in out
    lens.write_text(json.dumps({"t1@0.25": 1000, "t1@0.5": 9000, "t2@0.25": 500, "t3@0.25": 3000, "t6@0.25": 100, "t7@0.25": 2000}))      # the default mode: all 6
    ss.main(["--histories", str(hp), "--out", str(tmp_path / "m.json"), "--recorded-lengths", str(lens), "--tok-per-s", "20"])
    out = capsys.readouterr().out
    e = json.loads((tmp_path / "m.json").read_text())["expected_tokens"]
    assert e["stage2_tokens_per_arm"] == 15600 and e["stage2_tokens_both_arms"] == 31200 and not e["missing_recorded_lengths"]
    assert "6 distinct prompts" in out and "6 representatives to run" in out and "list sha256" in out and "15600 per arm" in out


def test_position_seed_mode_is_a_labelled_sensitivity_option(tmp_path, capsys):
    d, hp = synth_stage1(tmp_path)
    r = ss.select(d, hp, "position_seed")
    assert r["mode"] == "position_seed" and not r["primary"] and r["rule_sha256"] == ss.RULE_POSITION_SEED_SHA256 and "SENSITIVITY, not primary" in r["rule"]
    pos = {p["history_id"]: p for p in r["positions"]}
    assert {k for k, p in pos.items() if p["stage"] == "stage2"} == {"t1@0.5", "t3@0.25", "t4@0.25", "t8@0.25"}      # own outcomes trigger
    runs = {x["run_key"]: x for x in r["runs"]}
    assert set(runs) == {"p2:1", "p4:7", "p7:1"} and runs["p4:7"]["positions"] == ["t3@0.25", "t4@0.25"]            # (prompt, seed) dedupe only
    assert pos["t4@0.25"]["representative_seed_reuse"] is True and pos["t3@0.25"]["representative_seed_reuse"] is False
    with pytest.raises(ValueError):
        ss.select(d, hp, "bogus")
    ss.main(["--stage1", str(d), "--histories", str(hp), "--out", str(tmp_path / "l.json"), "--mode", "position_seed"])
    assert "SENSITIVITY, not primary" in capsys.readouterr().out


# ---- stage 2 on the tiny model ------------------------------------------------------------------------------------------------

def small_histories(histories, tmp_path):
    doc = json.loads(histories[0].read_text())
    ok = [h for h in doc["histories"] if h["status"] == "ok"][:2]
    p = tmp_path / "h2.json"
    p.write_text(json.dumps({**doc, "histories": ok, "n_ok": 2}))
    return p


def forbid_stop(tok):
    stops = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]

    def proc(logits, gen, ci):
        out = logits.clone()
        out[stops] = -1e9
        return out
    return proc


def stop_after(tok, n):
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    forbid = forbid_stop(tok)

    def proc(logits, gen, ci):
        if len(gen) < n:
            return forbid(logits, gen, ci)
        out = torch.full_like(logits, float("-inf"))
        out[stop] = 0.0
        return out
    return proc


def stage1_run(model, tok, fmt, h2, artifact, out, processor):
    gc.main(["--model", "tiny", "--histories", str(h2), "--artifact", str(artifact), "--out", str(out), "--device", "cpu", "--dtype", "float32",
             "--repeats", "2", "--cap", "6", "--no-warmup"], inject={"model": model, "tokenizer": tok, "fmt": fmt, "logits_processor": processor})


def stage2_args(h2, artifact, lst, out, total, extra=()):
    return ["--model", "tiny", "--histories", str(h2), "--artifact", str(artifact), "--stage2-list", str(lst), "--out", str(out), "--device", "cpu",
            "--dtype", "float32", "--max-total-tokens", str(total), "--no-warmup", *extra]


def test_stage2_same_seed_prefix_natural_stop_and_verifier(model, tok, fmt, histories, artifact, tmp_path, capsys, monkeypatch):
    h2 = small_histories(histories, tmp_path)
    s1 = tmp_path / "s1"
    stage1_run(model, tok, fmt, h2, artifact, s1, forbid_stop(tok))          # never stops: both hit the stage-1 cap of 6
    assert all(json.loads(x)["finish_reason"] == "cap" for x in (s1 / "calls.jsonl").read_text().splitlines())
    lst = tmp_path / "list.json"
    ss.main(["--stage1", str(s1), "--histories", str(h2), "--out", str(lst)])
    sel = json.loads(lst.read_text())
    assert sel["summary"]["positions_to_rerun"] == 2 and sel["summary"]["representatives_to_run"] == 2
    monkeypatch.setattr(g2, "PROGRESS_EVERY", 4)
    s2 = tmp_path / "s2"
    inj = {"model": model, "tokenizer": tok, "fmt": fmt, "logits_processor": stop_after(tok, 10)}       # the natural stop comes at token 11
    assert g2.main(stage2_args(h2, artifact, lst, s2, 400, ["--stage1", str(s1)]), inject=inj) == 0
    rows = [json.loads(x) for x in (s2 / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(r["status"] == "ok" and r["finish_reason"] == "natural" and r["n_generated"] == 11 and r["stage"] == "stage2" for r in rows)
    assert all(r["seed"] == gc.seed_call(r["history_id"], 0) for r in rows)                  # the stage-1 seed
    assert [r["arm"] for r in rows if r["order_position"] == 0] == ["A0", "A1"]            # alternating by representative index: A0 first for index 0, A1 first for index 1
    assert all(r["representative_index"] in (0, 1) and r["parse"]["parse_failure"] is False for r in rows)
    out = capsys.readouterr().out
    assert "8 tokens generated" in out and "memory estimate" in out and "run 1/2" in out
    o = [json.loads(x) for x in (s2 / "outputs.jsonl").read_text().splitlines()]
    assert len(o) == 2 and all(x["finish_reason"] == "natural" and x["text_with_special_tokens"].endswith("<|im_end|>") and x["n_tokens"] == 11 for x in o)
    assert [x["arm_order"] for x in o] == [["A0", "A1"], ["A1", "A0"]] and all(x["arm"] == "A0" and x["token_hash"] for x in o)    # A1-first pairs still save A0's output
    cfg = json.loads((s2 / "run_config.json").read_text())["config"]
    assert cfg["stage2_rule_sha256"] == ss.RULE_ALL_SHA256 and "no per-call cap" in cfg["cap_policy"] and cfg["selection_mode"] == "all_representatives"
    assert cfg["arm_order_rule"].startswith("alternating by representative index") and list(cfg["arm_orders"].values()) == [["A0", "A1"], ["A1", "A0"]]
    assert cfg["stage1_run_config_sha256"] == json.loads((s1 / "run_config.json").read_text())["config_sha256"] or cfg["stage1_run_config_sha256"] is None
    v = vs.verify(s1, s2, sel)
    assert v["all_checks_pass"] and not v["mismatches"]
    pre = [i for i in v["items"] if i.get("compared_tokens")]
    assert pre and all(i["compared_tokens"] == 6 and i["prefix_match"] for i in pre)           # the capped stage-1 stream is a prefix
    # a deliberate mismatch in the prefix, and an A0/A1 disagreement, are flagged
    lines = (s2 / "outputs.jsonl").read_text().splitlines()
    bad = json.loads(lines[0])
    bad["token_ids"][3] = (bad["token_ids"][3] + 1) % 300
    (s2 / "outputs.jsonl").write_text("\n".join([json.dumps(bad)] + lines[1:]) + "\n")
    v2 = vs.verify(s1, s2, sel)
    assert not v2["all_checks_pass"]
    kinds = {(m["run_key"], m.get("arm"), m.get("check")): m for m in v2["mismatches"]}
    assert any(m.get("arm") == "A0" and m["first_mismatch_index"] == 3 for m in v2["mismatches"])
    assert any(m.get("check") == "stage-2 A0 vs A1" and m["identical"] is False and m["first_mismatch_index"] == 3 for m in v2["mismatches"]), kinds


def test_context_ceiling_is_recorded_and_never_natural(model, tok, fmt, histories, artifact, tmp_path):
    h2 = small_histories(histories, tmp_path)
    s1 = tmp_path / "s1"
    stage1_run(model, tok, fmt, h2, artifact, s1, forbid_stop(tok))
    lst = tmp_path / "list.json"
    ss.main(["--stage1", str(s1), "--histories", str(h2), "--out", str(lst)])
    sel = json.loads(lst.read_text())
    total = max(h["prompt_tokens"] for h in json.loads(h2.read_text())["histories"]) + 14
    s2 = tmp_path / "s2"
    g2.main(stage2_args(h2, artifact, lst, s2, total), inject={"model": model, "tokenizer": tok, "fmt": fmt, "logits_processor": forbid_stop(tok)})
    rows = [json.loads(x) for x in (s2 / "calls.jsonl").read_text().splitlines()]
    assert all(r["finish_reason"] == "context_ceiling" and r["hit_context_ceiling"] and not r["stopped_naturally"] for r in rows)
    assert all(r["prompt_tokens"] + r["n_generated"] == total for r in rows)                  # prompt + generated = the ceiling
    assert vs.verify(s1, s2, sel)["all_checks_pass"]
    r = mc.build(s1, s2, sel)
    assert r["counts"]["context_ceiling_hits_a0"] == 2 and r["counts"]["natural_stops_a0"] == 0 and "context-ceiling hits A0 2" in mc.to_markdown(r)
    # one position per run: the A1 decisions pooled every generated token (no stop token)
    a1 = next(x for x in rows if x["arm"] == "A1")
    assert a1["decisions"][0]["n_tokens"] == a1["n_generated"]


def test_progress_wrapper_prints_counts_only_and_keeps_the_logits(capsys, monkeypatch):
    monkeypatch.setattr(g2, "PROGRESS_EVERY", 3)
    p = g2.progress_wrapper("x y")
    logits = torch.zeros(5)
    assert p(logits, [], 0) is logits and p(logits, [1, 2], 0) is logits and capsys.readouterr().out == ""
    assert p(logits, [1, 2, 3], 0) is logits and "[x y] 3 tokens generated" in capsys.readouterr().out


def test_kv_memory_estimate_counts_full_attention_layers_only(model):
    tc = model.config.get_text_config()
    per = g2.kv_bytes_per_token(model)
    n_full = sum(t == "full_attention" for t in tc.layer_types)
    assert per == n_full * 2 * tc.num_key_value_heads * tc.head_dim * 4 and n_full == 1               # tiny fp32 model: one full-attention layer
    # the real Qwen3.8-27B: 16 of 64 layers, 4 kv heads, head dim 256, bf16
    assert 16 * 2 * 4 * 256 * 2 == 65536
    e = g2.memory_estimate(model, 47000, 100000, 5120, 54 * 1024 ** 3)
    assert e["kv_bytes_per_token"] == per and e["total_GiB"] > e["weights_GiB"]


def test_resume_refuses_a_changed_config_and_changed_inputs(model, tok, fmt, histories, artifact, tmp_path):
    h2 = small_histories(histories, tmp_path)
    s1 = tmp_path / "s1"
    stage1_run(model, tok, fmt, h2, artifact, s1, forbid_stop(tok))
    lst = tmp_path / "list.json"
    ss.main(["--stage1", str(s1), "--histories", str(h2), "--out", str(lst)])
    inj = {"model": model, "tokenizer": tok, "fmt": fmt, "logits_processor": stop_after(tok, 10)}
    s2 = tmp_path / "s2"
    g2.main(stage2_args(h2, artifact, lst, s2, 400), inject=inj)
    n = len((s2 / "calls.jsonl").read_text().splitlines())
    cfg_bytes = (s2 / "run_config.json").read_bytes()
    assert g2.main(stage2_args(h2, artifact, lst, s2, 400), inject=inj) == 0 and len((s2 / "calls.jsonl").read_text().splitlines()) == n
    with pytest.raises(SystemExit, match="refusing to resume"):
        g2.main(stage2_args(h2, artifact, lst, s2, 401), inject=inj)
    sel = json.loads(lst.read_text())
    sel["rule_sha256"] = "0" * 64                                               # a different stage-2 list: different config
    other = tmp_path / "list2.json"
    other.write_text(json.dumps(sel))
    with pytest.raises(SystemExit, match="refusing to resume"):
        g2.main(stage2_args(h2, artifact, other, s2, 400), inject=inj)
    assert (s2 / "run_config.json").read_bytes() == cfg_bytes
    other_h = tmp_path / "other_hist.json"
    other_h.write_text(h2.read_text() + " ")
    with pytest.raises(SystemExit, match="differs from the one the stage-2 list"):
        g2.main(stage2_args(other_h, artifact, lst, tmp_path / "s3", 400), inject=inj)


# ---- main table ---------------------------------------------------------------------------------------------------------------

def test_capped_only_mode_main_table_uses_the_representative_pair_for_every_position(tmp_path):
    d, hp = synth_stage1(tmp_path)
    hs = [h["history_id"] for h in json.loads(hp.read_text())["histories"]]
    rows = []
    for i, h in enumerate(hs):                               # stage-1 times: position i has A0 = 10 + i, A1 = A0 + 0.01 (repeat 0) and A0 + 0.02 (repeat 1)
        for rep, delta in ((0, 0.01), (1, 0.02)):
            rows += [crow(h, rep, "A0", el=10.0 + i, seed=SEEDS.get(h, 1), h="x"), crow(h, rep, "A1", el=10.0 + i + delta, seed=SEEDS.get(h, 1), h="x")]
    for r in rows:                                           # the capped repeat-0 calls of the rerun groups (bounded values that must not enter the primary)
        if r["history_id"] in ("t1@0.5", "t3@0.25", "t4@0.25") and r["repeat"] == 0:
            r["finish_reason"], r["elapsed_s"], r["n_generated"] = "cap", 1.0, 4096
        if r["history_id"] == "t8@0.25" and r["repeat"] == 0 and r["arm"] == "A0":
            r["finish_reason"], r["elapsed_s"], r["n_generated"] = "cap", 2.0, 4096          # a non-representative label's own cap: irrelevant for the primary
    (d / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sel = ss.select(d, hp, "representative")
    s2 = tmp_path / "s2"
    s2.mkdir()

    def s2row(key, arm, el, fin="natural", n=9000, status="ok"):
        return {"run_key": key, "arm": arm, "status": status, "finish_reason": fin if status == "ok" else None, "elapsed_s": el if status == "ok" else None,
                "n_generated": n, "token_hash": "z"}
    srows = [s2row("p2", "A0", 400.0), s2row("p2", "A1", 400.5), s2row("p4", "A0", 100.0, n=3000), s2row("p4", "A1", 100.3, n=3000)]
    (s2 / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in srows))
    r = mc.build(d, s2, sel)
    pos = {p["history_id"]: p for p in r["positions"]}
    assert r["complete"] and r["n_valid_positions"] == 9 and r["distinct_prompts"] == 6 and r["positions_by_source"] == {"stage1_r0": 5, "stage2": 4}
    # every position of a group takes the representative's pair (t3's rerun for t3, t4, t5; t7's stage-1 pair for t7 and t8)
    assert pos["t3@0.25"]["a0_s"] == pos["t4@0.25"]["a0_s"] == pos["t5@0.25"]["a0_s"] == 100.0 and pos["t4@0.25"]["source_stage"] == "stage2"
    assert pos["t8@0.25"]["a0_s"] == pos["t7@0.25"]["a0_s"] == 10.0 + hs.index("t7@0.25") and pos["t8@0.25"]["source_stage"] == "stage1_r0"
    assert pos["t8@0.25"]["representative"] == "t7@0.25" and pos["t8@0.25"]["representative_seed_reuse"] and not pos["t7@0.25"]["representative_seed_reuse"]
    assert pos["t1@0.5"]["a0_s"] == 400.0 and pos["t1@0.25"]["a0_s"] == 10.0
    assert all(p["weight"] == pytest.approx(1 / 9) for p in r["positions"])
    a0 = [pos[h]["a0_s"] for h in hs]
    p = r["primary"]
    assert p["no_tap_mean_s_per_call"]["estimate"] == pytest.approx(sum(a0) / 9)                  # equal preselected weights 1 / 9, shared representative counted per position
    assert p["with_probe_mean_s_per_call"]["estimate"] == pytest.approx(sum(pos[h]["a1_s"] for h in hs) / 9)
    exp = sum(1000 * (pos[h]["a1_s"] - pos[h]["a0_s"]) for h in hs) / 9
    assert p["paired_difference_ms_per_call"]["estimate"] == pytest.approx(exp)
    lo, hi = p["paired_difference_ms_per_call"]["ci95"]
    assert lo <= exp <= hi and p["n_tasks"] == 8 and "task" in p["bootstrap"]["unit"]            # tasks, not 9 (or 30) independent prompts, are resampled
    assert r["groups_spanning_several_tasks"] == 2 and len(r["groups"]) == 6 and r["positions_reusing_a_representative_seed"] == 3
    g4 = next(g for g in r["groups"] if g["prompt_sha256"] == "p4")
    assert g4["representative"] == "t3@0.25" and g4["positions"] == ["t3@0.25", "t4@0.25", "t5@0.25"] and g4["token_hash_a0"] == "z" == g4["token_hash_a1"]
    assert g4["source_stage"] == "stage2" and g4["finish_reason_a0"] == "natural" and g4["seed"] == 7
    # sensitivity tables are separate and labelled; the own-seed table uses each position's own capped repeat-0 pair
    s_rep, s_own = r["sensitivity_stage1_repeat1"], r["sensitivity_own_seed_repeat0"]
    assert "SENSITIVITY, not the primary" in s_rep["label"] and "SENSITIVITY, not the primary" in s_own["label"] and s_own["cap_hits_a0"] >= 3
    assert s_rep["paired_difference_ms_per_call"]["estimate"] == pytest.approx(20.0) and p["paired_difference_ms_per_call"]["estimate"] != pytest.approx(20.0)
    md = mc.to_markdown(r)
    assert "Sensitivity (not the primary)" in md and "ever chosen" in md and "representative_seed_reuse" in md and "6 distinct prompts" in md
    assert "never a mismatch between A0 and A1" in md
    # a missing stage-2 pair makes the primary incomplete and says so; it is never replaced by a stage-1 realization
    (s2 / "calls.jsonl").write_text("".join(json.dumps(x) + "\n" for x in srows[:2]))
    r2 = mc.build(d, s2, sel)
    assert not r2["complete"] and set(r2["missing_positions"]) == {"t3@0.25", "t4@0.25", "t5@0.25"} and r2["n_valid_positions"] == 6
    assert "INCOMPLETE" in mc.to_markdown(r2)
    mc.main(["--stage1", str(d), "--stage2", str(s2), "--stage2-list", str(_write(tmp_path / "l.json", sel)), "--out", str(tmp_path / "o")])
    assert (tmp_path / "o" / "main_complete_call.json").exists()


def test_the_tasks_are_the_resampling_unit_when_a_representative_is_shared(tmp_path):
    # four positions of ONE task share one representative pair: with task resampling the interval is that of one task, not of four independent prompts
    items = [{"task": "a", "weight": 0.25, "a0_s": 10.0, "a1_s": 10.001}] * 2 + [{"task": "b", "weight": 0.25, "a0_s": 10.0, "a1_s": 10.003}] * 2
    e = mc.weighted_estimates(items)
    assert e["n_tasks"] == 2 and e["paired_difference_ms_per_call"]["estimate"] == pytest.approx(2.0)
    lo, hi = e["paired_difference_ms_per_call"]["ci95"]
    assert lo == pytest.approx(1.0) and hi == pytest.approx(3.0)             # with two tasks the resampled means are 1, 2 or 3 ms: wide, as it should be


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def test_main_table_of_the_main_stage_uses_stage_2_only_and_stage_1_is_supporting(tmp_path):
    d, hp = synth_stage1(tmp_path)
    sel = ss.select(None, hp)
    hs = [h["history_id"] for h in json.loads(hp.read_text())["histories"]]
    s2 = tmp_path / "s2"
    s2.mkdir()
    reps = [g["representative"] for g in sel["groups"]]
    srows = []
    for i, g in enumerate(sel["groups"]):
        key = g["prompt_sha256"]
        base = 100.0 * (i + 1)
        fin = "context_ceiling" if i == 5 else "natural"
        srows += [{"run_key": key, "arm": "A0", "status": "ok", "finish_reason": fin, "elapsed_s": base, "n_generated": 1000 * (i + 1), "token_hash": f"h{i}"},
                  {"run_key": key, "arm": "A1", "status": "ok", "finish_reason": fin, "elapsed_s": base + 0.01 * (i + 1), "n_generated": 1000 * (i + 1), "token_hash": f"h{i}"}]
    (s2 / "calls.jsonl").write_text("".join(json.dumps(x) + "\n" for x in srows))
    r = mc.build(d, s2, sel)
    pos = {p["history_id"]: p for p in r["positions"]}
    assert r["primary_uses_stage1_pairs"] == 0 and r["positions_by_source"] == {"stage1_r0": 0, "stage2": 9} and r["complete"] and r["distinct_prompts"] == 6
    # every position takes its group's representative pair of THIS stage: t3, t4, t5 share group 3 (A0 = 400 s); the stage-1 times (10 s) are not used
    assert pos["t3@0.25"]["a0_s"] == pos["t4@0.25"]["a0_s"] == pos["t5@0.25"]["a0_s"] == 400.0 and pos["t8@0.25"]["a0_s"] == 600.0 == pos["t7@0.25"]["a0_s"]
    a0 = [pos[h]["a0_s"] for h in hs]
    p = r["primary"]
    assert p["no_tap_mean_s_per_call"]["estimate"] == pytest.approx(sum(a0) / 9) and p["n_tasks"] == 8
    assert p["paired_difference_ms_per_call"]["estimate"] == pytest.approx(sum(1000 * (pos[h]["a1_s"] - pos[h]["a0_s"]) for h in hs) / 9)
    assert r["counts"]["context_ceiling_hits_a0"] == 2 and r["counts"]["natural_stops_a0"] == 7 and r["lengths"]["max"] == 6000        # t7 and t8 share the ceiling group
    sp = r["supporting_stage1_pairs"]
    assert "SUPPORTING, not the primary" in sp["label"] and sp["n_groups"] == 6
    row = {x["representative"]: x for x in sp["groups"]}
    assert row["t1@0.25"]["stage1_pair_natural"] and row["t1@0.25"]["stage1_a0_s"] == 10.0 and row["t1@0.25"]["main_a0_s"] == 100.0           # times need not agree
    assert row["t1@0.25"]["main_vs_stage1_token_stream_identical"] is False                                                                  # synthetic hashes differ: reported
    assert not row["t3@0.25"]["stage1_pair_natural"] and row["t3@0.25"]["main_vs_stage1_token_stream_identical"] is None
    md = mc.to_markdown(r)
    assert "Supporting stage-1 pairs (not the primary)" in md and "Primary positions taken from stage 1: 0" in md


def test_verifier_requires_identical_full_natural_streams(model, tok, fmt, histories, artifact, tmp_path):
    h2 = small_histories(histories, tmp_path)
    s1 = tmp_path / "s1"
    gc.main(["--model", "tiny", "--histories", str(h2), "--artifact", str(artifact), "--out", str(s1), "--device", "cpu", "--dtype", "float32", "--repeats", "1", "--cap", "40",
             "--no-warmup"], inject={"model": model, "tokenizer": tok, "fmt": fmt, "logits_processor": stop_after(tok, 10)})
    assert all(json.loads(x)["finish_reason"] == "natural" for x in (s1 / "calls.jsonl").read_text().splitlines())      # stage 1 stopped naturally after 11 tokens
    lst = tmp_path / "list.json"
    ss.main(["--histories", str(h2), "--out", str(lst)])
    sel = json.loads(lst.read_text())
    s2 = tmp_path / "s2"
    g2.main(stage2_args(h2, artifact, lst, s2, 400), inject={"model": model, "tokenizer": tok, "fmt": fmt, "logits_processor": stop_after(tok, 10)})
    v = vs.verify(s1, s2, sel)
    assert v["all_checks_pass"] and not v["mismatches"]
    full = [i for i in v["items"] if i.get("comparison") == "full natural stream identical"]
    assert len(full) == 4 and all(i["prefix_match"] and i["stage1_n_tokens"] == i["stage2_n_tokens"] == 11 for i in full)
    # a stage-2 stream that is LONGER than the stage-1 natural stream (identical prefix) is a mismatch: natural streams must be identical
    lines = (s2 / "outputs.jsonl").read_text().splitlines()
    longer = json.loads(lines[0])
    longer["token_ids"] = longer["token_ids"] + [5]
    (s2 / "outputs.jsonl").write_text("\n".join([json.dumps(longer)] + lines[1:]) + "\n")
    v2 = vs.verify(s1, s2, sel)
    assert not v2["all_checks_pass"] and any(m.get("arm") == "A0" and m["first_mismatch_index"] == 11 and m["stage1_natural"] for m in v2["mismatches"])
    # without any stage-1 output the comparison is unavailable (stage 1 was curtailed), not a failure
    empty = tmp_path / "s1empty"
    empty.mkdir()
    (empty / "calls.jsonl").write_text("")
    v3 = vs.verify(empty, tmp_path / "s2", sel)
    assert v3["mismatches"] == [] or all(m.get("check") == "stage-2 A0 vs A1" for m in v3["mismatches"])
    assert any("no stage-1 output ids" in (u.get("note") or "") for u in v3["unavailable"])
