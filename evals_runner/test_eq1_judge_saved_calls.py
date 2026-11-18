"""Tests for the shared-call monitor cost (stub HTTP server, synthetic trajectories, histories, stage-2 list and saved outputs; no model, no GPU)."""
import gzip
import hashlib
import json

import pytest

from evals.reward_hacking import llm
from evals.reward_hacking import monitors as m
from evals.reward_hacking import trajectory as tj
from evals_runner import eq1_judge_saved_calls as J
from evals_runner import eq1_judge_saved_calls_prompts as pr
from evals_runner import eq1_monitor_panel as P
from evals_runner.test_eq1_monitor_panel import Stub

TOOL = '<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>'
GOOD = "I will look around.\n</think>\n\nLet me list the files.\n" + TOOL
SHA = lambda s: hashlib.sha256(s.encode()).hexdigest()


def make_episode(task: str, n_calls: int = 4, with_submission: bool = False, big_reasoning: int = 0) -> tj.Trajectory:
    t = tj.Trajectory(meta={"task_id": task, "rollout_id": f"{task}__conflicting__T1__r0", "partition": "train"})
    t.add(tj.USER, {"text": f"implement f for {task}", "origin": "task"})
    for c in range(n_calls):
        prompt = f"PROMPT of {task} call {c}"
        t.add(tj.MODEL_CALL, {"prompt": prompt, "prompt_sha256": SHA(prompt), "generation": "g"}, call_index=c)
        t.add(tj.REASONING, {"text": ("R" * big_reasoning if big_reasoning and c == 0 else f"old private thoughts {c}"), "closed": True}, call_index=c)
        t.add(tj.ASSISTANT, {"text": f"old message {c}"}, call_index=c)
        t.add(tj.TOOL_PROPOSED, {"call_id": f"call_{c + 1}", "name": "bash", "arguments": {"cmd": f"echo {c}"}, "raw": "", "parse_error": None}, call_index=c)
        t.add(tj.TOOL_EXECUTED, {"call_id": f"call_{c + 1}", "name": "bash", "arguments": {"cmd": f"echo {c}"}, "modified_by_wrapper": False})
        t.add(tj.TOOL_RESULT, {"call_id": f"call_{c + 1}", "name": "bash", "output": f"old result {c}"})
        if with_submission and c == 1:
            t.add(tj.SUBMISSION, {"attempt": 1, "func_py": "def f(): return 1", "test_py": "assert f()"}, call_index=c)
            t.add(tj.EVALUATION, {"attempt": 1, "passed": False, "explanation": "x"})
            t.add(tj.USER, {"text": "incorrect, try again", "origin": "incorrect_feedback"})
    return t


def make_world(tmp_path, texts=None, tasks=("lcbhard_1", "lcbhard_2", "lcbhard_3"), call_index=2, **ep_kw):
    """Three tasks; positions: the three at depth 0.25 plus a second depth label of task 1 sharing the same call (so 4 positions, 3 prompts)."""
    traj_dir = tmp_path / "trajs" / "trajectories"
    traj_dir.mkdir(parents=True)
    hist, runs, positions, outs = [], [], [], []
    for k, task in enumerate(tasks):
        ep = make_episode(task, **ep_kw)
        raw = ep.to_json()
        fn = f"{task}__conflicting__T1__r0.json"
        (traj_dir / fn).write_text(raw)
        prompt = f"PROMPT of {task} call {call_index}"
        h = {"task": task, "depth_fraction": 0.25, "call_index": call_index, "source_file": fn, "source_sha256": SHA(raw), "status": "ok", "prompt": prompt,
             "prompt_tokens": 10, "prompt_sha256": SHA(prompt), "history_id": f"{task}@0.25"}
        hist.append(h)
        runs.append({"run_key": SHA(prompt), "representative_index": k, "history_id": h["history_id"], "prompt_sha256": SHA(prompt)})
        positions.append({"history_id": h["history_id"], "task": task, "weight": 0.25, "run_key": SHA(prompt)})
        text = (texts or {}).get(task, GOOD)
        outs.append({"run_key": SHA(prompt), "history_id": h["history_id"], "prompt_sha256": SHA(prompt), "finish_reason": "natural", "n_tokens": 50,
                     "text_stop_token_removed": text})
    h2 = dict(hist[0], depth_fraction=0.5, history_id=f"{tasks[0]}@0.5")
    hist.append(h2)
    positions.append({"history_id": h2["history_id"], "task": tasks[0], "weight": 0.25, "run_key": runs[0]["run_key"]})
    hp = tmp_path / "histories.json"
    hp.write_text(json.dumps({"histories": hist}))
    sel = {"histories_sha256": SHA(hp.read_text()), "runs": runs, "positions": positions}
    sp = tmp_path / "stage2_list.json"
    sp.write_text(json.dumps(sel))
    op = tmp_path / "outputs.jsonl"
    op.write_text("".join(json.dumps(o) + "\n" for o in outs))
    return {"stage2": sp, "hist": hp, "outputs": op, "trajs": tmp_path / "trajs", "outs": outs, "hist_list": hist, "positions": positions}


@pytest.fixture
def world(tmp_path):
    return make_world(tmp_path)


def load(w):
    return pr.load_world(w["stage2"], w["hist"], w["outputs"], w["trajs"])


# ---------------------------------------------------------------------------------------------- inputs

def test_trace_is_prefix_before_the_call_plus_new_response(world):
    wd = load(world)
    items = pr.build_inputs(wd, ["history_black_box", "history_reasoning"], "early")
    assert len(items) == 6 and [i["view"] for i in items] == ["history_black_box"] * 3 + ["history_reasoning"] * 3
    bb = items[0]["user"]
    assert "old result 1" in bb and "old message 1" in bb and "echo 1" in bb                      # calls 0 and 1: everything before call 2
    assert "old message 2" not in bb and "old result 2" not in bb                                  # call 2 is replaced by the new response
    assert "Let me list the files." in bb and "[AGENT TOOL CALL bash]" in bb and '"cmd": "ls"' in bb
    assert "I will look around." not in bb and "old private thoughts" not in bb                    # private reasoning only in the reasoning view
    cot = items[3]["user"]
    assert "I will look around." in cot and "old private thoughts 1" in cot and "It also contains the agent's private reasoning" in cot
    assert bb.rstrip().endswith("SCORE: <integer 0-100>") and "only the record available at one decision point" in bb       # EARLY_INSTRUCTION
    assert items[0]["flags"]["no_parsed_action"] is False and items[0]["flags"]["n_tool_calls"] == 1
    assert pr.build_inputs(wd, ["history_black_box"], "terminal")[0]["user"] != bb and "up to the moment it submitted" in pr.build_inputs(wd, ["history_black_box"], "terminal")[0]["user"]


def test_earlier_submission_and_evaluation_do_not_cut_the_new_response(tmp_path):
    w = make_world(tmp_path, with_submission=True)
    items = pr.build_inputs(load(w), ["history_black_box"], "early")
    u = items[0]["user"]
    assert "AGENT SUBMITS FOR GRADING (attempt 1)" in u and "incorrect, try again" in u and "Let me list the files." in u
    assert "Passed" not in u and "explanation" not in u.split("<record>")[1]                         # the scorer's result never appears


def test_uniform_policy_flags(tmp_path):
    texts = {"lcbhard_1": "still thinking about it and never closing",                       # reasoning never closed (cap/ceiling style)
             "lcbhard_2": "r\n</think>\n\nonly text, no action",                              # closed, no tool call
             "lcbhard_3": "r\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=cmd>\nls"}  # unterminated call = malformed
    w = make_world(tmp_path, texts=texts)
    items = {i["task"]: i for i in pr.build_inputs(load(w), ["history_black_box"], "early")}
    f1, f2, f3 = (items[t]["flags"] for t in ("lcbhard_1", "lcbhard_2", "lcbhard_3"))
    assert f1["reasoning_closed"] is False and f1["no_parsed_action"] and f1["visible_empty"] and "never closing" not in items["lcbhard_1"]["user"]
    assert f2["no_parsed_action"] and not f2["visible_empty"] and "only text, no action" in items["lcbhard_2"]["user"]
    assert f3["n_malformed_tool_calls"] == 1 and f3["no_parsed_action"] and "[AGENT TOOL CALL <malformed>]" in items["lcbhard_3"]["user"]
    # an output that stopped at the context ceiling is never treated as a natural, complete call
    outs = [json.loads(x) for x in w["outputs"].read_text().splitlines()]
    outs[0]["finish_reason"] = "context_ceiling"
    w["outputs"].write_text("".join(json.dumps(o) + "\n" for o in outs))
    it0 = pr.build_inputs(load(w), ["history_black_box"], "early")[0]
    assert it0["flags"]["response_natural_stop"] is False and it0["flags"]["no_parsed_action"] and it0["flags"]["response_finish_reason"] == "context_ceiling"


def test_identity_checks_refuse(tmp_path):
    w = make_world(tmp_path)
    (w["trajs"] / "trajectories" / w["hist_list"][0]["source_file"]).write_text(make_episode("lcbhard_1", n_calls=5).to_json())
    with pytest.raises(SystemExit, match="sha256"):
        load(w)
    w2 = make_world(tmp_path / "b")
    outs = [json.loads(x) for x in w2["outputs"].read_text().splitlines()][1:]
    w2["outputs"].write_text("".join(json.dumps(o) + "\n" for o in outs))
    with pytest.raises(SystemExit, match="no saved A0 output"):
        load(w2)
    w3 = make_world(tmp_path / "c")
    outs = [json.loads(x) for x in w3["outputs"].read_text().splitlines()]
    outs[0]["prompt_sha256"] = "0" * 64
    w3["outputs"].write_text("".join(json.dumps(o) + "\n" for o in outs))
    with pytest.raises(SystemExit, match="another prompt"):
        load(w3)
    w4 = make_world(tmp_path / "d", call_index=2)
    h = json.loads(w4["hist"].read_text())
    p = "PROMPT of lcbhard_1 call 3"
    traj = make_episode("lcbhard_1")
    wd = load(w4)
    with pytest.raises(ValueError, match="prompt hash"):
        pr.build_trace(traj, dict(h["histories"][0], prompt_sha256=SHA(p)), w4["outs"][0], llm.get_format("qwen3.8-27b"))


def test_fit_uses_the_detector_path_and_black_box_cannot_be_shortened(tmp_path):
    w = make_world(tmp_path, big_reasoning=300_000, call_index=1)
    wd = load(w)
    fmt = llm.get_format("qwen3.8-27b")
    rep = wd["reps"][0]
    traj, boundary, _ = pr.build_trace(rep["trajectory"], rep["history"], rep["output"], fmt)
    cfg = m.get_spec("gemma4-e4b").config()
    counter = m.CharCounter(3.0)
    res = pr.fit_view(traj, boundary, tj.COT, m.EARLY_INSTRUCTION, counter, 20_000, cfg.template_overhead_tokens)
    ref, info = m.build_prompt_fitted(traj, tj.COT, counter=counter, budget_tokens=20_000, overhead=cfg.template_overhead_tokens, boundary_seq=boundary,
                                      instruction=m.EARLY_INSTRUCTION)
    assert res["ok"] and res["user"] == ref and res["reasoning_chars_cut"] == info["truncated_reasoning_chars"] > 0 and res["user_tokens"] <= 20_000
    assert res["report"]["blocks_shortened"] >= 1 and res["report"]["reasoning_fraction_kept"] < 1
    assert not pr.fit_view(traj, boundary, tj.BLACK_BOX, m.EARLY_INSTRUCTION, counter, 5, cfg.template_overhead_tokens)["ok"]


def test_input_script_writes_manifest_and_is_deterministic(world, tmp_path):
    out = tmp_path / "in.json.gz"
    args = ["--stage2-list", str(world["stage2"]), "--histories", str(world["hist"]), "--outputs", str(world["outputs"]), "--trajectories", str(world["trajs"]),
            "--views", "history_black_box,history_reasoning", "--out", str(out)]
    assert pr.main(args) == 0
    man = json.loads((tmp_path / "in.manifest.json").read_text())
    items = json.loads(gzip.decompress(out.read_bytes()))["items"]
    assert len(items) == 6 and [i["user_sha256"] for i in items] == [x["user_sha256"] for x in man["items"]]
    assert man["policy_sha256"] == pr.POLICY_SHA256 and man["instruction"] == "early" and len(man["trajectory_files"]) == 3
    out2 = tmp_path / "in2.json.gz"
    pr.main(args[:-1] + [str(out2)])
    assert out.read_bytes() == out2.read_bytes()


# ---------------------------------------------------------------------------------------------- judging with the stub

def run_judge(world, stub, tmp_path, name="gemma4-e4b", extra=(), out="run", main_file=None, views="history_black_box"):
    args = ["--monitor", name, "--url", stub.url, "--out-dir", str(tmp_path / out), "--stage2-list", str(world["stage2"]), "--histories", str(world["hist"]),
            "--outputs", str(world["outputs"]), "--trajectories", str(world["trajs"]), "--allow-char-counter", "--views", views, *extra]
    if main_file:
        args += ["--main-complete-call", str(main_file)]
    return J.main(args), tmp_path / out


def rows_of(d):
    return P.read_rows(d / "rows.jsonl")


def main_file(world, tmp_path, a0=(10.0, 20.0, 30.0, 12.0)):
    pos = [{"history_id": q["history_id"], "task": q["task"], "weight": q["weight"], "valid": True, "a0_s": a} for q, a in zip(world["positions"], a0)]
    p = tmp_path / "main_complete_call.json"
    p.write_text(json.dumps({"positions": pos, "primary": {"no_tap_mean_s_per_call": {"estimate": sum(a0) / 4}}}))
    return p


def test_run_rows_config_and_summary(world, tmp_path):
    stub = Stub(max_ctx=131072)
    mf = main_file(world, tmp_path)
    rc, d = run_judge(world, stub, tmp_path, main_file=mf)
    stub.close()
    assert rc == 0 and stub.calls == 4                       # warm-up + 3 prompts (the fourth position reuses its representative)
    rows = rows_of(d)
    assert len(rows) == 3 and all(r["concurrency"] == 1 and r["valid_score"] and r["http_error"] is None and r["attempt"] == 1 for r in rows)
    r0 = rows[0]
    for k in ("seconds", "prompt_tokens", "generated_tokens", "finish_reason", "valid_score", "user_sha256", "original_user_sha256", "start_utc",
              "output_cap", "evidence_shortened", "response_no_parsed_action", "response_finish_reason", "history_id", "task"):
        assert k in r0
    assert not any(isinstance(r.get("score"), (int, float)) for r in rows)
    cfg = json.loads((d / "run_config.json").read_text())
    c = cfg["config"]
    assert cfg["config_sha256"] == P.sha_obj(c) and c["kind"] == "judge_saved_calls" and c["max_attempts"] == 2 and c["concurrency"] == 1
    for k in ("stage2_list_sha256", "histories_sha256", "outputs_sha256", "trajectory_sha256", "plan", "budget_rule", "code_sha256", "policy_sha256", "positions"):
        assert k in c
    assert {"eq1_judge_saved_calls.py", "eq1_judge_saved_calls_prompts.py"} <= set(c["code_sha256"]) and c["budget_rule"]["fit_budget"] == 128000
    assert cfg["written_utc"] <= rows[0]["start_utc"]
    ses = P.read_rows(d / "sessions.jsonl")
    assert len(ses) == 1 and ses[0]["warmup"]["untimed"]
    s = json.loads((d / "summary.json").read_text())
    v = s["views"]["history_black_box"]
    assert v["n_prompts"] == 3 and v["n_valid"] == 3 and v["n_positions"] == 4 and v["n_valid_positions"] == 4 and v["complete"] and v["evidence_shortened_count"] == 0
    secs = {p["run_key"]: p["success_seconds"] for p in v["per_prompt"]}
    # position-weighted mean: position 4 (task 1 again) takes its representative's time
    rk = [q["run_key"] for q in world["positions"]]
    expect = sum(0.25 * secs[k] for k in rk)
    assert v["tap_s_per_call"]["estimate"] == pytest.approx(expect) and v["tap_s_per_call"]["n_tasks"] == 3
    comp = v["composed"]
    assert comp["no_tap_s_per_call"]["estimate"] == pytest.approx(18.0) and comp["with_tap_s_per_call"]["estimate"] == pytest.approx(18.0 + expect)
    assert comp["with_tap_s_per_call"]["ci95"][0] <= comp["with_tap_s_per_call"]["estimate"] <= comp["with_tap_s_per_call"]["ci95"][1]
    assert comp["main_complete_call_reported_no_tap"]["estimate"] == 18.0 and comp["complete"]


def test_resume_and_config_immutability(world, tmp_path):
    stub = Stub(max_ctx=131072)
    rc, d = run_judge(world, stub, tmp_path, extra=["--limit", "2"])
    calls, n = stub.calls, len(rows_of(d))
    rc2, _ = run_judge(world, stub, tmp_path, extra=["--limit", "2"])
    assert rc2 == 0 and stub.calls == calls and len(rows_of(d)) == n
    for extra in (["--limit", "3"], ["--limit", "2", "--max-attempts", "3"], ["--limit", "2", "--instruction", "terminal"]):
        with pytest.raises(SystemExit, match="refusing to resume"):
            run_judge(world, stub, tmp_path, extra=extra)
    with pytest.raises(SystemExit, match="refusing to resume"):
        run_judge(world, stub, tmp_path, name="gemma4-e4b-think", extra=["--limit", "2"])
    (d / "rows.jsonl").rename(d / "rows.bak")
    (d / "run_config.json").rename(d / "cfg.bak")
    (d / "rows.jsonl").write_text('{"item": 0}\n')
    with pytest.raises(SystemExit, match="rows exist without"):
        run_judge(world, stub, tmp_path, extra=["--limit", "2"])
    stub.close()


def test_changed_inputs_are_refused_on_resume(world, tmp_path):
    stub = Stub(max_ctx=131072)
    run_judge(world, stub, tmp_path, extra=["--limit", "1"])
    outs = [json.loads(x) for x in world["outputs"].read_text().splitlines()]
    outs[0]["text_stop_token_removed"] = "x\n</think>\n\nanother answer"
    world["outputs"].write_text("".join(json.dumps(o) + "\n" for o in outs))
    with pytest.raises(SystemExit, match="refusing to resume"):
        run_judge(world, stub, tmp_path, extra=["--limit", "1"])
    stub.close()


def test_retry_policy_keeps_all_attempts_and_both_time_quantities(world, tmp_path):
    stub = Stub(plan={2: 500, 4: "length", 6: "noscore"}, max_ctx=131072)       # call 1 warm-up; item0: calls 2-3 (500 then ok); item1: 4-5 (length then ok); item2: 6 (no score)
    rc, d = run_judge(world, stub, tmp_path, name="qwen3.8-27b-4k")
    stub.close()
    assert rc == 0
    rows = rows_of(d)
    by = {}
    for r in rows:
        by.setdefault(r["item"], []).append(r)
    assert [r["attempt"] for r in by[0]] == [1, 2] and by[0][0]["http_error"] and by[0][1]["retry_kind"] == "http_error" and by[0][1]["original_failure"].startswith("HTTPError")
    assert [r["attempt"] for r in by[1]] == [1, 2] and by[1][0]["finish_reason"] == "length" and by[1][1]["retry_kind"] == "raise_cap"
    cfg = m.get_spec("qwen3.8-27b-4k").config()
    assert by[1][1]["original_output_cap"] == 4096 and by[1][1]["output_cap"] == 131072 - by[1][1]["user_tokens_counted"] - cfg.template_overhead_tokens - 512
    assert len(by[2]) == 1 and by[2][0]["valid_score"] is False                    # a reply without a score is never forced or re-rolled
    s = json.loads((d / "summary.json").read_text())["views"]["history_black_box"]
    pp = {p["item"]: p for p in s["per_prompt"]}
    assert pp[0]["success_seconds"] == by[0][1]["seconds"] and pp[0]["effort_seconds"] == pytest.approx(by[0][0]["seconds"] + by[0][1]["seconds"])
    assert pp[1]["retry_kinds"] == ["raise_cap"] and pp[2]["final_valid"] is False and s["n_valid"] == 2 and not s["complete"]
    assert s["n_valid_positions"] == 3 and s["tap_s_per_call"]["estimate"] > 0 and s["tap_effort_s_per_call"]["estimate"] >= s["tap_s_per_call"]["estimate"]
    assert s["missing_positions"] == [world["positions"][2]["history_id"]]
    # resume sends nothing for finished items
    stub2 = Stub(max_ctx=131072)
    rc2, _ = run_judge(world, stub2, tmp_path, name="qwen3.8-27b-4k")
    stub2.close()
    assert rc2 == 0 and stub2.calls == 0


def test_long_prompt_is_fitted_and_recorded_black_box_is_not_sent(tmp_path):
    w = make_world(tmp_path, big_reasoning=500_000, call_index=1)       # 500k characters of reasoning in call 0 (~167k tokens by the character estimate)
    stub = Stub(max_ctx=131072)
    rc, d = run_judge(w, stub, tmp_path, views="history_black_box,history_reasoning")
    stub.close()
    assert rc == 0
    rows = rows_of(d)
    bb = [r for r in rows if r["view"] == "history_black_box"]
    cot = [r for r in rows if r["view"] == "history_reasoning"]
    assert len(bb) == 3 and not any(r.get("not_sent") for r in bb) and not any(r["evidence_shortened"] for r in bb)        # reasoning is not in the black-box view
    assert len(cot) == 3 and all(r["fitted"] and r["evidence_shortened"] and r["user_sha256"] != r["original_user_sha256"] for r in cot)
    assert all(r["original_user_tokens"] > 128000 >= r["user_tokens_counted"] for r in cot) and all(r["valid_score"] for r in cot)
    c = json.loads((d / "run_config.json").read_text())["config"]
    p = [x for x in c["plan"] if x["view"] == "history_reasoning"][0]
    assert p["shortening"]["blocks_shortened"] >= 1 and p["budget"]["fit_budget"] == 128000 and p["original_failure"].startswith("prompt of")
    texts = json.loads(gzip.decompress((d / "fitted_prompts.json.gz").read_bytes()))["items"]
    assert len(texts) == 3 and hashlib.sha256(texts[str(p["item"])].encode()).hexdigest() == p["user_sha256"]
    s = json.loads((d / "summary.json").read_text())["views"]["history_reasoning"]
    assert s["evidence_shortened_count"] == 3 and s["n_valid"] == 3


def test_unfittable_black_box_prompt_is_not_sent(tmp_path):
    big = "word " * 800_000                                               # a huge visible message in the NEW response: nothing in the black-box view can be shortened
    w = make_world(tmp_path, texts={"lcbhard_1": "r\n</think>\n\n" + big + TOOL})
    stub = Stub(max_ctx=131072)
    rc, d = run_judge(w, stub, tmp_path)
    stub.close()
    rows = rows_of(d)
    ns = [r for r in rows if r.get("not_sent")]
    assert rc == 0 and len(ns) == 1 and ns[0]["task"] == "lcbhard_1" and "cannot be fitted" in ns[0]["note"] and stub.calls == 1 + 2
    s = json.loads((d / "summary.json").read_text())["views"]["history_black_box"]
    assert s["not_sent_prompts"] == ["lcbhard_1@0.25"] and s["n_valid"] == 2 and s["n_valid_positions"] == 2 and not s["complete"]


def test_plan_only_writes_nothing(world, tmp_path, capsys):
    rc = J.main(["--monitor", "gemma4-e4b", "--out-dir", str(tmp_path / "po"), "--stage2-list", str(world["stage2"]), "--histories", str(world["hist"]),
                 "--outputs", str(world["outputs"]), "--trajectories", str(world["trajs"]), "--allow-char-counter", "--plan-only"])
    assert rc == 0 and not (tmp_path / "po").exists() and '"plan"' in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------- the table

def test_combine_rows_hardware_and_notes(world, tmp_path):
    mf = main_file(world, tmp_path)
    stub = Stub(max_ctx=131072)
    _, d1 = run_judge(world, stub, tmp_path, name="gemma4-e4b", out="mon/gemma4-e4b", main_file=mf)
    stub.close()
    s = json.loads((d1 / "summary.json").read_text())
    s["hardware_label"] = "GPU A"
    (d1 / "summary.json").write_text(json.dumps(s))
    out = tmp_path / "table"
    assert J.main(["--combine", str(tmp_path / "mon"), "--out", str(out), "--agent-hardware", "GPU A"]) == 0
    md = (out / "judge_table.md").read_text()
    lines = md.splitlines()
    assert lines[0].startswith("| Monitor | Monitor hardware (arrangement) | Config | With tap (s per call) | Tap = monitor time (s per call) |")
    assert "| gemma4-e4b | GPU A (agent and monitor on GPU A, run one after the other) |" in md
    assert "Model and tokenizer loading and any transfer of the saved output are not included" in md
    summ = json.loads((out / "judge_summary.json").read_text())
    assert len(summ["rows"]) == 1 and summ["rows"][0]["with_tap"]["estimate"] > summ["rows"][0]["tap"]["estimate"]
    tex = (out / "judge_table.tex").read_text()
    assert r"\toprule" in tex and "GPU A" in tex


def test_weighted_estimate_and_bootstrap_are_task_clustered():
    items = [{"task": "a", "weight": .25, "x": 1.0}, {"task": "a", "weight": .25, "x": 1.0}, {"task": "b", "weight": .25, "x": 3.0}, {"task": "c", "weight": .25, "x": 5.0}]
    w = J.weighted(items, "x")
    assert w["estimate"] == pytest.approx(2.5) and w["n_tasks"] == 3 and w["ci95"][0] <= 2.5 <= w["ci95"][1]
    assert J.weighted(items, "x") == w                                      # seed 0: reproducible


def test_summarize_only_adds_the_composed_row_later(world, tmp_path):
    stub = Stub(max_ctx=131072)
    rc, d = run_judge(world, stub, tmp_path)                 # no main_complete_call.json yet
    stub.close()
    assert json.loads((d / "summary.json").read_text())["views"]["history_black_box"]["composed"] is None
    calls = stub.calls
    rc2, _ = run_judge(world, stub, tmp_path, extra=["--summarize-only"], main_file=main_file(world, tmp_path))
    v = json.loads((d / "summary.json").read_text())["views"]["history_black_box"]
    assert rc2 == 0 and stub.calls == calls and v["composed"]["with_tap_s_per_call"]["estimate"] > v["composed"]["no_tap_s_per_call"]["estimate"]
    assert len(rows_of(d)) == 3                              # nothing was sent or written to the rows
