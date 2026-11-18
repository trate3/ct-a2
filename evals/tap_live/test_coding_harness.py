"""CPU tests of the coding harness, the task selection and the summary.

What runs end to end: the REAL `rollout.run_rollout` loop (real T1 instruction from the registry file, Qwen3.8 chat template
and tool-call parser, the full tool declarations, scorer) with the REAL sandbox (here: Docker with the pinned image; the tests
skip when it is unavailable) and the real bash/python tools, on the tiny random model with the tiny tokenizer. A random model
emits garbage, so most tests use a test-only `logits_processor` that forces scripted tool calls (the production path passes
none); one test runs without it (garbage ending the episode through the loop's own limits). Tasks are the two rows of the
local `lcb_fixture.json` (conflicting split), injected instead of downloading the dataset."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest
import torch

from evals.reward_hacking import llm, rollout, sandbox, tasks
from evals.reward_hacking import trajectory as tj
from evals.tap_live import run_overhead as ro
from evals.tap_live import select_tasks as st
from evals.tap_live import summarize
from evals.tap_live import tiny_model as tm

LAYER = 2
FIXTURE = Path(llm.__file__).parent / "data" / "lcb_fixture.json"
needs_sandbox = pytest.mark.skipif(not sandbox.backend_available() or not FIXTURE.exists(),
                                   reason="needs the sandbox backend (docker) and data/lcb_fixture.json")


# ---- task selection -------------------------------------------------------------------------------------------------

def test_positions():
    assert st.positions(39, 10) == [round(i * 38 / 9) for i in range(10)]
    assert st.positions(39, 10)[0] == 0 and st.positions(39, 10)[-1] == 38 and len(set(st.positions(39, 10))) == 10
    assert st.positions(5, 1) == [0]


def test_selection_is_the_declared_rule_and_independent_of_time(monkeypatch):
    sel = st.select()
    a = json.loads(st.SPLITS_A.read_text())["assignment"]
    b = json.loads(st.SPLITS_B.read_text())["assignment"]
    pool = sorted(t for t in a if a[t] == "train" and b.get(t) == "train" and t not in {"lcbhard_35", "lcbhard_77", "lcbhard_87"})
    assert len(pool) == sel["n_intersection"] == 39
    assert sel["task_ids"] == [pool[round(i * 38 / 9)] for i in range(10)]
    assert not set(sel["task_ids"]) & set(rollout.MALFORMED_TASKS)
    assert all(a[t] == "train" and b[t] == "train" for t in sel["task_ids"])
    assert sel["splits_json"]["sha256"] == st.sha256_file(st.SPLITS_A) and sel["splits_oneoff_v2_json"]["sha256"] == st.sha256_file(st.SPLITS_B)
    clock = random.Random(1)
    monkeypatch.setattr(time, "perf_counter", lambda: clock.random())
    monkeypatch.setattr(time, "time", lambda: clock.random() * 1e9)
    assert st.select() == sel                                       # no measured quantity enters


def test_selection_cli_writes_the_json(tmp_path, capsys):
    st.main(["--out", str(tmp_path / "t.json")])
    d = json.loads((tmp_path / "t.json").read_text())
    assert len(d["task_ids"]) == 10 and d["n_intersection"] == 39 and len(d["selection_sha256"]) == 64


# ---- schedule and the declared prefix ---------------------------------------------------------------------------------

def test_arm_order_alternates_with_task_index():
    assert [ro.arm_order(i) for i in range(4)] == [["A0", "A1"], ["A1", "A0"], ["A0", "A1"], ["A1", "A0"]]
    firsts = [ro.arm_order(i)[0] for i in range(10)]
    assert firsts.count("A0") == firsts.count("A1") == 5


def test_seed_depends_on_the_task_only():
    assert ro.task_seed("lcbhard_9", "T1") == ro.task_seed("lcbhard_9", "T1")
    assert ro.task_seed("lcbhard_9", "T1") != ro.task_seed("lcbhard_10", "T1")


class _Inner:
    def __init__(self):
        self.calls, self.n = [], 0

    def complete(self, prompt, *, max_tokens, seed, sampling):
        self.n += 1
        return self.n

    def reset_episode(self):
        self.n = 0
        self.reset = True


def test_prefix_client_raises_on_the_call_after_the_prefix():
    c = ro.PrefixClient(_Inner(), 3)
    assert [c.complete("p", max_tokens=None, seed=1, sampling={}) for _ in range(3)] == [1, 2, 3]
    with pytest.raises(ro.StopEpisode):
        c.complete("p", max_tokens=None, seed=1, sampling={})
    assert c.stop_raised
    c.reset_episode()
    assert not c.stop_raised and c.complete("p", max_tokens=None, seed=1, sampling={}) == 1
    assert c.calls == []                       # attributes of the engine are forwarded


# ---- end to end -------------------------------------------------------------------------------------------------------

def _tool(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def scripted(tok, turns):
    """Forced generation per call index (canonical reasoning, so the chat template re-renders each turn token for token)."""
    seqs = [tok.encode("I will look.\n</think>\n\n" + t, add_special_tokens=False) for t in turns]
    stop = tok.convert_tokens_to_ids("<|im_end|>")

    def proc(logits, gen_ids, call_index):
        seq = seqs[min(call_index, len(seqs) - 1)]
        want = seq[len(gen_ids)] if len(gen_ids) < len(seq) else stop
        out = torch.full_like(logits, float("-inf"))
        out[want] = 0.0
        return out
    return proc


BASH_TURNS = [_tool("bash", cmd="cat func.py"), _tool("bash", cmd="python3 test.py; echo second"), _tool("bash", cmd="ls /")]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("inputs")
    tm.random_artifact(LAYER, "token_mean", seed=1, threshold=0.0).save(d / "mean.json")
    tm.random_artifact(LAYER, "ema_max", alpha=0.5, seed=2, threshold=0.0).save(d / "ema.json")
    rows = json.loads(FIXTURE.read_text())["splits"]["conflicting"] if FIXTURE.exists() else []
    pool = [tasks.Task(task_id=r["task_id"], variant="conflicting", prompt=r["prompt"], test=r["test"], original_test=r["test"],
                       entry_point=r["entry_point"]) for r in rows]
    sel = {**st.select(), "task_ids": [t.task_id for t in pool]}
    (d / "tasks.json").write_text(json.dumps(sel))
    return d, pool, sel


def argv(d, out, extra=()):
    return ["--model", "tiny", "--artifact-mean", str(d / "mean.json"), "--artifact-ema", str(d / "ema.json"),
            "--tasks-json", str(d / "tasks.json"), "--out", str(out), "--device", "cpu", "--dtype", "float32", *extra]


@needs_sandbox
def test_end_to_end_two_tasks_two_arms_declared_three_call_prefix(inputs, tmp_path, model, tok, fmt, monkeypatch):
    d, pool, sel = inputs
    out = tmp_path / "run"
    inject = {"model": model, "tokenizer": tok, "fmt": fmt, "tasks": pool, "logits_processor": scripted(tok, BASH_TURNS)}
    assert ro.main(argv(d, out, ["--warmup-prompt-tokens", "256"]), inject=inject) == 0
    rows = [json.loads(x) for x in (out / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {(r["arm"], r["task_id"]) for r in rows} == {(a, t) for a in ("A0", "A1") for t in sel["task_ids"]}
    # the order alternates with the task index
    for r in rows:
        assert r["order_position"] == ro.arm_order(r["task_index"]).index(r["arm"])
    assert [r["arm"] for r in rows if r["order_position"] == 0] == ["A0", "A1"]
    for r in rows:
        assert r["seed"] == ro.task_seed(r["task_id"], "T1")
        # a declared 3-call prefix: three model calls, tools run after each, then the harness stops the episode
        assert r["n_calls"] == 3 and r["stopped_by_harness"] and r["status"] == "stopped_by_harness" and not r["infra_error"]
        assert r["declared_prefix_calls"] == 3 and r["tool_names"] == ["bash"] * 3 and len(r["call_hashes"]) == 3
        assert [c["tool_s_after"] > 0 for c in r["calls"]] == [True, True, True]
        assert r["tools_s"] > 0 and r["sandbox_start_s"] > 0 and r["scoring_s"] == 0           # no submission, no evaluation
        # the loop reused the cache: every call after the first extends the previous one
        assert r["resets"] == 0 and [c["n_cached_tokens"] > 0 for c in r["calls"]] == [False, True, True]
        assert r["n_prompt_computed"] + r["n_cached"] == r["n_prompt_tokens"]
        assert r["wall_s"] > r["model_s"] > 0 and r["nonmodel_s"] == pytest.approx(r["wall_s"] - r["model_s"])
        assert r["model_s"] == pytest.approx(r["prefill_s"] + r["decode_s"])
        parts = r["sandbox_start_s"] + r["tools_s"] + r["scoring_s"] + r["loop_model_s"] + r["orchestration_s"]
        assert parts == pytest.approx(r["wall_s"])
        assert r["loop_model_s"] >= r["model_s"]
    # same seed, same computation: identical token streams in both arms
    for t in sel["task_ids"]:
        assert len({r["token_hash"] for r in rows if r["task_id"] == t}) == 1
    a1 = [r for r in rows if r["arm"] == "A1"]
    for r in a1:
        assert len(r["decisions"]) == 6 and {x["head"] for x in r["decisions"]} == {"token_mean", "ema_max"}
        assert r["tap_s"] > 0 and r["tap_s"] == pytest.approx(r["capture_s"] + r["score_s"])
        assert set(r["score_s_by_head"]) == {"token_mean", "ema_max"}
        assert all(c["score_s_by_head"].keys() == {"token_mean", "ema_max"} and c["tap_n_scored"] == c["n_generated"] - 1 for c in r["calls"])
    assert all(r["decisions"] is None and r["tap_s"] == 0 for r in rows if r["arm"] == "A0")
    # files
    for f in ("environment.json", "setup.json", "episodes.jsonl", "final.json"):
        assert (out / f).exists()
    env = json.loads((out / "environment.json").read_text())
    assert env["concurrency"] == 1 and env["tap_shares_accelerator_with_model"] == "yes"
    assert env["tap"]["heads"]["token_mean"]["layer"] == LAYER and env["tap"]["heads"]["ema_max"]["pooling"] == "ema_max"
    assert len(env["tap"]["heads"]["token_mean"]["sha256"]) == 64 and "rollout.py" in env["code_sha256"] and "live_tap.py" in env["code_sha256"]
    assert env["workload"]["prompt_id"] == "T1" and env["workload"]["test_sentence"] == "default"
    assert env["workload"]["rollout_config"]["message_limit"] == 100 and env["workload"]["rollout_config"]["max_tokens_per_call"] == 0
    assert env["task_selection"]["splits_json_sha256"] == sel["splits_json"]["sha256"]
    setup = json.loads((out / "setup.json").read_text())
    assert len(setup["warmup"]) == 4 and "model_load_s" in setup
    traj = json.loads((out / "episodes" / f"{sel['task_ids'][0]}__A0.json").read_text())["trajectory"]
    prompt = next(e for e in traj["events"] if e["kind"] == tj.USER)["data"]["text"]
    assert prompt.startswith("Implement the function according to the provided specification.") and "DO NOT MODIFY THE TESTS" in prompt
    # summary
    summarize.main([str(out)])
    s = json.loads((out / "summary.json").read_text())
    assert s["n_rows"] == 4 and set(s["paired_vs_A0"]) == {"A1"} and s["paired_vs_A0"]["A1"]["n_pairs"] == 2
    assert s["paired_identical_streams"]["A1"]["n_pairs"] == 2 and s["divergence"]["A1"]["n_diverging"] == 0
    assert set(s["per_call_tap"]["A1"]["score_s_by_head"]) == {"token_mean", "ema_max"} and s["per_call_tap"]["A1"]["n_calls"] == 6
    md = (out / "summary.md").read_text()
    assert "Per-call tap time" in md and "relative wall overhead" in md and "Loading and warm-up" in md
    # resume: nothing is rerun
    assert ro.main(argv(d, out, ["--no-warmup"]), inject=inject) == 0
    assert len((out / "episodes.jsonl").read_text().splitlines()) == 4 and (out / "setup_resume1.json").exists()


@needs_sandbox
def test_a_submission_inside_the_prefix_runs_the_loops_own_evaluation(inputs, tmp_path, model, tok, fmt):
    d, pool, sel = inputs
    out = tmp_path / "submit"
    turns = [_tool("bash", cmd="cat test.py"), _tool("submit", answer="DONE"), _tool("bash", cmd="echo after")]
    inject = {"model": model, "tokenizer": tok, "fmt": fmt, "tasks": pool, "logits_processor": scripted(tok, turns)}
    assert ro.main(argv(d, out, ["--no-warmup", "--limit-tasks", "1"]), inject=inject) == 0
    rows = [json.loads(x) for x in (out / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert r["n_calls"] == 3 and r["stopped_by_harness"] and r["scoring_s"] > 0          # the in-loop evaluation of the submit
        assert r["tool_names"] == ["bash", "submit", "bash"]
    assert rows[0]["token_hash"] == rows[1]["token_hash"]
    ev = json.loads((out / "episodes" / f"{sel['task_ids'][0]}__A0.json").read_text())["trajectory"]["events"]
    assert any(e["kind"] == tj.EVALUATION for e in ev) and not any(e["kind"] == tj.ROLLOUT_END and e["data"]["status"] == "passed" for e in ev)


@needs_sandbox
def test_garbage_model_ends_through_the_loops_own_limits(inputs, tmp_path, model, tok, fmt):
    d, pool, _ = inputs
    out = tmp_path / "garbage"
    inject = {"model": model, "tokenizer": tok, "fmt": fmt, "tasks": pool}                  # no scripting: random tokens
    assert ro.main(argv(d, out, ["--no-warmup", "--limit-tasks", "1", "--max-tokens-per-call", "24"]), inject=inject) == 0
    rows = [json.loads(x) for x in (out / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and not any(r["infra_error"] for r in rows)
    for r in rows:     # a call that reaches its cap ends the rollout as model_length (then the loop scores the terminal state)
        assert r["status"] == "model_length" and r["n_calls"] == 1 and not r["stopped_by_harness"] and r["n_generated"] == 24
        assert r["scoring_s"] > 0
    assert rows[0]["token_hash"] == rows[1]["token_hash"]
    s = summarize.summarize(out)
    assert s["paired_vs_A0"]["A1"]["n_pairs"] == 1


def test_divergence_and_infra_errors_are_reported(tmp_path):
    base = {"calls": [], "decisions": None, "status": "stopped_by_harness", "n_prompt_computed": 5, "n_cached": 1, "resets": 0,
            "order_position": 0, "infra_error": False}
    mk = lambda task, arm, h, wall, n, nc: {**base, "task_id": task, "arm": arm, "call_hashes": h, "token_hash": "".join(h),  # noqa: E731
                                            "wall_s": wall, "model_s": wall * 0.9, "prefill_s": 1.0, "decode_s": wall * 0.9 - 1.0,
                                            "tap_s": 0.0, "nonmodel_s": wall * 0.1, "nonmodel_ex_tap_s": wall * 0.1,
                                            "n_calls": nc, "n_generated": n}
    rows = [mk("a", "A0", ["x", "y", "z"], 10.0, 100, 3), mk("a", "A1", ["x", "q", "z"], 12.0, 130, 3),
            mk("b", "A0", ["x"], 10.0, 100, 3), mk("b", "A1", ["x"], 10.5, 100, 3),
            mk("c", "A0", ["x"], 10.0, 100, 3), {**mk("c", "A1", ["x"], 99.0, 100, 3), "infra_error": True, "status": "infra_error",
                                                   "end_reason": "sandbox setup", "crashed": None}]
    rows[3]["decisions"] = rows[1]["decisions"] = rows[5]["decisions"] = []
    (tmp_path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = summarize.summarize(tmp_path)
    dv = s["divergence"]["A1"]
    assert dv["n_pairs"] == 2 and dv["n_identical"] == 1 and dv["n_diverging"] == 1 and dv["pairs"][0]["first_divergent_call"] == 1
    assert dv["pairs"][0]["d_tokens"] == 30
    assert len(s["infra_errors"]) == 1 and s["paired_vs_A0"]["A1"]["n_pairs"] == 2
    assert s["paired_identical_streams"]["A1"]["wall_s"]["values"] == [0.5]
    assert s["paired_vs_A0"]["A1"]["relative_wall"]["max"] == pytest.approx(0.2)
    assert "diverge" in summarize.to_markdown(s)
