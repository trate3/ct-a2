"""Raising the per-call cap by continuing capped calls (extend.py): plan, replay, continuation, end to end."""

from __future__ import annotations

import pytest

from evals.reward_hacking import extend, llm, rollout
from evals.reward_hacking import trajectory as tj
from evals.reward_hacking.conftest import requires_sandbox
from evals.reward_hacking.test_rollout import _call, _task

CAP = 8


def _traj(status: str, stops: list[str], ntoks: list[int]) -> tj.Trajectory:
    t = tj.Trajectory(meta={"rollout_id": "r"})
    for i, (s, n) in enumerate(zip(stops, ntoks)):
        t.add(tj.MODEL_CALL, {"prompt": f"p{i}", "generation": "g", "tokens": list(range(n)), "stop_type": s,
                              "prompt_tokens": 3, "cached_tokens": 0}, call_index=i)
    t.add(tj.ROLLOUT_END, {"status": status, "reason": "", "label": 0})
    return t


OLD = {"max_tokens_per_call": CAP, "max_context_tokens": 100}
BIGGER_CAP = {"max_tokens_per_call": 4 * CAP, "max_context_tokens": 100}
NO_CAP_MORE_CTX = {"max_tokens_per_call": 0, "max_context_tokens": 200}


def test_plan():
    assert extend.plan(None, OLD, BIGGER_CAP) == extend.FRESH
    assert extend.plan(_traj("infra_error", ["eos"], [3]), OLD, BIGGER_CAP) == extend.FRESH
    assert extend.plan(_traj("attempts_exhausted", ["eos", "eos"], [3, 7]), OLD, BIGGER_CAP) == extend.REUSE
    ctx = _traj("context_exhausted", ["eos"], [5])
    assert extend.plan(ctx, OLD, BIGGER_CAP) == extend.REUSE           # context rule unchanged
    assert extend.plan(ctx, OLD, NO_CAP_MORE_CTX) == extend.EXTEND     # context rule grew: resume
    capped = _traj("model_length", ["eos", "limit"], [3, CAP])
    assert extend.plan(capped, OLD, BIGGER_CAP) == extend.EXTEND
    assert extend.plan(capped, OLD, NO_CAP_MORE_CTX) == extend.EXTEND
    assert extend.plan(capped, OLD, {**OLD, "max_context_tokens": 200}) == extend.REUSE   # cap unchanged
    with pytest.raises(ValueError):          # a capped call but the rollout did not stop on it
        extend.plan(_traj("passed", ["limit", "eos"], [CAP, 3]), OLD, BIGGER_CAP)
    with pytest.raises(ValueError):          # model_length whose last call is not at the old cap
        extend.plan(_traj("model_length", ["eos", "limit"], [3, 5]), OLD, BIGGER_CAP)


class Inner:
    """Fake server: tokenize = one id per char; records what it was asked to continue."""

    def __init__(self, texts):
        self.texts, self.sent = list(texts), []

    def tokenize(self, text):
        return [ord(c) for c in text]

    def complete(self, prompt, *, max_tokens, seed, sampling):
        self.sent.append((prompt, max_tokens, seed))
        text = self.texts.pop(0)
        return llm.Completion(text=text, tokens=[1000 + i for i in range(len(text))], stop_type="eos",
                              prompt_tokens=len(prompt), cached_tokens=0, seconds=0.0, timings={})


def _rec(prompt, gen, stop):
    return {"prompt": prompt, "generation": gen, "tokens": [500 + i for i in range(len(gen))],
            "stop_type": stop, "prompt_tokens": len(prompt), "cached_tokens": 0, "timings": {}}


def test_client_replays_then_continues_then_samples():
    recorded = [_rec("AB", "xy", "eos"), _rec("ABxyC", "partial", "limit")]
    inner = Inner([" rest", "next"])
    c = extend.ExtendingClient(inner, recorded)
    a = c.complete("AB", max_tokens=20, seed=1, sampling={})
    assert (a.text, a.tokens, a.stop_type) == ("xy", [500, 501], "eos") and not inner.sent
    b = c.complete("ABxyC", max_tokens=20, seed=2, sampling={})
    ids, budget, seed = inner.sent[0]
    assert ids == [ord(ch) for ch in "ABxyC"] + [500 + i for i in range(7)]   # prompt ids + recorded gen ids
    assert budget == 20 - 7 and seed == 2
    assert b.text == "partial rest" and b.tokens[:7] == recorded[1]["tokens"] and len(b.tokens) == 12
    assert b.stop_type == "eos" and b.prompt_tokens == 5
    assert c.info == {"extended_call": 1, "recorded_tokens": 7, "continued_tokens": 5,
                      "replayed_tool_outputs_differed": 0, "boundary_replacement_char": False}
    d = c.complete("anything", max_tokens=20, seed=3, sampling={})
    assert d.text == "next" and inner.sent[-1][0] == "anything"


def test_client_refuses_a_different_prompt():
    c = extend.ExtendingClient(Inner([]), [_rec("AB", "xy", "eos"), _rec("ABxyC", "p", "limit")])
    with pytest.raises(extend.ReplayMismatch):
        c.complete("AX", max_tokens=20, seed=1, sampling={})


class CappedScripted:
    """Scripted model whose generations carry their own stop type (to fake a capped call)."""

    def __init__(self, items):
        self.items, self.prompts = list(items), []

    def tokenize(self, text):
        return list(range(len(text)))

    def complete(self, prompt, *, max_tokens, seed, sampling):
        self.prompts.append(prompt)
        text, stop = self.items.pop(0)
        ntok = max_tokens if stop == "limit" else len(text) // 4 + 1
        return llm.Completion(text=text, tokens=list(range(ntok)), stop_type=stop,
                              prompt_tokens=len(prompt), cached_tokens=0, seconds=0.0)


@requires_sandbox
def test_capped_rollout_is_continued_not_resampled():
    task = _task()
    first = "Look.\n</think>\n\n" + _call("bash", cmd="date +%s%N; cat test.py; echo x > notes.txt")
    cut = "Thinking hard about the problem and"
    orig = rollout.run_rollout(task, rollout_id="t", seed=5, client=CappedScripted([(first, "eos"), (cut, "limit")]),
                               config=rollout.RolloutConfig(max_tokens_per_call=CAP))
    assert orig.of(tj.ROLLOUT_END)[-1].data["status"] == "model_length"
    assert extend.plan(orig, OLD, BIGGER_CAP) == extend.EXTEND

    inner = CappedScripted([(" so I implement it.\n</think>\n\n" + _call("submit", answer="DONE"), "eos"),
                            ("Again.\n</think>\n\n" + _call("submit", answer="DONE"), "eos"),
                            ("Again.\n</think>\n\n" + _call("submit", answer="DONE"), "eos")])
    client = extend.ExtendingClient(inner, [e.data for e in orig.of(tj.MODEL_CALL)], orig)
    new = rollout.run_rollout(task, rollout_id="t", seed=5, client=client,
                              config=rollout.RolloutConfig(max_tokens_per_call=4 * CAP))
    calls = new.of(tj.MODEL_CALL)
    o = orig.of(tj.MODEL_CALL)
    assert calls[0].data["prompt"] == o[0].data["prompt"] and calls[0].data["generation"] == first
    assert calls[1].data["prompt"] == o[1].data["prompt"]          # tool outputs replayed identically
    assert calls[1].data["generation"].startswith(cut + " so I implement it.")
    assert calls[1].data["tokens"][:CAP] == o[1].data["tokens"]
    assert len(inner.prompts[0]) == o[1].data["prompt_tokens"] + CAP   # ids + recorded tokens
    assert client.info["extended_call"] == 1
    assert client.info["replayed_tool_outputs_differed"] == 1     # the clock; the model saw the recorded one
    assert new.of(tj.TOOL_RESULT)[0].data["output"] == orig.of(tj.TOOL_RESULT)[0].data["output"]
    assert new.of(tj.ROLLOUT_END)[-1].data["status"] != "model_length"
    assert all(c.data["prefix_consistent"] is not False for c in calls[1:])


def test_workspace_check():
    t = tj.Trajectory(meta={})
    t.add(tj.WORKSPACE, {"after_call_id": None, "sha256": "s0"})
    t.add(tj.TOOL_RESULT, {"call_id": "call_1", "output": "o"})
    t.add(tj.TOOL_RESULT, {"call_id": "call_2", "output": "o"})
    t.add(tj.WORKSPACE, {"after_call_id": "call_2", "sha256": "s2"})
    c = extend.ExtendingClient(Inner([]), [], t)
    c.replay_check_workspace("call_1", "s0")
    c.replay_check_workspace("call_2", "s2")
    c.replay_check_workspace("call_9", "anything")      # not replayed: unchecked
    with pytest.raises(extend.ReplayMismatch):
        c.replay_check_workspace("call_2", "s0")


def test_no_cap_continuation_passes_none():
    inner = Inner([" rest"])
    c = extend.ExtendingClient(inner, [_rec("AB", "partial", "limit")])
    c.complete("AB", max_tokens=None, seed=1, sampling={})
    assert inner.sent[0][1] is None


@requires_sandbox
def test_context_exhausted_rollout_resumes_with_live_tools():
    task = _task()
    big = "Plan.\n</think>\n\n" + _call("bash", cmd="echo resumed-live > marker.txt; cat marker.txt")
    orig = rollout.run_rollout(task, rollout_id="t", seed=7,
                               client=CappedScripted([("Look.\n</think>\n\n" + _call("bash", cmd="ls"), "eos"),
                                                      (big, "eos")]),
                               config=rollout.RolloutConfig(max_context_tokens=10))
    # CappedScripted reports prompt_tokens = len(prompt) chars, so the first call already exceeds 10
    assert orig.of(tj.ROLLOUT_END)[-1].data["status"] == "context_exhausted"
    old = {"max_tokens_per_call": 32768, "max_context_tokens": 10}
    new = {"max_tokens_per_call": 0, "max_context_tokens": 10**9}
    assert extend.plan(orig, old, new) == extend.EXTEND
    inner = CappedScripted([(big, "eos"), ("Done.\n</think>\n\n" + _call("submit", answer="DONE"), "eos")] +
                           [("Again.\n</think>\n\n" + _call("submit", answer="DONE"), "eos")] * 2)
    client = extend.ExtendingClient(inner, [e.data for e in orig.of(tj.MODEL_CALL)], orig)
    res = rollout.run_rollout(task, rollout_id="t", seed=7, client=client,
                              config=rollout.RolloutConfig(max_tokens_per_call=0, max_context_tokens=10**9))
    calls, o = res.of(tj.MODEL_CALL), orig.of(tj.MODEL_CALL)
    assert [c.data["prompt"] for c in calls[:len(o)]] == [c.data["prompt"] for c in o]
    assert client.info["resumed_after_call"] == len(o) - 1
    # the recorded last call's tool, never run originally, now ran live
    first_live = res.of(tj.TOOL_RESULT)[len(orig.of(tj.TOOL_RESULT))]
    assert "resumed-live" not in "".join(e.data["output"] for e in orig.of(tj.TOOL_RESULT))
    assert res.of(tj.ROLLOUT_END)[-1].data["status"] != "context_exhausted"
    assert first_live.data["output"] is not None
