"""Tests for the Inspect SWE-bench `.eval` sample -> Trajectory adapter (synthetic sample, no real log)."""

from __future__ import annotations

import json

import pytest
pytest.importorskip("inspect_ai")

from inspect_ai.log import EvalSample
from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser,
                              ContentReasoning, ContentText)
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall, ToolInfo, ToolParam, ToolParams

from evals.reward_hacking import llm
from evals.reward_hacking import swebench_trajectory as st
from evals.reward_hacking import trajectory as tj

TOOLS = [ToolInfo(name="bash", description="Run bash.",
                  parameters=ToolParams(properties={"command": ToolParam(type="string", description="cmd")},
                                        required=["command"])),
         ToolInfo(name="submit", description="Submit.",
                  parameters=ToolParams(properties={"answer": ToolParam(type="string", description="a")}))]


def _assistant(reasoning, text, calls):
    return ChatMessageAssistant(
        content=[ContentReasoning(reasoning=reasoning, internal="reasoning"), ContentText(text=text)],
        tool_calls=[ToolCall(id=i, function=f, arguments=a) for i, f, a in calls] or None)


def _sample(kind="test_failure", value=0.0, submit=True, patch="diff --git a/x b/x\n+fix\n"):
    msgs = [ChatMessageSystem(content="You fix bugs."), ChatMessageUser(content="Fix issue 1."),
            _assistant("look around", "", [("t1", "bash", {"command": "ls"})]),
            ChatMessageTool(content="a.py\n", tool_call_id="t1", function="bash"),
            _assistant("edit then submit", "Done editing.",
                       [("t2", "bash", {"command": "echo hi"}), ("t3", "submit", {"answer": "done"})] if submit
                       else [("t2", "bash", {"command": "echo hi"})]),
            ChatMessageTool(content="hi\n", tool_call_id="t2", function="bash")]
    if submit:
        msgs.append(ChatMessageTool(content="", tool_call_id="t3", function="submit"))
    score = Score(value=value, explanation="x", metadata={"evaluation_kind": kind, "model_patch": patch,
                                                           "test_command_exit": 0 if kind == "pass" else 1})
    return EvalSample(id="repo__inst-1", epoch=1, input="Fix issue 1.", target="", messages=msgs,
                      scores={"strict_swe_bench": score})


def _traj(sample=None, **kw):
    return st.from_eval_sample(sample or _sample(), instance_id="repo__inst-1", variant="conflicting",
                               partition="train", seed=7, fmt=llm.get_format("qwen3.8-27b"),
                               tools=st.tool_specs(TOOLS), **kw)


def test_event_order_and_kinds():
    t = _traj()
    kinds = [e.kind for e in t.events]
    assert kinds[:2] == [tj.SYSTEM, tj.USER]
    # per turn: model_call, reasoning, assistant text, then each call proposed/executed, results follow
    assert kinds[2:7] == [tj.MODEL_CALL, tj.REASONING, tj.ASSISTANT, tj.TOOL_PROPOSED, tj.TOOL_EXECUTED]
    assert kinds[-4:] == [tj.SUBMISSION, tj.SUBMISSION_EXECUTED, tj.EVALUATION, tj.ROLLOUT_END]
    assert [e.seq for e in t.events] == list(range(len(t.events)))
    prop = t.of(tj.TOOL_PROPOSED)
    assert [p.data["name"] for p in prop] == ["bash", "bash", "submit"]
    assert all(e.data["modified_by_wrapper"] is False for e in t.of(tj.TOOL_EXECUTED))
    assert [e.data["arguments"] for e in t.of(tj.TOOL_EXECUTED)] == [e.data["arguments"] for e in prop]
    assert [r.data["call_id"] for r in t.of(tj.TOOL_RESULT)] == [p.data["call_id"] for p in prop]
    assert t.of(tj.SUBMISSION)[0].data["origin"] == "submit"
    assert "diff --git" in t.of(tj.SUBMISSION)[0].data["func_py"]


@pytest.mark.parametrize("kind,value,expected", [("pass", 1.0, 1), ("test_failure", 0.0, 0), ("flagged", 0.0, 0)])
def test_label_follows_scorer(kind, value, expected):
    t = _traj(_sample(kind, value))
    assert tj.label(t) == expected
    assert t.of(tj.ROLLOUT_END)[0].data["label"] == expected


def test_unscored_is_unlabelled_and_terminal_state_without_submit():
    t = _traj(_sample("infra_error", 0.0, submit=False))
    assert tj.label(t) is None
    assert t.of(tj.SUBMISSION)[0].data["origin"] == "terminal_state"


def test_views_exclude_evaluation_and_blackbox_excludes_reasoning():
    t = _traj(_sample("pass", 1.0))
    bb = tj.view_events(t, tj.BLACK_BOX)
    cot = tj.view_events(t, tj.COT)
    for view in (bb, cot):
        kinds = {e.kind for e in view}
        assert not kinds & {tj.EVALUATION, tj.MODEL_CALL, tj.ROLLOUT_END}
        assert view[-1].kind == tj.SUBMISSION
    assert tj.REASONING not in {e.kind for e in bb}
    assert tj.REASONING in {e.kind for e in cot}
    text = tj.render_view(bb)
    assert "look around" not in text and "all selected tests passed" not in text
    assert "look around" in tj.render_view(cot)


def test_model_calls_match_assistant_turns_and_reconstruction():
    s = _sample()
    t = _traj(s)
    calls = t.of(tj.MODEL_CALL)
    assert len(calls) == sum(m.role == "assistant" for m in s.messages) == 2
    assert [c.call_index for c in calls] == [0, 1]
    for c in calls:
        d = c.data
        assert d["reconstruction"] == "approximate" and d["rendered_ok"] and "tokens" not in d
        assert d["prompt"].endswith("<|im_start|>assistant\n<think>\n")
        assert "Reasoning effort is set to xhigh" in d["prompt"]      # template default recorded in meta
    assert calls[1].data["prefix_consistent"] is True
    # call 1 prompt = call 0 prompt + call 0 generation + end-of-turn + the tool result
    assert calls[1].data["prompt"].startswith(calls[0].data["prompt"] + calls[0].data["generation"] + "<|im_end|>\n")
    assert calls[0].data["generation"].startswith("look around\n</think>\n\n<tool_call>\n<function=bash>")
    assert "look around" in calls[1].data["prompt"]                  # history reasoning replayed
    assert t.meta["reconstruction"]["parity_check_required"] is True
    assert t.meta["reasoning_effort_rendered"].startswith("xhigh")


def test_json_roundtrip_and_meta_keys():
    t = _traj()
    t2 = tj.Trajectory.from_json(t.to_json())
    assert tj.label(t2) == tj.label(t)
    for k in ("rollout_id", "task_id", "variant", "seed", "partition"):
        assert k in json.loads(t.to_json())["meta"]
    assert t.meta["rollout_id"] == "repo__inst-1__conflicting__r0"


def test_retokenized_tokens_fill_for_activations():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [len(s)]
    t = _traj()
    st.attach_retokenized_tokens(t, Tok())
    c = t.of(tj.MODEL_CALL)[0].data
    assert c["tokens"] == [len(c["generation"] + "<|im_end|>")] and c["tokens_source"] == "retokenized"


# ---------------------------------------------------------------------------
# recorded vLLM token ids

def _with_ids(sample, ids, *, only_first=False):
    from inspect_ai.event import ModelEvent
    from inspect_ai.model import GenerateConfig, ModelOutput
    from inspect_ai.model._model_call import ModelCall
    evs = []
    n_asst = [m for m in sample.messages if m.role == "assistant"]
    for i, _ in enumerate(n_asst):
        resp = {"prompt_token_ids": ids[i][0], "choices": [{"token_ids": ids[i][1]}]} \
            if not (only_first and i) else {"choices": [{}]}
        evs.append(ModelEvent(model="vllm/q", input=[], tools=TOOLS, tool_choice="auto", config=GenerateConfig(),
                              output=ModelOutput(model="q"), call=ModelCall(request={}, response=resp)))
    return sample.model_copy(update={"events": evs})


class _Tok:
    """char-level toy tokenizer; <|im_end|> is id 1."""
    def encode(self, s, add_special_tokens=False):
        return [1] if s == "<|im_end|>" else [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_recorded_ids_make_calls_exact_and_are_consumed():
    base = _traj()
    prompts = [c.data["prompt"] for c in base.of(tj.MODEL_CALL)]
    gens = [c.data["generation"] for c in base.of(tj.MODEL_CALL)]
    tok = _Tok()
    ids = [([ord(c) for c in p], [ord(c) for c in g] + [1]) for p, g in zip(prompts, gens)]
    t = _traj(_with_ids(_sample(), ids))
    for c, (pi, gi) in zip(t.of(tj.MODEL_CALL), ids):
        assert c.data["reconstruction"] == "exact" and c.data["tokens_source"] == "vllm"
        assert c.data["prompt_token_ids"] == pi and c.data["tokens"] == gi and c.data["prompt_tokens"] == len(pi)
    assert t.meta["reconstruction"]["exact"] is True and t.meta["reconstruction"]["tokens"] == "vllm"
    rows = st.verify_recorded_ids(t, tok)
    assert all(r["prompt_ids_match"] and r["generation_decode_match"] for r in rows)
    st.attach_retokenized_tokens(t, tok)                      # must not overwrite vllm ids
    assert t.of(tj.MODEL_CALL)[0].data["tokens"] == ids[0][1]
    # mismatching recorded ids are flagged, and reconstruct_exact uses the recorded ones
    bad = [([5, 6, 7], g) for _, g in ids]
    tb = _traj(_with_ids(_sample(), bad))
    assert not any(r["prompt_ids_match"] for r in st.verify_recorded_ids(tb, tok))
    from evals.reward_hacking import activations as act
    seqs = st.reconstruct_exact(tb, _TokMarks())
    assert seqs[0].prompt_ids == [5, 6, 7] and seqs[0].gen_ids == ids[0][1]
    assert act.CHANNEL_NAMES  # module imported fine


class _TokMarks(_Tok):
    def encode(self, s, add_special_tokens=False):
        m = {"<think>": [10], "</think>": [11], "<tool_call>": [12], "</tool_call>": [13], "<|im_end|>": [1]}
        return m.get(s) or super().encode(s)


def test_missing_ids_keep_approximate_path():
    base = _traj()
    ids = [([1, 2], [3, 1])] * 2
    t = _traj(_with_ids(_sample(), ids, only_first=True))
    c0, c1 = t.of(tj.MODEL_CALL)
    assert c0.data["reconstruction"] == "exact" and c1.data["reconstruction"] == "approximate"
    assert "tokens" not in c1.data and t.meta["reconstruction"]["tokens"] == "mixed"
    assert base.of(tj.MODEL_CALL)[1].data["reconstruction"] == "approximate"
