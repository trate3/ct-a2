"""Muse Glimmer 30B in the SWE-bench Inspect-log converter.

No real Muse run log exists yet, so the inputs are synthetic: Inspect messages built from the real model samples in
`data/muse_raw_fixtures.json` (parsed with `MuseFormat.parse_generation`) and from the shape of the Qwen converter
tests. Every piece of the Inspect log format assumed here is listed in `ASSUMPTIONS` below and is meant to be checked
against a real log. The Qwen path is covered by test_swebench_trajectory.py and must not change."""

from __future__ import annotations

import json
import pathlib
import re

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalDataset, EvalConfig, EvalLog, EvalSample, EvalSpec, read_eval_log, write_eval_log
from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser,
                              ContentReasoning, ContentText, GenerateConfig, ModelOutput)
from inspect_ai.model._model_call import ModelCall
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall, ToolCallError, ToolInfo, ToolParam, ToolParams

from evals.reward_hacking import first_violation as fv
from evals.reward_hacking import llm
from evals.reward_hacking import muse_format
from evals.reward_hacking import swebench_trajectory as st
from evals.reward_hacking import trajectory as tj

ASSUMPTIONS = """
1. The Inspect model name contains "muse" (vllm/muse-glimmer-30b); the reasoning strength is in the generate config's
   extra_body.chat_template_kwargs.reasoning_strength.
2. An assistant message with reasoning has list content [ContentReasoning(reasoning=<parsed to=self text>),
   ContentText(text=<parsed to=user text>)]; one without reasoning has string content. The reasoning is the model's
   `to=self` body as the vLLM muse_glimmer reasoning parser returned it (verbatim or stripped: tolerated, see
   test_stripped_reasoning_is_repaired_from_recorded_ids).
3. Tool calls are ToolCall(id, function=<tool name>, arguments=<dict>) with the ATEM parameters already coerced by
   vLLM's muse_glimmer tool parser; parallel calls are several ToolCalls on ONE assistant message, in generation order;
   a call vLLM or Inspect could not parse keeps its function name, arguments {} and parse_error set.
4. ModelEvent.call.response is vLLM's raw chat response: top-level prompt_token_ids, choices[0].token_ids (last id is
   <|eot|> 200008) and choices[0].message.tool_calls[*].{id, function.arguments} (JSON string, the malformed text for a
   call Inspect could not parse).
5. Inspect replays a list-content assistant message as content "\\n"+text with reasoning in reasoning_content
   (inspect_ai 0.3.277 openai_assistant_content); the Muse template does not trim, so the next prompt shows "\\n"+text.
6. Tool result messages carry `function` (the tool name) and `tool_call_id`; a failed tool call has `error`.
7. A turn with no text, no tool call and stop reason max_tokens/model_length was cut off inside the reasoning.
"""

DATA = pathlib.Path(__file__).parent / "data"
FIX = json.loads((DATA / "muse_raw_fixtures.json").read_text())
EOT = muse_format.EOT
EOT_ID = muse_format.EOT_TOKEN_ID
FMT = llm.get_format("muse-glimmer-30b", reasoning_effort="medium")

# ---- fixture-grounded tools (as test_muse_format.py) -------------------------------------------------------------
FIX_SYSTEM = "You are a careful assistant. Use tools when helpful."
FIX_USER = {"single_call": "What is the weather in Testville in celsius?",
            "parallel_calls": "Get the weather in Paris and in Rome, both in one turn.",
            "code_call_multiline": "Use run_python to print the first 5 squares, one per line, using a for loop.",
            "plain_answer": "In one sentence, what is a prime number? Do not use tools."}
FIX_TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Return the weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "run_python", "description": "Run python code and return stdout.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]

# ---- SWE-bench-like tools ----------------------------------------------------------------------------------------
SWE_TOOLS = [ToolInfo(name="bash", description="Run bash.",
                      parameters=ToolParams(properties={"command": ToolParam(type="string", description="cmd")},
                                            required=["command"])),
             ToolInfo(name="submit", description="Submit.",
                      parameters=ToolParams(properties={"answer": ToolParam(type="string", description="a")}))]
SWE_SPECS = st.tool_specs(SWE_TOOLS)


def asst(reasoning, text, calls=(), *, plain=False):
    """Assistant message as Inspect builds it from a vLLM reply: list content when there is reasoning."""
    content = text if plain else [ContentReasoning(reasoning=reasoning, internal="reasoning_content"),
                                  ContentText(text=text)]
    return ChatMessageAssistant(content=content,
                                tool_calls=[ToolCall(id=i, function=f, arguments=a,
                                                     parse_error="bad arguments" if bad else None)
                                            for i, f, a, bad in
                                            [(c + (False,))[:4] if len(c) == 3 else c for c in calls]] or None)


def tool(i, fn, out="ok\n", error=None):
    return ChatMessageTool(content=out, tool_call_id=i, function=fn,
                           error=None if error is None else ToolCallError(type="parsing", message=error))


def sample(msgs, *, kind="test_failure", value=0.0, events=None):
    score = Score(value=value, explanation="x", metadata={"evaluation_kind": kind, "model_patch": "diff --git a/x b/x\n+1\n",
                                                           "test_command_exit": 0 if kind == "pass" else 1})
    return EvalSample(id="repo__inst-1", epoch=1, input="Fix issue 1.", target="", messages=msgs,
                      scores={"strict_swe_bench": score}, events=events or [])


def convert(s, *, tools=None, fmt=FMT):
    return st.from_eval_sample(s, instance_id="repo__inst-1", variant="conflicting", partition="train", seed=7,
                               fmt=fmt, tools=tools if tools is not None else SWE_SPECS)


def events_for(msgs, ids=None, responses=None, stop=None):
    """One ModelEvent per assistant message; `ids[i]` = (prompt_ids, generated_ids) put in the raw vLLM response."""
    evs = []
    n = 0
    for m in msgs:
        if m.role != "assistant":
            continue
        resp = {"choices": [{}]}
        if ids is not None and ids[n] is not None:
            resp = {"prompt_token_ids": ids[n][0], "choices": [{"token_ids": ids[n][1]}]}
        if responses is not None and responses.get(n):
            resp["choices"][0]["message"] = {"tool_calls": responses[n]}
        out = ModelOutput.from_content("vllm/muse-glimmer-30b", "") if not stop or not stop.get(n) else \
            ModelOutput.from_content("vllm/muse-glimmer-30b", "", stop_reason=stop[n])
        evs.append(ModelEvent(model="vllm/muse-glimmer-30b", input=[], tools=SWE_TOOLS, tool_choice="auto",
                              config=GenerateConfig(), output=out, call=ModelCall(request={}, response=resp)))
        n += 1
    return evs


class Tok:
    """Toy tokenizer: characters are ids, the Muse specials one id each (so ids extend like real ones)."""
    SPECIAL = {"<|begin_of_text|>": 200000, "<|start|>": 200006, "<|eom|>": 200007, "<|eot|>": EOT_ID,
               "<|message|>": 200005}
    REV = {v: k for k, v in SPECIAL.items()}
    _SPLIT = re.compile("(" + "|".join(re.escape(k) for k in SPECIAL) + ")")

    def encode(self, s, add_special_tokens=False):
        out = []
        for part in self._SPLIT.split(s):
            if part in self.SPECIAL:
                out.append(self.SPECIAL[part])
            else:
                out.extend(ord(c) for c in part)
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.REV.get(i) or chr(i) for i in ids)


TOK = Tok()


def ids_from(traj, tok=TOK):
    return [(tok.encode(c.data["prompt"]), tok.encode(c.data["generation"] + EOT)) for c in traj.of(tj.MODEL_CALL)]


def swe_run(*, submit=True):
    """system, user, two tool-call turns, then a final answer (a submit call, or plain text at the end)."""
    msgs = [ChatMessageSystem(content="You fix bugs."), ChatMessageUser(content="Fix issue 1."),
            asst("look around first", "", [("t1", "bash", {"command": "ls"})]), tool("t1", "bash", "a.py\n"),
            asst("edit the file", "", [("t2", "bash", {"command": "echo 'x y' > a.py"})]), tool("t2", "bash", "")]
    if submit:
        msgs += [asst("all done, submit", "", [("t3", "submit", {"answer": "fixed"})]), tool("t3", "submit", "")]
    else:
        msgs += [asst("nothing else to do", "I fixed the bug in a.py.")]
    return msgs


# ---------------------------------------------------------------------------------------------------------------
# real samples: prompt and generation come back byte for byte

CASES = [(case, i, s) for case, c in FIX["cases"].items() for i, s in enumerate(c["samples"])]


@pytest.mark.parametrize("case,i,smp", CASES, ids=[f"{c}-{i}" for c, i, _ in CASES])
def test_fixture_samples_reconstruct_prompt_and_generation_exactly(case, i, smp):
    turn = muse_format.MuseFormat("medium").parse_generation(smp["text"], FIX_TOOLS)
    msgs = [ChatMessageSystem(content=FIX_SYSTEM), ChatMessageUser(content=FIX_USER[case]),
            asst(turn.reasoning_raw, turn.content_raw,
                 [(f"c{n}", c.name, dict(c.arguments)) for n, c in enumerate(turn.tool_calls)])]
    t = convert(sample(msgs), tools=FIX_TOOLS)
    (mc,) = t.of(tj.MODEL_CALL)
    assert mc.data["prompt"] == FIX["cases"][case]["prompt"]
    assert mc.data["generation"] == smp["text"] and mc.data["rendered_ok"]
    assert mc.data["reconstruction"] == "approximate"
    assert [p.data["name"] for p in t.of(tj.TOOL_PROPOSED)] == [c.name for c in turn.tool_calls]
    assert [p.data["arguments"] for p in t.of(tj.TOOL_PROPOSED)] == [dict(c.arguments) for c in turn.tool_calls]
    for p in t.of(tj.TOOL_PROPOSED):          # raw = the ATEM block, in the generation
        assert p.data["raw"].startswith("<atem:function_calls>") and p.data["raw"] in smp["text"]
    assert t.meta["target_model"] == "muse-glimmer-30b"


def test_fixture_history_continuation_with_tool_result():
    """The recorded single-call fixture + a tool result: the second prompt is the fixture's history prompt."""
    turn = muse_format.MuseFormat("medium").parse_generation(
        FIX["cases"]["single_call"]["samples"][0]["text"], FIX_TOOLS)
    c = turn.tool_calls[0]
    msgs = [ChatMessageSystem(content=FIX_SYSTEM), ChatMessageUser(content=FIX_USER["single_call"]),
            asst(turn.reasoning_raw, turn.content_raw, [("c0", c.name, dict(c.arguments))]),
            tool("c0", "get_weather", "Sunny, 19 C."),
            asst(*(lambda s: (muse_format.MuseFormat("medium").parse_generation(s["text"], FIX_TOOLS).reasoning_raw,
                              muse_format.MuseFormat("medium").parse_generation(s["text"], FIX_TOOLS).content_raw))(
                FIX["history_render"]["samples"][0]))]
    t = convert(sample(msgs), tools=FIX_TOOLS)
    c0, c1 = t.of(tj.MODEL_CALL)
    # the fixture history uses reasoning "I should call get_weather." so only the shape is compared here
    assert c1.data["prompt"].startswith(c0.data["prompt"] + c0.data["generation"] + EOT)
    assert c1.data["prompt"].endswith("<tool_output name=\"get_weather\">\nSunny, 19 C.\n</tool_output><|eot|><|start|>assistant")
    assert c1.data["prefix_consistent"] is True


# ---------------------------------------------------------------------------------------------------------------
# SWE-like logs

def test_two_tool_turns_and_submit_event_layout_and_boundaries():
    msgs = swe_run(submit=True)
    t = convert(sample(msgs, kind="pass", value=1.0))
    kinds = [e.kind for e in t.events]
    assert kinds[:2] == [tj.SYSTEM, tj.USER]
    assert kinds[2:7] == [tj.MODEL_CALL, tj.REASONING, tj.ASSISTANT, tj.TOOL_PROPOSED, tj.TOOL_EXECUTED]
    assert kinds[-4:] == [tj.SUBMISSION, tj.SUBMISSION_EXECUTED, tj.EVALUATION, tj.ROLLOUT_END]
    calls = t.of(tj.MODEL_CALL)
    assert [c.call_index for c in calls] == [0, 1, 2]
    for c in calls:
        d = c.data
        assert d["prompt"].startswith("<|begin_of_text|><|start|>system<|message|>You fix bugs.")
        assert "Reasoning strength: medium." in d["prompt"] and d["prompt"].endswith("<|start|>assistant")
        assert d["generation"].startswith(" to=self<|message|>") and d["rendered_ok"]
        assert "<think>" not in d["generation"] + d["prompt"] and "<|im_end|>" not in d["generation"] + d["prompt"]
        assert d["generation"].count("<atem:function_calls>") == 1
    assert calls[0].data["generation"] == (
        ' to=self<|message|>look around first<|eom|><|start|>assistant to=bash<|message|><atem:function_calls>\n'
        '<atem:invoke name="bash">\n<atem:parameter name="command">ls</atem:parameter>\n</atem:invoke>\n'
        '</atem:function_calls>')
    assert [c.data["prefix_consistent"] for c in calls] == [None, True, True]
    assert calls[1].data["prompt"].startswith(calls[0].data["prompt"] + calls[0].data["generation"] + EOT)
    assert "look around first" in calls[1].data["prompt"]                      # reasoning replayed in history
    assert '<|start|>tool bash<|message|><tool_output name="bash">\na.py\n\n</tool_output>' in calls[1].data["prompt"]
    # tools, results, submission
    prop = t.of(tj.TOOL_PROPOSED)
    assert [p.data["name"] for p in prop] == ["bash", "bash", "submit"]
    assert [r.data["call_id"] for r in t.of(tj.TOOL_RESULT)] == [p.data["call_id"] for p in prop]
    sub = t.of(tj.SUBMISSION)[0]
    assert sub.data["origin"] == "submit" and sub.data["answer"] == "fixed" and sub.call_index == 2
    assert tj.is_swe_submission(sub) and tj.label(t) == 1
    # monitors: terminal boundary, per-call decision points, black-box vs CoT
    assert tj.labelled_submission_max_call(t) == 3
    bb, cot = tj.view_events(t, tj.BLACK_BOX), tj.view_events(t, tj.COT)
    assert bb[-1].kind == cot[-1].kind == tj.SUBMISSION
    assert not {e.kind for e in bb + cot} & {tj.EVALUATION, tj.MODEL_CALL, tj.ROLLOUT_END}
    assert tj.REASONING not in {e.kind for e in bb} and tj.REASONING in {e.kind for e in cot}
    assert "look around first" not in tj.render_view(bb) and "look around first" in tj.render_view(cot)
    recs = fv.decision_point_records(t)
    assert [r["call_index"] for r in recs] == [0, 1, 2]
    assert all(t.events[r["seq"]].kind == tj.TOOL_PROPOSED for r in recs)
    # metadata: target model, format, effort, audit
    assert t.meta["target_model"] == "muse-glimmer-30b" and t.meta["template_sha256"] == FMT.template_digest()
    assert t.meta["reasoning_effort_rendered"] == "medium" and t.meta["end_token"] == EOT
    assert t.meta["chat_format"]["name"] == "muse-glimmer-30b" and t.meta["chat_format"]["eot_token_id"] == EOT_ID
    a = t.meta["conversion_audit"]
    assert (a["calls"], a["tool_calls"], a["parse_error_calls"], a["non_append_only_calls"]) == (3, 3, 0, 0)
    assert tj.Trajectory.from_json(t.to_json()).meta["conversion_audit"] == a


def test_final_answer_without_submit_is_terminal_state():
    t = convert(sample(swe_run(submit=False)))
    last = t.of(tj.MODEL_CALL)[-1]
    assert last.data["generation"] == (' to=self<|message|>nothing else to do<|eom|>'
                                       '<|start|>assistant to=user<|message|>I fixed the bug in a.py.')
    assert [p.data["name"] for p in t.of(tj.TOOL_PROPOSED)] == ["bash", "bash"]
    assert t.of(tj.ASSISTANT)[-1].data["text"] == "I fixed the bug in a.py."
    sub = t.of(tj.SUBMISSION)[0]
    assert sub.data["origin"] == "terminal_state" and sub.call_index == 2
    assert tj.labelled_submission_max_call(t) == 3
    assert tj.label(t) == 0


def test_reasoning_without_tool_call_then_continue():
    msgs = [ChatMessageSystem(content="You fix bugs."), ChatMessageUser(content="Fix issue 1."),
            asst("let me think about it", "I am thinking."),
            ChatMessageUser(content="Please continue."),
            asst("now act", "", [("t1", "bash", {"command": "ls"})]), tool("t1", "bash", "a.py\n")]
    t = convert(sample(msgs))
    c0, c1 = t.of(tj.MODEL_CALL)
    assert c0.data["generation"] == (" to=self<|message|>let me think about it<|eom|>"
                                     "<|start|>assistant to=user<|message|>I am thinking.")
    assert [e.kind for e in t.events[2:6]] == [tj.MODEL_CALL, tj.REASONING, tj.ASSISTANT, tj.USER]
    assert t.of(tj.USER)[1].data["origin"] == "continue" and not t.of(tj.TOOL_PROPOSED)[0].data["parse_error"]
    assert t.of(tj.REASONING)[0].data["text"] == "let me think about it"
    # Inspect replays the reply as "\n" + text and the Muse template does not trim it (assumption 5): the text-level
    # prefix check records the difference instead of failing
    assert "<|start|>assistant to=user<|message|>\nI am thinking.<|eot|>" in c1.data["prompt"]
    assert c1.data["prefix_consistent_text"] is False and c1.data["prefix_consistent"] is False
    assert t.meta["conversion_audit"]["text_prefix_inconsistent_calls"] == 1


def test_plain_string_content_is_replayed_unchanged():
    msgs = [ChatMessageSystem(content="s"), ChatMessageUser(content="u"),
            asst("", "no reasoning here", plain=True), ChatMessageUser(content="go on"),
            asst("", "", [("t1", "bash", {"command": "ls"})], plain=True), tool("t1", "bash")]
    t = convert(sample(msgs))
    c0, c1 = t.of(tj.MODEL_CALL)
    assert c0.data["generation"] == " to=user<|message|>no reasoning here"
    assert c1.data["prefix_consistent"] is True
    assert tj.REASONING not in {e.kind for e in t.events}


def test_parallel_calls_one_turn():
    msgs = [ChatMessageSystem(content="You fix bugs."), ChatMessageUser(content="Fix issue 1."),
            asst("two commands at once", "", [("t1", "bash", {"command": "ls"}), ("t2", "bash", {"command": "pwd"}),
                                              ("t3", "submit", {"answer": "done"})]),
            tool("t1", "bash", "a.py\n"), tool("t2", "bash", "/w\n"), tool("t3", "submit", "")]
    t = convert(sample(msgs, kind="pass", value=1.0))
    (mc,) = t.of(tj.MODEL_CALL)
    assert mc.data["generation"].count("<|start|>assistant to=") == 3          # the first segment (self) is in the prompt
    assert mc.data["generation"].count("<|eom|>") == 3 and not mc.data["generation"].endswith("<|eom|>")
    prop = t.of(tj.TOOL_PROPOSED)
    assert [p.data["name"] for p in prop] == ["bash", "bash", "submit"]
    assert [p.data["arguments"] for p in prop] == [{"command": "ls"}, {"command": "pwd"}, {"answer": "done"}]
    raws = [p.data["raw"] for p in prop]
    assert len(set(raws)) == 3 and all(r.startswith("<atem:function_calls>") for r in raws)
    assert 'name="bash"' in raws[1] and "pwd" in raws[1] and 'name="submit"' in raws[2]
    assert [r.data["call_id"] for r in t.of(tj.TOOL_RESULT)] == ["call_1", "call_2", "call_3"]
    assert t.of(tj.SUBMISSION)[0].data["origin"] == "submit" and tj.labelled_submission_max_call(t) == 1
    # the next prompt (if there were one) renders the consecutive results, each closed by <|eot|> (tool messages
    # always end with it); the three calls inside the assistant turn are separated by <|eom|>
    msgs2 = msgs[:6] + [asst("again", "", [("t4", "bash", {"command": "ls"})]), tool("t4", "bash")]
    c1 = convert(sample(msgs2)).of(tj.MODEL_CALL)[1]
    assert "</tool_output><|eot|><|start|>tool bash<|message|>" in c1.data["prompt"]
    assert c1.data["prompt"].startswith(mc.data["prompt"] + mc.data["generation"] + EOT)
    assert c1.data["prefix_consistent"] is True


def test_malformed_call_is_kept_counted_and_not_a_submission():
    msgs = [ChatMessageSystem(content="s"), ChatMessageUser(content="u"),
            asst("run two", "", [("t1", "bash", {"command": "ls"}), ("t2", "bash", {}, True),
                                 ("t3", "submit", {}, True)]),
            tool("t1", "bash", "a.py\n"), tool("t2", "bash", "", error="could not parse"),
            tool("t3", "submit", "", error="could not parse"),
            ChatMessageUser(content="Please continue."),
            asst("retry", "", [("t4", "bash", {"command": "ls -la"})]), tool("t4", "bash")]
    resp = {0: [{"id": "t1", "function": {"arguments": '{"command": "ls"}'}},
                {"id": "t2", "function": {"arguments": '{"command": "ls'}},
                {"id": "t3", "function": {"arguments": "{oops"}}]}
    t = convert(sample(msgs, events=events_for(msgs, responses=resp)))
    prop = t.of(tj.TOOL_PROPOSED)
    assert [p.data["name"] for p in prop] == ["bash", "bash", "submit", "bash"]
    assert [bool(p.data["parse_error"]) for p in prop] == [False, True, True, False]
    assert prop[1].data["raw"] == '{"command": "ls' and prop[2].data["raw"] == "{oops"     # what vLLM returned
    assert prop[0].data["raw"].startswith("<atem:function_calls>")
    assert len(t.of(tj.TOOL_EXECUTED)) == 4 and len(t.of(tj.TOOL_RESULT)) == 4
    assert t.of(tj.TOOL_RESULT)[1].data["error_type"] == "parsing"
    # the malformed submit is not a submission: the next user message is a plain continue, the origin terminal_state
    assert t.of(tj.USER)[1].data["origin"] == "continue"
    assert t.of(tj.SUBMISSION)[0].data["origin"] == "terminal_state"
    a = t.meta["conversion_audit"]
    assert a["tool_calls"] == 4 and a["parse_error_calls"] == 2
    # history keeps all three calls (Inspect replays parse-error ones with empty arguments); results are named
    c1 = t.of(tj.MODEL_CALL)[1].data["prompt"]
    assert c1.count("<|start|>assistant to=bash<|message|>") == 2 and "<|start|>assistant to=submit<|message|>" in c1
    assert '<|start|>tool submit<|message|><tool_output name="submit">' in c1
    assert t.of(tj.MODEL_CALL)[1].data["prefix_consistent"] is True


def test_atem_markup_left_in_visible_text_is_flagged_not_dropped():
    txt = '<atem:function_calls>\n<atem:invoke name="bash">\n<atem:parameter name="command">ls'
    msgs = [ChatMessageSystem(content="s"), ChatMessageUser(content="u"), asst("try", txt),
            ChatMessageUser(content="Please continue.")]
    t = convert(sample(msgs))
    (a,) = t.of(tj.ASSISTANT)
    assert a.data["text"] == txt and a.data["atem_markup"] is True and not t.of(tj.TOOL_PROPOSED)
    assert t.meta["conversion_audit"]["atem_markup_in_text_calls"] == 1


def test_cut_off_reasoning_is_not_closed():
    msgs = [ChatMessageSystem(content="s"), ChatMessageUser(content="u"), asst("thinking and thinking", "")]
    t = convert(sample(msgs, events=events_for(msgs, stop={0: "max_tokens"})))
    assert t.of(tj.REASONING)[0].data["closed"] is False and t.of(tj.MODEL_CALL)[0].data["stop_type"] == "limit"
    t2 = convert(sample(msgs))
    assert t2.of(tj.REASONING)[0].data["closed"] is True


# ---------------------------------------------------------------------------------------------------------------
# recorded vLLM token ids

def test_recorded_ids_make_calls_exact_and_prefix_is_checked_by_ids():
    msgs = swe_run(submit=True)
    base = convert(sample(msgs))
    ids = ids_from(base)
    t = convert(sample(msgs, events=events_for(msgs, ids=ids)))
    for c, (pi, gi) in zip(t.of(tj.MODEL_CALL), ids):
        assert c.data["reconstruction"] == "exact" and c.data["tokens_source"] == "vllm"
        assert c.data["prompt_token_ids"] == pi and c.data["tokens"] == gi and gi[-1] == EOT_ID
    assert t.meta["reconstruction"]["exact"] is True and t.meta["reconstruction"]["tokens"] == "vllm"
    assert [c.data["prefix_consistent_ids"] for c in t.of(tj.MODEL_CALL)] == [None, True, True]
    a = t.meta["conversion_audit"]
    assert (a["append_only_checked"], a["non_append_only_calls"], a["non_append_only_call_indices"]) == (2, 0, [])
    rows = st.verify_recorded_ids(t, TOK)
    assert all(r["prompt_ids_match"] and r["generation_decode_match"] for r in rows)
    assert st.adopt_recorded_text(t, TOK) == 0
    assert {c.data["text_source"] for c in t.of(tj.MODEL_CALL)} == {"reconstructed_verified"}
    st.attach_retokenized_tokens(t, TOK)                     # must not overwrite recorded ids
    assert t.of(tj.MODEL_CALL)[0].data["tokens"] == ids[0][1]


def test_prompt_that_does_not_extend_the_previous_one_is_recorded_not_fatal():
    msgs = swe_run(submit=True)
    ids = ids_from(convert(sample(msgs)))
    ids[1] = ([7, 7, 7] + ids[1][0][3:], ids[1][1])          # call 1's prompt rewrote the start of the context
    t = convert(sample(msgs, events=events_for(msgs, ids=ids)))
    c = t.of(tj.MODEL_CALL)
    assert [x.data["prefix_consistent"] for x in c] == [None, False, False]    # call 2 also fails to extend call 1
    assert [x.data["prefix_consistent_text"] for x in c] == [None, True, True]
    a = t.meta["conversion_audit"]
    assert a["non_append_only_calls"] == 2 and a["non_append_only_call_indices"] == [1, 2]
    assert a["append_only_checked"] == 2 and tj.label(t) == 0 and len(t.of(tj.TOOL_PROPOSED)) == 3
    st.verify_recorded_ids(t, TOK)
    assert st.adopt_recorded_text(t, TOK) == 1                # only call 1's prompt disagrees with its ids
    d = c[1].data
    assert d["text_source"] == "decoded_ids" and d["prompt"].startswith("\x07\x07\x07")
    assert d["prompt_reconstructed"].startswith("<|begin_of_text|>") and d["prompt_sha256"] != c[0].data["prompt_sha256"]
    assert c[0].data["text_source"] == "reconstructed_verified"


def test_calls_without_ids_are_approximate_and_unaudited():
    msgs = swe_run(submit=True)
    ids = ids_from(convert(sample(msgs)))
    t = convert(sample(msgs, events=events_for(msgs, ids=[ids[0], None, ids[2]])))
    c = t.of(tj.MODEL_CALL)
    assert [x.data["reconstruction"] for x in c] == ["exact", "approximate", "exact"]
    assert t.meta["reconstruction"]["tokens"] == "mixed"
    assert t.meta["conversion_audit"]["append_only_checked"] == 0
    assert t.meta["conversion_audit"]["calls_without_recorded_ids"] == 1
    assert [x.data["prefix_consistent"] for x in c] == [None, True, True]     # falls back to the text comparison


def test_stripped_reasoning_is_repaired_from_recorded_ids():
    """If the vLLM parser stripped the reasoning's trailing newlines the reconstruction differs from what the model
    sampled; with the tokenizer the recorded ids give the real text."""
    msgs = [ChatMessageSystem(content="s"), ChatMessageUser(content="u"),
            asst("plan the edit", "", [("t1", "bash", {"command": "ls"})]), tool("t1", "bash")]
    base = convert(sample(msgs))
    real_gen = base.of(tj.MODEL_CALL)[0].data["generation"].replace("plan the edit", "plan the edit\n\n")
    ids = [(TOK.encode(base.of(tj.MODEL_CALL)[0].data["prompt"]), TOK.encode(real_gen + EOT))]
    t = convert(sample(msgs, events=events_for(msgs, ids=ids)))
    assert not any(r["generation_decode_match"] for r in st.verify_recorded_ids(t, TOK))
    assert st.adopt_recorded_text(t, TOK) == 1
    d = t.of(tj.MODEL_CALL)[0].data
    assert d["generation"] == real_gen and "plan the edit\n\n<|eom|>" in d["generation"]
    assert d["generation_reconstructed"] != real_gen and d["prompt"] == d["prompt_reconstructed"]


def test_reconstruct_exact_takes_muse_marks():
    msgs = swe_run(submit=True)
    ids = ids_from(convert(sample(msgs)))
    t = convert(sample(msgs, events=events_for(msgs, ids=ids)))
    from evals.reward_hacking import activations as act
    marks = {act.THINK_OPEN: 200006, act.THINK_CLOSE: 200007, act.CALL_OPEN: 200006, act.CALL_CLOSE: 200007,
             "end": frozenset({EOT_ID})}
    seqs = st.reconstruct_exact(t, TOK, marks=marks)
    assert [s.prompt_ids for s in seqs] == [p for p, _ in ids] and [s.gen_ids for s in seqs] == [g for _, g in ids]
    assert all(s.channel[-1] == act.END for s in seqs)


# ---------------------------------------------------------------------------------------------------------------
# the Inspect log file path (convert_log)

def _write_log(tmp_path, msgs, *, model, extra_body=None, effort=None):
    spec = EvalSpec(created="2026-01-01T00:00:00+00:00", task="swebench", dataset=EvalDataset(), model=model,
                    config=EvalConfig(), task_args={"split": "conflicting"},
                    model_generate_config=GenerateConfig(extra_body=extra_body, seed=5, reasoning_effort=effort))
    d = tmp_path / "inspect" / "repo__inst-1__r2"
    d.mkdir(parents=True)
    path = d / "log.eval"
    s = sample(msgs, events=events_for(msgs))
    write_eval_log(EvalLog(eval=spec, status="success", samples=[s]), str(path))
    return path


def test_convert_log_detects_muse_and_reads_strength(tmp_path):
    path = _write_log(tmp_path, swe_run(submit=True), model="vllm/muse-glimmer-30b",
                      extra_body={"top_k": 64, "chat_template_kwargs": {"reasoning_strength": "xhigh"}})
    assert st.detect_target(read_eval_log(str(path))) == "muse-glimmer-30b"
    (t,) = list(st.convert_log(path))
    assert t.meta["target_model"] == "muse-glimmer-30b" and t.meta["reasoning_effort_rendered"] == "xhigh"
    assert t.meta["rollout_id"] == "repo__inst-1__conflicting__r2" and t.meta["seed"] == 5
    assert "Reasoning strength: xhigh." in t.of(tj.MODEL_CALL)[0].data["prompt"]
    assert t.of(tj.MODEL_CALL)[0].data["generation"].startswith(" to=self<|message|>")


def test_convert_log_with_tokenizer_verifies_ids(tmp_path):
    msgs = swe_run(submit=True)
    ids = ids_from(convert(sample(msgs)))
    path = _write_log(tmp_path, msgs, model="vllm/muse-glimmer-30b",
                      extra_body={"chat_template_kwargs": {"reasoning_strength": "medium"}})
    log = read_eval_log(str(path))
    log.samples[0].events = events_for(msgs, ids=ids)
    write_eval_log(log, str(path))
    (t,) = list(st.convert_log(path, tokenizer=TOK))
    assert all(c.data["prompt_ids_match"] and c.data["generation_decode_match"] for c in t.of(tj.MODEL_CALL))
    assert {c.data["text_source"] for c in t.of(tj.MODEL_CALL)} == {"reconstructed_verified"}


def test_convert_log_target_model_override_and_qwen_default(tmp_path):
    msgs = swe_run(submit=True)
    path = _write_log(tmp_path, msgs, model="vllm/some-served-name",
                      extra_body={"chat_template_kwargs": {"reasoning_strength": "low"}})
    (t,) = list(st.convert_log(path, target_model="muse-glimmer-30b"))
    assert t.meta["target_model"] == "muse-glimmer-30b" and t.meta["reasoning_effort_rendered"] == "low"
    (q,) = list(st.convert_log(path))                          # nothing says Muse: the existing Qwen default
    assert q.meta["target_model"] == "qwen3.8-27b" and "conversion_audit" not in q.meta
    assert q.of(tj.MODEL_CALL)[0].data["prompt"].endswith("<|im_start|>assistant\n<think>\n")


def test_qwen_conversion_has_no_muse_fields():
    msgs = swe_run(submit=True)
    q = st.from_eval_sample(sample(msgs), instance_id="repo__inst-1", variant="conflicting", partition="train",
                            seed=7, fmt=llm.get_format("qwen3.8-27b"), tools=SWE_SPECS)
    assert "conversion_audit" not in q.meta and "chat_format" not in q.meta
    assert all("prefix_consistent_ids" not in c.data for c in q.of(tj.MODEL_CALL))
    assert all("atem_markup" not in a.data for a in q.of(tj.ASSISTANT))
    assert q.of(tj.MODEL_CALL)[0].data["generation"].startswith("look around first\n</think>\n\n<tool_call>")
