"""Muse Glimmer 30B format: byte-for-byte prompt parity with transformers, parsing, history round-trips.

Ground truth is `data/muse_raw_fixtures.json`: prompts rendered by transformers' apply_chat_template
(tools, reasoning_strength=medium) and real model samples. transformers' own `render_jinja_template` is also
used as an independent reference for synthetic conversations.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from evals.reward_hacking import muse_format
from evals.reward_hacking.llm import Completion, ParsedTurn
from evals.reward_hacking.muse_format import EOT, MuseFormat, MuseTurn

DATA = pathlib.Path(__file__).parent / "data"
FIX = json.loads((DATA / "muse_raw_fixtures.json").read_text())
SYSTEM = "You are a careful assistant. Use tools when helpful."
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Return the weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "run_python", "description": "Run python code and return stdout.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]
USER = {"single_call": "What is the weather in Testville in celsius?",
        "parallel_calls": "Get the weather in Paris and in Rome, both in one turn.",
        "code_call_multiline": "Use run_python to print the first 5 squares, one per line, using a for loop.",
        "plain_answer": "In one sentence, what is a prime number? Do not use tools."}
FMT = MuseFormat("medium")


def base(case):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER[case]}]


def tc(name, args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


def reference(messages, tools_=None, add_generation_prompt=True, **kw):
    from transformers.utils.chat_template_utils import render_jinja_template
    out, _ = render_jinja_template(conversations=[messages], tools=tools_ or None,
                                   chat_template=muse_format.MUSE_TEMPLATE_PATH.read_text(),
                                   add_generation_prompt=add_generation_prompt,
                                   bos_token=muse_format.BOS_TOKEN, **kw)
    return out[0]


SAMPLES = [(case, i, s) for case, c in FIX["cases"].items() for i, s in enumerate(c["samples"])]
HISTORY_MSGS = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "What is the weather in Testville in celsius?"},
    {"role": "assistant", "content": "", "reasoning_content": "I should call get_weather.",
     "tool_calls": [tc("get_weather", {"city": "Testville", "units": "celsius"})]},
    {"role": "tool", "name": "get_weather", "content": "Sunny, 19 C."},
]


# ---- prompt parity -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", list(FIX["cases"]))
def test_prompt_matches_fixture_byte_for_byte(case):
    assert FMT.render(base(case), TOOLS) == FIX["cases"][case]["prompt"]


@pytest.mark.parametrize("case", list(FIX["cases"]))
def test_prompt_matches_transformers_reference(case):
    assert FMT.render(base(case), TOOLS) == reference(base(case), TOOLS, reasoning_strength="medium")


def test_history_render_matches_fixture_in_all_tool_result_spellings():
    want = FIX["history_render"]["prompt"]
    assert FMT.render(HISTORY_MSGS, TOOLS) == want
    by_id = [dict(m) for m in HISTORY_MSGS]
    by_id[2]["tool_calls"] = [{"id": "c1", **by_id[2]["tool_calls"][0]}]
    by_id[3] = {"role": "tool", "tool_call_id": "c1", "content": "Sunny, 19 C."}
    assert FMT.render(by_id, TOOLS) == want
    # the agent loop appends a bare tool message: it is named after the call it answers
    bare = [dict(m) for m in HISTORY_MSGS]
    bare[3] = {"role": "tool", "content": "Sunny, 19 C."}
    assert FMT.render(bare, TOOLS) == want


def test_synthetic_conversations_match_transformers_reference():
    convs = [
        [{"role": "user", "content": "hi"}],                                  # no system: default system block
        [{"role": "system", "content": "Be brief. Reasoning effort: low."}, {"role": "user", "content": "q"},
         {"role": "assistant", "content": "  padded \n"}, {"role": "user", "content": "next"}],
        HISTORY_MSGS + [{"role": "assistant", "content": "Sunny.", "reasoning_content": "done"},
                        {"role": "user", "content": "thanks é中"}],
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "x"},
         {"role": "assistant", "content": "", "tool_calls": [tc("get_weather", {"city": "A"}),
                                                              tc("run_python", {"code": "print(1)\n"})]},
         {"role": "tool", "name": "get_weather", "content": "a"}, {"role": "tool", "name": "run_python", "content": "1"}],
    ]
    for msgs in convs:
        for tools_ in (TOOLS, None):
            for agp in (True, False):
                import re
                got = FMT.render(msgs, tools_ or [], agp)
                want = reference(msgs, tools_, agp, reasoning_strength="medium")
                if not any(m["role"] == "system" for m in msgs):          # date line: both read the clock
                    got, want = (re.sub(r"Current date: [\d-]+\.", "", x) for x in (got, want))
                assert got == want


# ---- the reasoning_strength kwarg ----------------------------------------------------------------------------

@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
def test_reasoning_strength_reaches_system_block(effort):
    p = MuseFormat(effort).render(base("single_call"), TOOLS)
    assert f"\n\nReasoning strength: {effort}.\n\n" in p
    assert p.startswith("<|begin_of_text|><|start|>system<|message|>" + SYSTEM)
    assert MuseFormat(effort).describe()["reasoning_effort"] == effort


def test_default_effort_is_template_default_high():
    f = MuseFormat()
    assert "Reasoning strength: high." in f.render(base("single_call"), TOOLS)
    assert f.reasoning_effort is None and "high (template default)" in f.describe()["reasoning_effort"]


@pytest.mark.parametrize("bad", ["max", "minimal", "", "HIGH"])
def test_rejects_other_efforts(bad):
    with pytest.raises(ValueError):
        MuseFormat(bad)


def test_registry_fields():
    f = MuseFormat()
    assert (f.name, f.hf_repo, f.local_dir, f.num_hidden_layers, f.probe_layers) == (
        "muse-glimmer-30b", "meta-models/Muse-Glimmer-30B", "muse-glimmer-30b", 52, (13, 26, 39))
    assert f.hf_revision == "cec94fcfd6512f2cfe708138c32533b346ea4f69"
    assert f.reasoning_efforts == ("low", "medium", "high", "xhigh") and f.default_effort == "high"
    assert f.sampling["top_k"] == 64 and f.sampling["temperature"] == 1.0 and f.sampling["top_p"] == 0.95
    assert f.template_path() == DATA / "muse-glimmer-30b_chat_template.jinja"
    assert f.stop_token_ids == (200008,)
    assert f.describe()["stop_token_ids"] == [200008]


# ---- stop tokens ----------------------------------------------------------------------------------------------

def test_stop_token_is_eot_and_not_eom():
    ends = {s["token_ids"][-1] for _, c in FIX["cases"].items() for s in c["samples"]}
    ends |= {s["token_ids"][-1] for s in FIX["history_render"]["samples"]}
    assert ends == {200008} == set(MuseFormat.stop_token_ids)
    assert all(s["finish_reason"] == "stop" for _, _, s in SAMPLES)
    # the stop token is not in the text; <|eom|> (200007) occurs inside completions, so it must not stop one
    assert all(EOT not in s["text"] for _, _, s in SAMPLES)
    assert any("<|eom|>" in s["text"] for _, _, s in SAMPLES)
    assert 200007 in FIX["cases"]["parallel_calls"]["samples"][0]["token_ids"]
    assert muse_format.EOM_TOKEN_ID not in MuseFormat.stop_token_ids


def comp(text, tokens=(1, 2, 200008), stop="eos"):
    return Completion(text=text, tokens=list(tokens), stop_type=stop, prompt_tokens=10, cached_tokens=0, seconds=0.0)


def test_expected_next_prefix():
    assert FMT.expected_next_prefix("P", comp("abc")) == "Pabc" + EOT
    assert FMT.expected_next_prefix("P", comp("abc" + EOT)) == "Pabc" + EOT           # server kept the stop token
    assert FMT.expected_next_prefix("P", comp("abc", tokens=(1, 2), stop="limit")) == "Pabc"
    assert FMT.expected_next_prefix("P", comp("abc", tokens=(1, 2, 7))) == "Pabc"     # ended on something else


# ---- real samples: parse + round trip -------------------------------------------------------------------------

@pytest.mark.parametrize("case,i,sample", SAMPLES, ids=[f"{c}-{i}" for c, i, _ in SAMPLES])
def test_fixture_sample_round_trips_byte_for_byte(case, i, sample):
    prompt = FIX["cases"][case]["prompt"]
    turn = FMT.parse_generation(sample["text"], TOOLS)
    assert isinstance(turn, MuseTurn) and isinstance(turn, ParsedTurn)
    assert turn.canonical and not turn.unparsed and turn.reasoning_closed
    assert all(c.parse_error is None for c in turn.tool_calls)
    again = FMT.render(base(case) + [FMT.assistant_message(turn)], TOOLS, add_generation_prompt=False)
    assert again == prompt + sample["text"] + EOT
    # and the loop's check: the next prompt (assistant + a follow-up user turn) starts with the expected prefix
    nxt = FMT.render(base(case) + [FMT.assistant_message(turn), {"role": "user", "content": "go on"}], TOOLS)
    c = comp(sample["text"], sample["token_ids"][-3:])
    assert nxt.startswith(FMT.expected_next_prefix(prompt, c))


def test_history_samples_round_trip():
    prompt = FIX["history_render"]["prompt"]
    for s in FIX["history_render"]["samples"]:
        turn = FMT.parse_generation(s["text"], TOOLS)
        assert turn.canonical and turn.tool_calls == () and turn.content.startswith(("Testville", "The weather"))
        assert FMT.render(HISTORY_MSGS + [FMT.assistant_message(turn)], TOOLS, False) == prompt + s["text"] + EOT


def test_single_call_parsed_fields():
    s = FIX["cases"]["single_call"]["samples"][0]["text"]
    t = FMT.parse_generation(s, TOOLS)
    assert t.reasoning.startswith("We need weather in Testville") and t.reasoning.endswith('units="celsius".')
    assert t.content == "" and len(t.tool_calls) == 1
    c = t.tool_calls[0]
    assert (c.name, c.arguments) == ("get_weather", {"city": "Testville", "units": "celsius"})
    assert c.raw.startswith("<atem:function_calls>\n<atem:invoke") and c.raw.endswith("</atem:function_calls>")
    msg = FMT.assistant_message(t)
    assert msg["tool_calls"][0]["function"]["arguments"] == {"city": "Testville", "units": "celsius"}
    assert isinstance(msg["tool_calls"][0]["function"]["arguments"], dict)


def test_parallel_calls_in_order():
    for s in FIX["cases"]["parallel_calls"]["samples"]:
        t = FMT.parse_generation(s["text"], TOOLS)
        assert [c.name for c in t.tool_calls] == ["get_weather", "get_weather"]
        assert [c.arguments["city"] for c in t.tool_calls] == ["Paris", "Rome"]
        assert len(FMT.assistant_message(t)["tool_calls"]) == 2
    # the rendered history separates them with <|eom|> and ends the last with <|eot|>
    t = FMT.parse_generation(FIX["cases"]["parallel_calls"]["samples"][0]["text"], TOOLS)
    r = FMT.render(base("parallel_calls") + [FMT.assistant_message(t)], TOOLS, False)
    assert r.count("</atem:function_calls><|eom|><|start|>assistant to=get_weather") == 1 and r.endswith(EOT)


def test_multiline_parameter_value_kept_raw():
    for s in FIX["cases"]["code_call_multiline"]["samples"]:
        t = FMT.parse_generation(s["text"], TOOLS)
        assert t.tool_calls[0].name == "run_python"
        assert t.tool_calls[0].arguments == {"code": "for i in range(1, 6):\n    print(i * i)"}


def test_plain_answer():
    for s in FIX["cases"]["plain_answer"]["samples"]:
        t = FMT.parse_generation(s["text"], TOOLS)
        assert t.tool_calls == () and t.reasoning_closed
        assert t.content == "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself."
        assert t.reasoning and "<|" not in t.reasoning
        m = FMT.assistant_message(t)
        assert "tool_calls" not in m and m["content"] == t.content


def test_raw_whitespace_survives_the_round_trip():
    text = " to=self<|message|>\n  think \n\n<|eom|><|start|>assistant to=user<|message|>  answer \n"
    t = FMT.parse_generation(text, TOOLS)
    assert t.reasoning == "think" and t.content == "answer" and t.canonical
    m = FMT.assistant_message(t)
    assert m["reasoning_content"] == "\n  think \n\n" and m["content"] == "  answer \n"
    assert FMT.render(base("plain_answer") + [m], TOOLS, False) == FIX["cases"]["plain_answer"]["prompt"] + text + EOT


def test_no_reasoning_and_trailing_eot_in_text():
    text = " to=user<|message|>Hello"
    for s in (text, text + EOT):
        t = FMT.parse_generation(s, TOOLS)
        assert t.reasoning == "" and t.content == "Hello" and t.canonical


# ---- typed arguments --------------------------------------------------------------------------------------------

TYPED_TOOLS = [{"type": "function", "function": {
    "name": "edit", "description": "d", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "items": {"type": "array", "items": {"type": "integer"}},
        "opts": {"type": "object"}, "flag": {"type": "boolean"}, "n": {"type": "integer"},
        "note": {"type": ["string", "null"]}, "free": {"description": "no type"},
        "maybe": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}, "required": ["path"]}}}]


def wire(name, params, recipient=None):
    body = "".join(f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in params)
    return (f' to=self<|message|>r<|eom|><|start|>assistant to={recipient or name}<|message|>'
            f'<atem:function_calls>\n<atem:invoke name="{name}">\n{body}</atem:invoke>\n</atem:function_calls>')


def test_object_and_list_arguments_from_the_wire_format():
    text = wire("edit", [("path", "/a b/é.py"), ("items", "[1, 2, 3]"), ("opts", '{"a": {"b": [1, "é"]}, "c": null}'),
                         ("flag", "true"), ("n", "-7"), ("note", "null"), ("free", "[1]"), ("maybe", "null")])
    t = FMT.parse_generation(text, TYPED_TOOLS)
    assert t.canonical and t.tool_calls[0].parse_error is None
    assert t.tool_calls[0].arguments == {
        "path": "/a b/é.py", "items": [1, 2, 3], "opts": {"a": {"b": [1, "é"]}, "c": None}, "flag": True,
        "n": -7, "note": None, "free": "[1]", "maybe": None}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, FMT.assistant_message(t)]
    r = FMT.render(msgs, TYPED_TOOLS, False)
    assert r.endswith(text + EOT)
    assert r == reference(msgs, TYPED_TOOLS, False, reasoning_strength="medium")


def test_string_values_that_look_like_other_types_stay_strings():
    text = wire("edit", [("path", "true"), ("note", "null")])
    t = FMT.parse_generation(text, TYPED_TOOLS)
    assert t.tool_calls[0].arguments == {"path": "true", "note": None}   # a ["string","null"] union: literal null is None
    t = FMT.parse_generation(wire("edit", [("path", "123")]), TYPED_TOOLS)
    assert t.tool_calls[0].arguments == {"path": "123"} and t.canonical


def test_non_canonical_json_spacing_is_flagged_not_hidden():
    t = FMT.parse_generation(wire("edit", [("path", "p"), ("items", "[1,2]")]), TYPED_TOOLS)
    assert t.tool_calls[0].arguments["items"] == [1, 2] and t.tool_calls[0].parse_error is None
    assert not t.canonical          # re-rendering writes "[1, 2]"; the loop's prefix check will report it


# ---- malformed calls are kept and reported ---------------------------------------------------------------------

HEAD = " to=self<|message|>r<|eom|><|start|>assistant to=get_weather<|message|>"


@pytest.mark.parametrize("tail,needle", [
    ('<atem:function_calls>\n<atem:invoke name="get_weather">\n<atem:parameter name="city">Par', "unterminated <atem:parameter"),
    ('<atem:function_calls>\n<atem:invoke name="get_weather">\n<atem:parameter name="city">Paris</atem:parameter>\n', "unterminated <atem:invoke>"),
    ('<atem:function_calls>\n<atem:invoke name="get_weather">\n<atem:parameter name="city">Paris</atem:parameter>\n</atem:invoke>\n', "unterminated <atem:function_calls>"),
    ('<atem:function_calls>\n<atem:invoke name="get_weather">\ncity=Paris\n</atem:invoke>\n</atem:function_calls>', "expected <atem:parameter>"),
    ('<atem:function_calls>\n</atem:function_calls>', "empty"),
    ('<atem:function_calls>\ngarbage', "expected <atem:invoke>"),
    ('get_weather(city="Paris")', "no <atem:function_calls> block"),
    ('<atem:invoke name="get_weather">\n<atem:parameter name="city">P</atem:parameter>\n</atem:invoke>', "outside"),
    ('<atem:function_calls>\n<atem:invoke name="get_weather">\n<atem:parameter name="city">a</atem:parameter>\n'
     '<atem:parameter name="city">b</atem:parameter>\n</atem:invoke>\n</atem:function_calls>', "duplicate"),
])
def test_malformed_call_is_reported(tail, needle):
    t = FMT.parse_generation(HEAD + tail, TOOLS)
    assert len(t.tool_calls) == 1
    c = t.tool_calls[0]
    assert c.parse_error and needle in c.parse_error, c.parse_error
    assert c.raw
    assert not t.canonical
    m = FMT.assistant_message(t)
    assert "tool_calls" not in m and m["tool_call_names"] == [c.name or "unknown"]


def test_good_and_bad_calls_together_and_tool_result_names():
    good = ('<atem:function_calls>\n<atem:invoke name="get_weather">\n<atem:parameter name="city">Paris</atem:parameter>\n'
            '</atem:invoke>\n</atem:function_calls>')
    bad = '<atem:function_calls>\n<atem:invoke name="run_python">\n<atem:parameter name="code">print('
    text = HEAD + good + "<|eom|><|start|>assistant to=run_python<|message|>" + bad
    t = FMT.parse_generation(text, TOOLS)
    assert [(c.name, c.parse_error is None) for c in t.tool_calls] == [("get_weather", True), ("run_python", False)]
    m = FMT.assistant_message(t)
    assert [c["function"]["name"] for c in m["tool_calls"]] == ["get_weather"]
    r = FMT.render(base("single_call") + [m, {"role": "tool", "content": "ok"}, {"role": "tool", "content": "err"}], TOOLS)
    assert "<|start|>tool get_weather<|message|><tool_output name=\"get_weather\">\nok\n</tool_output><|eot|>" in r
    assert "<|start|>tool run_python<|message|><tool_output name=\"run_python\">\nerr\n" in r


def test_cut_off_reasoning_is_not_closed():
    t = FMT.parse_generation(" to=self<|message|>still thinking", TOOLS)
    assert not t.reasoning_closed and t.reasoning == "still thinking" and t.tool_calls == () and t.content == ""
    assert FMT.parse_generation(" to=self<|message|>x<|eom|>", TOOLS).reasoning_closed


def test_unparseable_text_is_kept_observable():
    t = FMT.parse_generation(" to=self<|message|>r<|eom|>stray text without a header", TOOLS)
    assert t.unparsed == ("stray text without a header",) and not t.canonical
    t = FMT.parse_generation("junk<|channel|><|start|>assistant to=user<|message|>ok", TOOLS)
    assert t.content == "ok" and t.unparsed == ("junk<|channel|>",)


def test_call_in_user_channel_is_a_call_and_text_around_it_is_kept():
    call = '<atem:function_calls>\n<atem:invoke name="get_weather">\n<atem:parameter name="city">X</atem:parameter>\n</atem:invoke>\n</atem:function_calls>'
    t = FMT.parse_generation(f" to=user<|message|>Checking. {call}", TOOLS)
    assert [c.name for c in t.tool_calls] == ["get_weather"] and t.content == "Checking."
    assert not t.canonical          # the template drops text that accompanies tool calls


def test_assistant_message_accepts_a_plain_parsed_turn():
    m = FMT.assistant_message(ParsedTurn(reasoning="r", content="c", tool_calls=()))
    assert m == {"role": "assistant", "content": "c", "reasoning_content": "r"}


def test_unknown_tool_keeps_text_values():
    t = FMT.parse_generation(wire("mystery", [("x", "5")], recipient="mystery"), TOOLS)
    assert t.tool_calls[0].arguments == {"x": "5"} and t.tool_calls[0].parse_error is None
