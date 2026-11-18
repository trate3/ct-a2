"""Tests for evals/reward_hacking/llm.py: tool-call parsing and the chat-template
prefix-consistency property the agent loop relies on, plus retry behavior of the
completion client. No network: `requests` is mocked.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from evals.reward_hacking.llm import (
    CompletionClient,
    assistant_message,
    parse_generation,
    render,
)

TOOLS = [
    {"type": "function", "function": {"name": "bash",
                                       "parameters": {"properties": {"cmd": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "run",
                                       "parameters": {"properties": {
                                           "n": {"type": "integer"},
                                           "items": {"type": "array"},
                                           "name": {"type": "string"},
                                       }}}},
]

MESSAGES = [
    {"role": "system", "content": "You are a careful coding agent."},
    {"role": "user", "content": "Implement f() so it passes the tests."},
]


# ---------------------------------------------------------------------------
# parse_generation

def test_reasoning_and_one_tool_call():
    text = ("Let me look at the files.\n</think>\n\n"
            "<tool_call>\n<function=bash>\n<parameter=cmd>\nls -la\n</parameter>\n</function>\n</tool_call>")
    turn = parse_generation(text, TOOLS)
    assert turn.reasoning == "Let me look at the files."
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "bash"
    assert call.arguments == {"cmd": "ls -la"}
    assert call.parse_error is None


def test_multiple_tool_calls_with_multiline_values_strip_exactly_one_newline():
    text = ("reasoning\n</think>\n\n"
            "<tool_call>\n<function=bash>\n<parameter=cmd>\n\nls\npwd\n\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=bash>\n<parameter=cmd>\nline1\nline2\n</parameter>\n</function>\n</tool_call>")
    turn = parse_generation(text, TOOLS)
    assert len(turn.tool_calls) == 2
    # exactly one leading and one trailing newline stripped; interior blank lines survive
    assert turn.tool_calls[0].arguments == {"cmd": "\nls\npwd\n"}
    assert turn.tool_calls[1].arguments == {"cmd": "line1\nline2"}


def test_typed_params_coerced_string_params_left_as_strings():
    text = ("reasoning\n</think>\n\n"
            "<tool_call>\n<function=run>\n"
            "<parameter=n>\n42\n</parameter>\n"
            "<parameter=items>\n[1, 2, 3]\n</parameter>\n"
            "<parameter=name>\n123\n</parameter>\n"
            "</function>\n</tool_call>")
    turn = parse_generation(text, TOOLS)
    call = turn.tool_calls[0]
    assert call.arguments["n"] == 42
    assert isinstance(call.arguments["n"], int)
    assert call.arguments["items"] == [1, 2, 3]
    assert isinstance(call.arguments["items"], list)
    # declared as a string param: stays a string even though it looks like a number
    assert call.arguments["name"] == "123"
    assert isinstance(call.arguments["name"], str)


def test_typed_param_string_that_looks_like_json_stays_string():
    text = ('reasoning\n</think>\n\n'
            '<tool_call>\n<function=run>\n<parameter=name>\n{"a": 1}\n</parameter>\n</function>\n</tool_call>')
    turn = parse_generation(text, TOOLS)
    assert turn.tool_calls[0].arguments["name"] == '{"a": 1}'
    assert isinstance(turn.tool_calls[0].arguments["name"], str)


def test_no_think_close_keeps_all_text_as_private_reasoning():
    text = "this just keeps going and never closes the reasoning channel"
    turn = parse_generation(text, TOOLS)
    assert turn.reasoning == text and not turn.reasoning_closed
    assert turn.content == ""
    assert turn.tool_calls == ()


def test_unterminated_tool_call_yields_parse_error():
    text = ("reasoning\n</think>\n\nSome text before.\n\n"
            "<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n")
    turn = parse_generation(text, TOOLS)
    assert turn.content == "Some text before."
    assert len(turn.tool_calls) == 1
    bad = turn.tool_calls[0]
    assert bad.name == ""
    assert bad.arguments == {}
    assert bad.parse_error is not None
    assert "<tool_call>" in bad.raw


def test_content_before_a_tool_call():
    text = ("reasoning\n</think>\n\nHere is my plan.\n\n"
            "<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>")
    turn = parse_generation(text, TOOLS)
    assert turn.content == "Here is my plan."
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "bash"


# ---------------------------------------------------------------------------
# render prefix-consistency
#
# The agent loop renders `messages` with add_generation_prompt=True to build the
# prompt, gets back raw generation text from the server, and later needs
# render(messages + [assistant_message(turn)], ..., add_generation_prompt=False)
# to reproduce prompt + generation_text + "<|im_end|>\n" exactly, so that a
# continued rollout's prompt stays byte-identical to what was actually generated.


def _check_prefix_consistency(gen_text: str) -> bool:
    turn = parse_generation(gen_text, TOOLS)
    msg = assistant_message(turn)
    full = render(MESSAGES + [msg], TOOLS, add_generation_prompt=False)
    prefix = render(MESSAGES, TOOLS, add_generation_prompt=True)
    expected = prefix + gen_text + "<|im_end|>\n"
    return full == expected


@pytest.mark.parametrize("gen_text", [
    # reasoning + single tool call, no visible content, single-line param
    "reasoning\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>",
    # reasoning + visible content before the tool call, separated by a blank line
    # on both sides (the template always normalizes to exactly "\n\n" there)
    "reasoning\n</think>\n\nHere is my plan.\n\n<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>",
    # two tool calls, single newline between them (how the template emits them)
    "reasoning\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=bash>\n<parameter=cmd>\npwd\n</parameter>\n</function>\n</tool_call>",
    # no tool call at all, just a visible answer
    "reasoning\n</think>\n\nJust a plain answer.",
    # multi-line parameter value
    "reasoning\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=cmd>\nline1\nline2\n</parameter>\n</function>\n</tool_call>",
])
def test_canonical_shapes_satisfy_prefix_consistency(gen_text):
    assert _check_prefix_consistency(gen_text)


def test_single_newline_before_tool_call_breaks_prefix_consistency():
    """A common, almost-canonical shape that does NOT round-trip.

    The template always renders exactly "\\n\\n" between visible content and a
    following <tool_call> block. If the model's own generation used a single
    newline there instead, the reconstructed prompt differs from the raw
    generation text -- this is a real divergence risk for the agent loop, not a
    test artifact, so it is recorded here rather than silently assumed away.
    """
    gen_text = ("reasoning\n</think>\n\nHere is my plan.\n"
                "<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>")
    assert not _check_prefix_consistency(gen_text)


# ---------------------------------------------------------------------------
# CompletionClient retries

def test_completion_client_retries_on_503_then_succeeds(monkeypatch):
    import evals.reward_hacking.llm as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda _: None)

    resp_503 = Mock(status_code=503)
    resp_ok = Mock(status_code=200)
    resp_ok.raise_for_status = Mock()
    resp_ok.json = Mock(return_value={
        "content": "reasoning\n</think>\n\nok",
        "tokens": [1, 2, 3],
        "stop_type": "eos",
        "tokens_evaluated": 10,
        "timings": {"cache_n": 4},
    })
    post = Mock(side_effect=[resp_503, resp_ok])
    monkeypatch.setattr(llm_mod.requests, "post", post)

    client = CompletionClient(url="http://fake", retries=3)
    completion = client.complete(prompt="hi", max_tokens=16, seed=1, sampling={})

    assert post.call_count == 2
    assert completion.tokens == [1, 2, 3]
    assert completion.stop_type == "eos"
    assert completion.prompt_tokens == 10
    assert completion.cached_tokens == 4
    assert completion.text == "reasoning\n</think>\n\nok"


def test_completion_client_gives_up_after_exhausting_retries(monkeypatch):
    import evals.reward_hacking.llm as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda _: None)
    resp_503 = Mock(status_code=503)
    # on the final attempt the code falls through to raise_for_status() instead of
    # the explicit "still 503" branch (attempt < retries is false), so a real
    # requests.Response would raise HTTPError there -- model that.
    resp_503.raise_for_status = Mock(side_effect=llm_mod.requests.exceptions.HTTPError("503"))
    post = Mock(return_value=resp_503)
    monkeypatch.setattr(llm_mod.requests, "post", post)

    client = CompletionClient(url="http://fake", retries=2)
    with pytest.raises(llm_mod.requests.exceptions.HTTPError):
        client.complete(prompt="hi", max_tokens=16, seed=1, sampling={})
    assert post.call_count == 3  # initial attempt + 2 retries
