"""Qwen3.8-27B chat format: official template pin, render parity with transformers, shared Qwen XML parsing.

Reference rendering: transformers' `render_jinja_template` over the OFFICIAL template (Qwen/Qwen3.8-27B
@1d4bf0f2, sha256 c3cf9e34...), independent of our sandboxed-jinja renderer.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from evals.reward_hacking import llm, tools

Q38 = llm.get_format("qwen3.8-27b")
Q36 = llm.get_format("qwen3.6")
TOOLS = tools.TOOL_SPECS


def tc(name, args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


CONVERSATIONS = {
    "system_user": [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "hi"}],
    "tool_loop": [
        {"role": "system", "content": "You are a coding agent.\nUse the tools."},
        {"role": "user", "content": "Implement func.py."},
        {"role": "assistant", "content": "Let me look.", "reasoning_content": "First read.\nThen edit.",
         "tool_calls": [tc("bash", {"cmd": "ls /; cat /func.py"}),
                        tc("text_editor", {"command": "insert", "path": "/func.py", "insert_line": 3,
                                           "new_str": "x = 1\n\ty = \"é中\"", "view_range": [1, -1]})]},
        {"role": "tool", "content": "func.py test.py"},
        {"role": "tool", "content": "Error: file not found"},
        {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [tc("submit", {"answer": "done"})]},
        {"role": "tool", "content": "done"},
        {"role": "user", "content": "Your submission was incorrect."},
    ],
    "no_reasoning_key": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "  padded answer \n"},
        {"role": "user", "content": "next"},
    ],
}


class FakeComp:
    def __init__(self, text, last_token, stop_type="eos"):
        self.text, self.tokens, self.stop_type = text, [5, 6, last_token], stop_type


def reference(messages, tools_=None, add_generation_prompt=True, **kw):
    from transformers.utils.chat_template_utils import render_jinja_template
    out, _ = render_jinja_template(conversations=[messages], tools=tools_ or None,
                                   chat_template=llm.QWEN38_TEMPLATE_PATH.read_text(),
                                   add_generation_prompt=add_generation_prompt, **kw)
    return out[0]


def test_template_pin_and_registry():
    assert hashlib.sha256(llm.QWEN38_TEMPLATE_PATH.read_bytes()).hexdigest() == \
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    assert Q38.name == "qwen3.8-27b" and Q38.hf_repo == "Qwen/Qwen3.8-27B"
    assert Q38.hf_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert Q38.template_digest() == hashlib.sha256(llm.QWEN38_TEMPLATE_PATH.read_bytes()).hexdigest()
    assert Q38.template_digest() != Q36.template_digest()
    assert Q38.stop_token_ids == (248046, 248044)         # generation_config.json: <|im_end|>, <|endoftext|>
    d = Q38.describe()
    assert d["reasoning_effort"] == "xhigh (template default)" and d["template_sha256"] == Q38.template_digest()
    cfg = json.loads((llm.QWEN38_TEMPLATE_PATH.parent / "qwen3.8-27b.config.json").read_text())["text_config"]
    assert cfg["num_hidden_layers"] == Q38.num_hidden_layers == 64
    assert Q38.probe_layers == tuple(round(Q38.num_hidden_layers * f) for f in (.25, .5, .75)) == (16, 32, 48)
    assert Q36.probe_layers == (10, 20, 30) and Q36.num_hidden_layers == 40       # the pilot layers


@pytest.mark.parametrize("name", sorted(CONVERSATIONS))
@pytest.mark.parametrize("with_tools", [True, False])
@pytest.mark.parametrize("gen_prompt", [True, False])
def test_render_matches_official_template_reference(name, with_tools, gen_prompt):
    msgs, t = CONVERSATIONS[name], (TOOLS if with_tools else [])
    assert Q38.render(msgs, t, add_generation_prompt=gen_prompt) == \
        reference(msgs, t, gen_prompt, preserve_thinking=True)


def test_preserve_thinking_default_makes_our_explicit_flag_a_no_op():
    msgs = CONVERSATIONS["tool_loop"]
    assert reference(msgs, TOOLS) == reference(msgs, TOOLS, preserve_thinking=True)


@pytest.mark.parametrize("effort", ["xhigh", "medium", "low"])
def test_reasoning_effort_values(effort):
    msgs = CONVERSATIONS["system_user"]
    f = llm.get_format("qwen3.8-27b", reasoning_effort=effort)
    assert f.render(msgs, TOOLS) == reference(msgs, TOOLS, preserve_thinking=True, reasoning_effort=effort)
    text = f.render(msgs, TOOLS)
    assert (f"Reasoning effort is set to {effort}." in text) == (effort != "medium")
    with pytest.raises(ValueError):
        llm.get_format("qwen3.8-27b", reasoning_effort="max")


def test_default_effort_is_xhigh_and_is_not_passed():
    msgs = CONVERSATIONS["system_user"]
    assert Q38.render(msgs, TOOLS) == llm.get_format("qwen3.8-27b", reasoning_effort="xhigh").render(msgs, TOOLS)
    assert Q38.render(msgs, [], True).startswith("<|im_start|>system\nReasoning effort is set to xhigh.")
    # thinking off: no effort instruction, and the closed empty think block, like Qwen3.6
    off = Q38.render(msgs, [], enable_thinking=False)
    assert "Reasoning effort" not in off and off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert off == reference(msgs, None, True, preserve_thinking=True, enable_thinking=False)


def test_differences_from_the_qwen36_pin_are_confined_to_the_system_turn_for_string_arguments():
    msgs = CONVERSATIONS["tool_loop"]
    a, b = Q36.render(msgs, TOOLS), Q38.render(msgs, TOOLS)
    effort = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
              "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
              "clarity in the final answer.\n\n")
    assert b == a.replace("<|im_start|>system\n# Tools", "<|im_start|>system\n" + effort + "# Tools", 1)


def test_non_string_scalars_render_as_json_in_38_but_python_in_36():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "reasoning_content": "r",
             "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"flag": True, "n": None}}}]},
            {"role": "tool", "content": "x"}]
    assert "<parameter=flag>\ntrue\n" in Q38.render(msgs, []) and "<parameter=n>\nnull\n" in Q38.render(msgs, [])
    assert "<parameter=flag>\nTrue\n" in Q36.render(msgs, [])


def test_render_requires_a_user_turn_and_a_leading_system():
    import jinja2
    with pytest.raises(jinja2.TemplateError):
        Q38.render([{"role": "system", "content": "s"}], [])
    with pytest.raises(jinja2.TemplateError):
        Q38.render([{"role": "user", "content": "u"}, {"role": "system", "content": "s"}], [])


# --- parsing and round trips are the Qwen3.6 behaviour (same XML format) -------------------------------

def _tool(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


GENS = [
    "reasoning\n</think>\n\n" + _tool("bash", cmd="ls"),
    "reasoning\n</think>\n\nHere is my plan.\n\n" + _tool("bash", cmd="ls"),
    "reasoning\n</think>\n\n" + _tool("bash", cmd="ls") + "\n" + _tool("bash", cmd="pwd"),
    "reasoning\n</think>\n\nJust a plain answer.",
    "reasoning\n</think>\n\n" + _tool("bash", cmd="line1\nline2"),
    "r\n</think>\n\n" + _tool("text_editor", command="insert", path="/f.py", insert_line=3, new_str="x\ny"),
]


@pytest.mark.parametrize("gen", GENS)
def test_parse_is_identical_to_qwen36_and_round_trips(gen):
    t38, t36 = Q38.parse_generation(gen, TOOLS), Q36.parse_generation(gen, TOOLS)
    assert t38 == t36 and Q38.assistant_message(t38) == Q36.assistant_message(t36)
    msgs = CONVERSATIONS["system_user"]
    full = Q38.render(msgs + [Q38.assistant_message(t38)], TOOLS, add_generation_prompt=False)
    assert full == Q38.render(msgs, TOOLS) + gen + "<|im_end|>\n"
    assert Q38.expected_next_prefix("P", FakeComp("G", 1)) == "PG<|im_end|>\n"


def test_default_format_is_still_qwen36():
    assert llm.get_format().name == "qwen3.6" and llm.DEFAULT_FORMAT == "qwen3.6"


def test_default_probe_layers_follow_the_target_depth():
    import argparse
    from evals.reward_hacking.scripts import extract_activations as ea
    from evals_runner import rh_probe
    assert ea.default_layers() == list(ea.DEFAULT_LAYERS) == [10, 20, 30]
    assert ea.default_layers("qwen3.8-27b") == [16, 32, 48]
    assert rh_probe._layers(argparse.Namespace(layers=None, target_model="qwen3.8-27b")) == [16, 32, 48]
    assert rh_probe._layers(argparse.Namespace(layers=None)) == list(rh_probe.LAYERS)
    assert rh_probe._layers(argparse.Namespace(layers=[7], target_model="qwen3.8-27b")) == [7]
    p = ea.argparse.ArgumentParser()
    ea.add_model_args(p)
    assert p.parse_args(["--model", "m", "--target-model", "qwen3.8-27b"]).target_model == "qwen3.8-27b"
