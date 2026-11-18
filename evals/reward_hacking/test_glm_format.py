"""GLM-5.3-Flash chat format: render parity with the official template, parsing, round-trips, Qwen intact.

The reference rendering is transformers' own `render_jinja_template` (the code path behind
`tokenizer.apply_chat_template`) over the OFFICIAL template file from zai-org/GLM-5.3-Flash (sha256
2614a3c8...), so our sandboxed-jinja rendering is checked against an independent implementation.
Fixtures are built from the template's documented format: `<tool_call>NAME<arg_key>K</arg_key>
<arg_value>V</arg_value>...</tool_call>`, `<think>..</think>` and `<|observation|><tool_response>..`.
"""

from __future__ import annotations

import json

import pytest

from evals.reward_hacking import llm, tools
from evals.reward_hacking.conftest import fixture_task, requires_sandbox
from evals.reward_hacking import rollout as rollout_mod
from evals.reward_hacking import trajectory as tj
from evals.reward_hacking.tasks import Task

GLM = llm.get_format("glm-5.3-flash")
TOOLS = tools.TOOL_SPECS


def reference(messages, tools_=None, add_generation_prompt=True, **kw):
    from transformers.utils.chat_template_utils import render_jinja_template
    out, _ = render_jinja_template(conversations=[messages], tools=tools_ or None,
                                   chat_template=llm.GLM_TEMPLATE_PATH.read_text(),
                                   add_generation_prompt=add_generation_prompt, **kw)
    return out[0]


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


@pytest.mark.parametrize("name", sorted(CONVERSATIONS))
@pytest.mark.parametrize("with_tools", [True, False])
@pytest.mark.parametrize("gen_prompt", [True, False])
def test_render_matches_official_template_reference(name, with_tools, gen_prompt):
    msgs, t = CONVERSATIONS[name], (TOOLS if with_tools else [])
    assert GLM.render(msgs, t, add_generation_prompt=gen_prompt) == reference(msgs, t, gen_prompt)


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_reasoning_effort_is_passed_to_the_template(effort):
    msgs = CONVERSATIONS["system_user"]
    f = llm.get_format("glm-5.3-flash", reasoning_effort=effort)
    assert f.render(msgs, TOOLS) == reference(msgs, TOOLS, reasoning_effort=effort)
    assert f"<|system|>Reasoning Effort: {effort.capitalize()}" in f.render(msgs, TOOLS)


def test_default_effort_is_max_and_generation_prompt_opens_think():
    p = GLM.render(CONVERSATIONS["system_user"], TOOLS)
    assert p.startswith("[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>\n# Tools")
    assert p.endswith("<|user|>hi<|assistant|><think>")


def test_non_thinking_prompt_closes_the_think_block_like_history_turns():
    msgs = CONVERSATIONS["system_user"]
    assert GLM.render(msgs, [], enable_thinking=False) == GLM.render(msgs, []) + "</think>"
    # ...which is exactly how a finished turn without reasoning re-renders in later prompts
    hist = GLM.render(msgs + [{"role": "assistant", "content": "ok", "reasoning_content": ""}], [], False)
    assert hist.endswith("<|assistant|><think></think>ok")


@pytest.mark.parametrize("name", sorted(CONVERSATIONS))
def test_unsloth_gguf_template_renders_identically(name):
    """llama-server (embedded Unsloth template) and our official-template render must agree on our shapes."""
    msgs = CONVERSATIONS[name]
    assert (GLM.render(msgs, TOOLS, template=llm.GLM_GGUF_TEMPLATE_PATH) == GLM.render(msgs, TOOLS))


def test_template_pins():
    import hashlib
    assert hashlib.sha256(llm.GLM_TEMPLATE_PATH.read_bytes()).hexdigest() == \
        "2614a3c896d53e265f564c23701173c0049f985dacfbae79fc4bcd9f5ba083f4"     # zai-org/GLM-5.3-Flash @46a050be
    assert hashlib.sha256(llm.GLM_GGUF_TEMPLATE_PATH.read_bytes()).hexdigest() == \
        "a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724"     # unsloth/GLM-5.3-Flash-GGUF @a38483c8
    assert set(GLM.server_template_digests()) == {GLM.template_digest(),
                                                 "a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724"}


# ---------------------------------------------------------------------------------------------
# parsing

def call(name, **params):
    return f"<tool_call>{name}" + "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>"
                                          for k, v in params.items()) + "</tool_call>"


def test_parse_reasoning_content_and_two_calls_with_typed_arguments():
    gen = ("First read the file.\nThen edit.</think>Let me look."
           + call("bash", cmd="ls /; cat /func.py")
           + call("text_editor", command="insert", path="/func.py", insert_line="3",
                  new_str="x = 1\n\ty = \"é\"", file_text="null"))
    t = GLM.parse_generation(gen, TOOLS)
    assert t.reasoning == "First read the file.\nThen edit." and t.reasoning_closed
    assert t.content == "Let me look."
    assert [c.name for c in t.tool_calls] == ["bash", "text_editor"]
    assert t.tool_calls[0].arguments == {"cmd": "ls /; cat /func.py"}
    edit = t.tool_calls[1].arguments
    assert edit["insert_line"] == 3 and edit["command"] == "insert"   # anyOf[integer,null] -> int
    assert edit["new_str"] == "x = 1\n\ty = \"é\""                     # raw string, not JSON-decoded
    assert edit["file_text"] is None                                  # anyOf[string,null] + literal null
    assert all(c.parse_error is None for c in t.tool_calls)
    assert t.tool_calls[0].raw == call("bash", cmd="ls /; cat /func.py")


def test_parse_string_values_that_look_like_json_stay_strings():
    t = GLM.parse_generation("r</think>" + call("bash", cmd="123") + call("submit", answer='{"a": 1}'), TOOLS)
    assert t.tool_calls[0].arguments == {"cmd": "123"} and t.tool_calls[1].arguments == {"answer": '{"a": 1}'}


def test_parse_reasoning_only_never_closed():
    t = GLM.parse_generation("still thinking about <tool_call>bash", TOOLS)
    assert not t.reasoning_closed and t.content == "" and t.tool_calls == ()
    assert t.reasoning == "still thinking about <tool_call>bash"      # a call inside reasoning is not an action


def test_parse_empty_reasoning_and_plain_answer():
    t = GLM.parse_generation("</think>Just an answer.", TOOLS)
    assert t.reasoning == "" and t.reasoning_closed and t.content == "Just an answer." and t.tool_calls == ()


def test_parse_flags_unterminated_and_malformed_calls():
    t = GLM.parse_generation("r</think>ok<tool_call>bash<arg_key>cmd</arg_key><arg_value>ls", TOOLS)
    assert t.content == "ok" and len(t.tool_calls) == 1 and t.tool_calls[0].parse_error
    t = GLM.parse_generation("r</think><tool_call>bash<arg_key>cmd</arg_key></tool_call>", TOOLS)
    assert t.tool_calls[0].parse_error and t.tool_calls[0].name == "bash"
    t = GLM.parse_generation("r</think><tool_call></tool_call>", TOOLS)
    assert t.tool_calls[0].parse_error


def test_parse_call_without_arguments_is_valid():
    t = GLM.parse_generation("r</think><tool_call>bash</tool_call>", TOOLS)
    assert t.tool_calls[0].name == "bash" and t.tool_calls[0].arguments == {} and not t.tool_calls[0].parse_error


# ---------------------------------------------------------------------------------------------
# round trips: parse -> assistant_message -> render == prompt + generation

MSGS = CONVERSATIONS["system_user"]


def roundtrip(gen: str) -> bool:
    turn = GLM.parse_generation(gen, TOOLS)
    full = GLM.render(MSGS + [GLM.assistant_message(turn)], TOOLS, add_generation_prompt=False)
    return full == GLM.render(MSGS, TOOLS) + gen


@pytest.mark.parametrize("gen", [
    "reasoning</think>" + call("bash", cmd="ls"),
    "reasoning</think>Here is my plan." + call("bash", cmd="ls"),
    "reasoning</think>" + call("bash", cmd="ls") + call("bash", cmd="pwd"),
    "reasoning</think>Just a plain answer.",
    "line one\n\nline two\n</think>" + call("bash", cmd="line1\nline2"),
    "</think>" + call("text_editor", command="view", path="/f.py", view_range="[1, -1]"),
    "r</think>" + call("submit", answer="DONE"),
    "r with  odd   spacing \n</think>ok",
])
def test_canonical_shapes_round_trip_exactly(gen):
    assert roundtrip(gen)


def test_whitespace_the_template_strips_breaks_prefix_consistency():
    """The template does content.strip(); a generation with padding around visible text does not round-trip.
    Recorded (as for Qwen) so the per-call `prefix_consistent` flag is understood, not assumed away."""
    assert not roundtrip("reasoning</think>\n\nHere is my plan." + call("bash", cmd="ls"))
    assert not roundtrip("reasoning</think>ok\n")


class FakeComp:
    def __init__(self, text, last_token, stop_type="eos"):
        self.text, self.tokens, self.stop_type = text, [5, 6, last_token], stop_type


def test_expected_next_prefix_appends_the_end_token_the_server_hides():
    p = GLM.render(MSGS, TOOLS)
    gen = "r</think>" + call("bash", cmd="ls")
    nxt = GLM.render(MSGS + [GLM.assistant_message(GLM.parse_generation(gen, TOOLS)),
                             {"role": "tool", "content": "out"}], TOOLS)
    assert nxt.startswith(GLM.expected_next_prefix(p, FakeComp(gen, 154829)))        # <|observation|>
    # a tool call turn that stopped on <|user|> or <|endoftext|> would not match the next render
    assert not nxt.startswith(GLM.expected_next_prefix(p, FakeComp(gen, 154827)))
    assert not nxt.startswith(GLM.expected_next_prefix(p, FakeComp(gen, 154820)))
    plain = "r</think>done"
    nxt = GLM.render(MSGS + [GLM.assistant_message(GLM.parse_generation(plain, TOOLS)),
                             {"role": "user", "content": "go on"}], TOOLS)
    assert nxt.startswith(GLM.expected_next_prefix(p, FakeComp(plain, 154827)))      # <|user|>
    # token limit: no end token
    assert GLM.expected_next_prefix(p, FakeComp("abc", 7, "limit")) == p + "abc"


def test_stop_ids_and_registry():
    assert GLM.stop_token_ids == (154820, 154827, 154829) == llm.GLM_STOP_TOKEN_IDS
    assert llm.GLM_BOUNDARY == {154820: "<|endoftext|>", 154827: "<|user|>", 154829: "<|observation|>"}
    assert llm.get_format().name == "qwen3.6"
    with pytest.raises(ValueError):
        llm.get_format("gpt-oss")
    d = GLM.describe()
    assert d["name"] == "glm-5.3-flash" and d["reasoning_effort"].startswith("max")


# ---------------------------------------------------------------------------------------------
# Qwen is untouched

def test_qwen_format_is_the_module_level_functions():
    q = llm.get_format("qwen3.6")
    msgs = CONVERSATIONS["tool_loop"]
    for think in (True, False):
        assert q.render(msgs, TOOLS, enable_thinking=think) == llm.render(msgs, TOOLS, enable_thinking=think)
    gen = "r\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>"
    assert q.parse_generation(gen, TOOLS) == llm.parse_generation(gen, TOOLS)
    assert q.parse_generation(gen, TOOLS).reasoning_raw is None
    assert q.assistant_message(q.parse_generation(gen, TOOLS)) == llm.assistant_message(llm.parse_generation(gen, TOOLS))
    assert q.template_digest() == llm.template_digest() and q.server_template_digests() == (llm.template_digest(),)
    assert q.stop_token_ids == llm.QWEN_STOP_TOKEN_IDS
    assert q.expected_next_prefix("P", FakeComp("G", 1)) == "PG<|im_end|>\n"


# ---------------------------------------------------------------------------------------------
# the agent loop with a scripted GLM

class ScriptedGLM:
    def __init__(self, gens):
        self.gens, self.prompts = list(gens), []

    def complete(self, prompt, *, max_tokens, seed, sampling):
        self.prompts.append(prompt)
        text = self.gens.pop(0)
        last = 154829 if "<tool_call>" in text else 154827
        return llm.Completion(text=text, tokens=[1, 2, last], stop_type="eos",
                              prompt_tokens=len(prompt) // 4, cached_tokens=0, seconds=0.0)


@requires_sandbox
def test_scripted_glm_rollout_is_append_only_and_records_the_target_model():
    d = fixture_task("conflicting", "lcbhard_0")
    task = Task(task_id=d["task_id"], variant="conflicting", prompt=d["prompt"], test=d["test"],
                original_test=d["test"], entry_point=d["entry_point"], instruction_prompt=d["instruction_prompt"])
    client = ScriptedGLM(["Look first.</think>" + call("bash", cmd="cat test.py"),
                          "Done.</think>I will submit." + call("submit", answer="DONE")])
    traj = rollout_mod.run_rollout(task, rollout_id="g1", seed=1, client=client, fmt=GLM,
                                   config=rollout_mod.RolloutConfig(max_attempts=1))
    calls = traj.of(tj.MODEL_CALL)
    assert len(calls) == 2 and calls[1].data["prefix_consistent"] is True
    assert traj.meta["target_model"] == "glm-5.3-flash" and traj.meta["template_sha256"] == GLM.template_digest()
    assert client.prompts[0].endswith("<think>") and "<tool_call>bash<arg_key>cmd" in client.prompts[1]
    assert [e.data["name"] for e in traj.of(tj.TOOL_PROPOSED)] == ["bash", "submit"]
    assert json.dumps(traj.of(tj.TOOL_PROPOSED)[0].data["arguments"]) == '{"cmd": "cat test.py"}'


def test_sampling_presets():
    assert llm.get_format("qwen3.6").sampling == rollout_mod.SAMPLING
    assert GLM.sampling["temperature"] == 1.0 and GLM.sampling["top_p"] == 0.95 and GLM.sampling["top_k"] == 0
    assert set(GLM.sampling) == set(rollout_mod.SAMPLING)       # every key has a vLLM mapping
