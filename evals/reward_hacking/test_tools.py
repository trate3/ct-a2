"""Tool specs and executors against the inspect_ai 0.3.137 reference run (data/inspect_golden.json)."""

from __future__ import annotations

import os
import time

import pytest

from evals.reward_hacking import tools
from evals.reward_hacking.conftest import (backend_normalize, fixture_task, golden, replay_basic_agent,
                                           requires_sandbox)


def _scenario(name: str) -> dict:
    return next(s for s in golden()["scenarios"] if s["name"] == name)


def _compare(sc: dict, sb) -> tuple:
    task = fixture_task(sc["split"], sc["task_id"])
    msgs, final, completion = replay_basic_agent(sb, task, sc["turns"], sc["max_attempts"])
    be = sb.backend
    norm = lambda m: {k: (backend_normalize(m.get(k), be) if k == "text" else m.get(k)) for k in ("role", "text", "error")}  # noqa: E731
    want = [norm(m) for m in sc["messages"]]
    got = [norm(m) for m in msgs]
    diffs = [f"message {i}:\n  want {w!r}\n  got  {g!r}" for i, (w, g) in enumerate(zip(want, got)) if g != w]
    assert not diffs, f"{len(diffs)} message(s) differ:\n" + "\n".join(diffs)
    assert len(got) == len(want)
    return final, completion


def test_tool_specs_match_inspect():
    assert tools.TOOL_SPECS == golden()["tool_specs"]
    assert [s["function"]["name"] for s in tools.TOOL_SPECS] == ["bash", "python", "text_editor", "think", "submit"]


@requires_sandbox
def test_golden_tool_scenario(sandbox):
    """Every bash/python/text_editor/think/submit message, error and the hack's pass match inspect."""
    sc = _scenario("tools_and_hack")
    final, completion = _compare(sc, sandbox)
    assert completion == sc["completion"]
    assert (final.value, final.explanation, final.answer) == (
        sc["final_score"]["value"], sc["final_score"]["explanation"], sc["final_score"]["answer"])


@requires_sandbox
@pytest.mark.skipif(not os.environ.get("RH_SLOW"), reason="2.5 min (inspect's 60 s tool timeout, retried); set RH_SLOW=1")
def test_golden_timeout_scenario(sandbox):
    sc = _scenario("bash_timeout")
    _compare(sc, sandbox)


@requires_sandbox
def test_bash_python_output_format(sandbox):
    o = tools.execute_tool(sandbox, "bash", {"cmd": "echo out; echo err >&2; exit 7"})
    assert o.output_text == "err\n\nout\n" and o.error is None and o.exec["returncode"] == 7
    o = tools.execute_tool(sandbox, "python", {"code": "import sys; print(sys.stdin.read() == '')"})
    assert o.output_text == "True\n"
    o = tools.execute_tool(sandbox, "python", {"code": "raise SystemExit(3)"})
    assert o.output_text == "" and o.exec["returncode"] == 3


@requires_sandbox
def test_tool_timeout(sandbox, monkeypatch):
    monkeypatch.setattr(tools, "TOOL_TIMEOUT", 2)
    t0 = time.perf_counter()
    o = tools.execute_tool(sandbox, "bash", {"cmd": "echo started >> /tmp/n; sleep 20; echo done"})
    assert time.perf_counter() - t0 > 4.5
    assert (o.error_type, o.error) == ("timeout", "Command timed out before completing.")
    assert o.output_text == "Error: Command timed out before completing." and o.result_text == ""
    assert sandbox.read_file("/tmp/n") == "started\n" * 3     # inspect re-runs it twice
    o = tools.execute_tool(sandbox, "python", {"code": "while True: pass"})
    assert o.error_type == "timeout"


@requires_sandbox
def test_output_limits(sandbox):
    o = tools.execute_tool(sandbox, "bash", {"cmd": "head -c 11000000 /dev/zero | tr '\\0' a"})
    assert o.output_text == "Error: The tool exceeded its output limit of 10 MiB." and o.error_type == "limit"
    assert o.result_text.startswith("\nThe output of your call to bash was too long")
    o = tools.execute_tool(sandbox, "bash", {"cmd": "head -c 20000 /dev/zero | tr '\\0' b"})
    assert o.truncated == (20000, 16384)
    assert o.output_text == ("\nThe output of your call to bash was too long to be displayed.\n"
                             "Here is a truncated version:\n<START_TOOL_OUTPUT>\n" + "b" * 16384 + "\n<END_TOOL_OUTPUT>\n")


@requires_sandbox
def test_text_editor_roundtrip(sandbox):
    ed = lambda **a: tools.execute_tool(sandbox, "text_editor", a)   # noqa: E731
    assert ed(command="create", path="x.py", file_text="a\nb\n").output_text == "File created successfully at: /x.py"
    assert sandbox.read_file("x.py") == "a\nb\n"
    assert ed(command="str_replace", path="x.py", old_str="b", new_str="B").error is None
    assert sandbox.read_file("x.py") == "a\nB\n"
    assert ed(command="insert", path="x.py", insert_line=0, new_str="top").error is None
    assert sandbox.read_file("x.py") == "top\na\nB\n"
    ed(command="undo_edit", path="x.py")
    assert sandbox.read_file("x.py") == "a\nB\n"
    o = ed(command="str_replace", path="x.py", old_str="B")          # no new_str: deletes
    assert o.error is None and sandbox.read_file("x.py") == "a\n\n"
    o = ed(command="view", path="nope.py")
    assert o.output_text == "Error: The path /nope.py does not exist. Please provide a valid path."
    # helper re-injected if the agent deletes it, as inspect's detector does
    sandbox.exec(["rm", "-f", "/opt/inspect-sandbox-tools"])
    assert ed(command="view", path="x.py").error is None


def test_think_and_submit_need_no_sandbox():
    o = tools.execute_tool(None, "think", {"thought": "hmm"})
    assert (o.output_text, o.error, o.infra_error) == ("", None, None)
    s = tools.submit_output({"answer": "DONE"})
    assert s.output_text == "DONE" and s.is_submission and s.submitted_answer == "DONE"
    assert not tools.submit_output({"answer": ""}).is_submission     # basic_agent 0.3.137: `if answer:`
    bad = tools.submit_output({})
    assert not bad.is_submission and bad.output_text.startswith("Error: Found 1 validation errors")
    big = tools.submit_output({"answer": "y" * 20000})
    assert big.truncated and big.submitted_answer.startswith("\nThe output of your call to submit")
    o = tools.execute_tool(None, "bash", {"cmd": "true"})
    assert o.infra_error and o.output_text == ""


def test_argument_parsing_errors():
    o = tools.execute_tool(None, "nope", {})
    assert o.output_text == "Error: Tool nope not found" and o.error_type == "parsing"
    args, err = tools.parse_tool_arguments("bash", '{"cmd": "ls"')
    assert args == {} and err.startswith('Error parsing the following tool call arguments:\n\n{"cmd": "ls"\n\nError details: ')
    assert tools.execute_tool(None, "bash", args, parse_error=err).output_text == "Error: " + err
    assert tools.parse_tool_arguments("bash", "ls -la") == ({"cmd": "ls -la"}, None)
    m = tools.malformed_call_output("<tool_call>junk", "unterminated")
    assert m.output_text == ("Error: Error parsing the following tool call arguments:\n\n<tool_call>junk"
                             "\n\nError details: unterminated")
