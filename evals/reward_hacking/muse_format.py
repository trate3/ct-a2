"""Muse Glimmer 30B chat format: prompt rendering, ATEM tool-call parsing, history round-trip.

Wire format (checked against real samples in `data/muse_raw_fixtures.json`). The prompt ends with
`<|start|>assistant`; the generation continues with channel segments, each `<|start|>assistant to=RECIPIENT
<|message|>BODY` (the first one without its `<|start|>assistant`, which the prompt already holds):

  ` to=self<|message|>REASONING<|eom|>`                               private reasoning
  `<|start|>assistant to=TOOL<|message|><atem:function_calls>\\n<atem:invoke name="TOOL">\\n
       <atem:parameter name="p">v</atem:parameter>\\n...</atem:invoke>\\n</atem:function_calls>`   one call
  (further tool segments, each after an `<|eom|>`)                    parallel calls
  ` to=user<|message|>TEXT`                                           the visible answer

The turn ends with the token `<|eot|>` (id 200008), which the server returns as the last token but drops from
the text. `<|eom|>` (id 200007) only separates segments inside ONE completion, so it is NOT a stop token.
String and scalar parameters are raw text; lists and objects are JSON (the template renders them with
`tojson`, booleans as true/false, None as null).

Rendering uses the pinned official template with the same Jinja environment as transformers'
`apply_chat_template` (trim_blocks/lstrip_blocks, `tojson` without ASCII escaping, `strftime_now`), and the
BOS string `<|begin_of_text|>` that every fixture prompt starts with. `parse_generation` / `assistant_message`
are inverses of the template's assistant rendering for every shape the template can produce, so
`render(history + [assistant_message(parse_generation(text))])` reproduces `prompt + text + "<|eot|>"` byte for
byte (`MuseTurn.canonical` says whether that holds for a given turn).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
import re
from dataclasses import dataclass
from datetime import datetime

from evals.reward_hacking import llm
from evals.reward_hacking.llm import ChatFormat, ParsedTurn, ToolCall

MUSE_TEMPLATE_PATH = pathlib.Path(__file__).parent / "data" / "muse-glimmer-30b_chat_template.jinja"
BOS_TOKEN = "<|begin_of_text|>"          # token 200000 (config.json bos_token_id); first thing in every fixture prompt
EOT, EOM, START_ASSISTANT, MESSAGE = "<|eot|>", "<|eom|>", "<|start|>assistant", "<|message|>"
EOT_TOKEN_ID = 200008                    # last token of every stopped fixture sample, absent from the sample text
EOM_TOKEN_ID = 200007                    # separates segments inside one completion (parallel calls): never a stop
STOP_TOKEN_IDS = (EOT_TOKEN_ID,)

_FC_OPEN, _FC_CLOSE, _INVOKE_CLOSE = "<atem:function_calls>", "</atem:function_calls>", "</atem:invoke>"
_INVOKE_OPEN = re.compile(r'<atem:invoke name="([^"]*)">')
_PARAM_OPEN = re.compile(r'<atem:parameter name="([^"]*)">')
# a value runs to the first `</atem:parameter>` that is followed (after whitespace) by the next parameter or the
# end of the invoke, so a value that merely contains the closing tag text is not cut short
_PARAM_VALUE = re.compile(r'(.*?)</atem:parameter>\s*?(?=<atem:parameter name="|</atem:invoke>)', re.S)
_PARAM_VALUE_AT_END = re.compile(r"(.*?)</atem:parameter>\s*\Z", re.S)     # output cut right after a complete parameter
_HEADER = re.compile(r"^ ?(?:to=([^\s<]+))?<\|message\|>")


@dataclass(frozen=True)
class MuseTurn(ParsedTurn):
    """ParsedTurn plus what the Muse template needs to re-render the turn exactly."""
    content_raw: str | None = None            # unstripped visible text
    unparsed: tuple[str, ...] = ()            # generation pieces that fit no channel segment (kept, not dropped)
    canonical: bool = True                    # render(assistant_message(self)) == the generation + <|eot|>


# ---------------------------------------------------------------------------------------------
# rendering

@functools.lru_cache(maxsize=8)
def _compile(template_text: str):
    import jinja2
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env.from_string(template_text)


def _with_tool_names(messages: list[dict]) -> list[dict]:
    """The template titles a tool result `tool NAME` from `name`, else from the `tool_call_id` of the call it
    answers. The agent loop appends bare `{"role": "tool", "content": ...}` messages, so give those the name of
    the call they answer: the k-th nameless tool message after an assistant message answers its k-th call
    (`tool_call_names` when assistant_message recorded them, else the `tool_calls` order)."""
    out, queue = [], []
    for m in messages:
        if m.get("role") == "assistant":
            names = m.get("tool_call_names")
            if names is None:
                names = [(c.get("function") or {}).get("name", "") for c in (m.get("tool_calls") or [])]
            queue = list(names)
        elif m.get("role") == "tool" and not m.get("name") and not m.get("tool_call_id") and queue:
            m = {**m, "name": queue.pop(0)}
        out.append(m)
    return out


def _schemas(tools) -> dict[str, dict]:
    return {(t.get("function") or t)["name"]: ((t.get("function") or t).get("parameters") or {}).get("properties", {})
            for t in tools or []}


class MuseFormat(ChatFormat):
    name = "muse-glimmer-30b"
    reasoning_efforts, default_effort = ("low", "medium", "high", "xhigh"), "high"
    hf_repo, hf_revision = "meta-models/Muse-Glimmer-30B", "cec94fcfd6512f2cfe708138c32533b346ea4f69"
    local_dir, num_hidden_layers, probe_layers = "muse-glimmer-30b", 52, (13, 26, 39)
    stop_token_ids = STOP_TOKEN_IDS
    # model card: temperature 1.0, top_p 0.95, top_k 64 (Qwen used top_k 20)
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0,
                "presence_penalty": 0.0, "repeat_penalty": 1.0}

    def template_path(self):
        return MUSE_TEMPLATE_PATH

    def describe(self):
        return {**super().describe(), "bos_token": BOS_TOKEN, "eot_token_id": EOT_TOKEN_ID}

    def render(self, messages, tools, add_generation_prompt=True, enable_thinking=True, *, template=None):
        """The pinned template as transformers' apply_chat_template renders it. `enable_thinking` has no
        counterpart in this template (reasoning depth is `reasoning_strength`), so it is accepted and ignored."""
        text = (template or self.template_path()).read_text()
        kw = {} if self.reasoning_effort is None else {"reasoning_strength": self.reasoning_effort}
        return _compile(text).render(messages=_with_tool_names(messages), tools=tools or None,
                                     add_generation_prompt=add_generation_prompt, bos_token=BOS_TOKEN, **kw)

    # -----------------------------------------------------------------------------------------
    # parsing

    def parse_generation(self, text, tools):
        """Split a generation (continuing after `<|start|>assistant`) into reasoning, visible text and ALL tool
        calls in order. Parameters are coerced with the tool's JSON schema (strings raw, the rest JSON). Calls
        that do not parse are returned with `parse_error` set, never dropped; text that fits no segment is kept
        in `MuseTurn.unparsed`."""
        gen = text[:-len(EOT)] if text.endswith(EOT) else text      # a server that keeps the stop token in the text
        props = _schemas(tools)
        chunks = gen.split(START_ASSISTANT)
        if len(chunks) > 1 and chunks[0] == "":
            chunks = chunks[1:]
        reasoning, content, calls, unparsed = [], [], [], []
        last_self_open = False
        for ci, chunk in enumerate(chunks):
            last = ci == len(chunks) - 1
            hm = _HEADER.match(chunk)
            if not hm:
                if chunk.strip():
                    unparsed.append(chunk)
                    if "<atem:" in chunk:
                        calls.append(ToolCall(name="", arguments={}, raw=chunk,
                                              parse_error="tool call outside a <|message|> segment"))
                continue
            recipient = hm.group(1) or ""
            body, eom, tail = chunk[hm.end():].partition(EOM)
            if tail.strip():
                unparsed.append(tail)
            if recipient == "self":
                reasoning.append(body)
                last_self_open = last and not eom
            elif _FC_OPEN in body or "<atem:invoke" in body:
                seg_calls, stray = _parse_calls(body, props)
                calls.extend(seg_calls)
                content.extend(s for s in stray if s.strip())
            elif recipient in ("user", ""):
                content.append(body)
            else:   # addressed to a tool but carries no function_calls block
                calls.append(ToolCall(name=recipient, arguments={}, raw=body,
                                      parse_error=f"segment addressed to {recipient!r} has no <atem:function_calls> block"))
        reasoning_raw = "\n\n".join(reasoning)
        content_raw = "\n\n".join(content)
        turn = MuseTurn(reasoning=reasoning_raw.strip(), content=content_raw.strip(), tool_calls=tuple(calls),
                        reasoning_closed=not last_self_open, reasoning_raw=reasoning_raw,
                        content_raw=content_raw, unparsed=tuple(unparsed), canonical=False)
        return dataclasses.replace(turn, canonical=self._is_canonical(turn, gen))

    def _is_canonical(self, turn: MuseTurn, gen: str) -> bool:
        if turn.unparsed or any(c.parse_error for c in turn.tool_calls):
            return False
        try:
            probe = [{"role": "system", "content": "."}]
            head = self.render(probe, [], add_generation_prompt=False)
            full = self.render(probe + [self.assistant_message(turn)], [], add_generation_prompt=False)
        except Exception:
            return False
        return full[len(head):] == START_ASSISTANT + gen + EOT

    # -----------------------------------------------------------------------------------------
    # history

    def assistant_message(self, turn):
        """The message the template re-renders for this turn: `reasoning_content` (raw, the template writes it
        verbatim), `content` (used only when there are no tool calls) and `tool_calls` with dict arguments.
        Calls that failed to parse are not re-rendered; their names go in `tool_call_names` so the tool results
        the loop appends for them still get a `tool NAME` header (render() reads it)."""
        reasoning = turn.reasoning if turn.reasoning_raw is None else turn.reasoning_raw
        content = getattr(turn, "content_raw", None)
        msg = {"role": "assistant", "content": turn.content if content is None else content,
               "reasoning_content": reasoning}
        good = [c for c in turn.tool_calls if c.parse_error is None]
        if good:
            msg["tool_calls"] = [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                 for c in good]
        if len(good) != len(turn.tool_calls):
            msg["tool_call_names"] = [c.name or "unknown" for c in turn.tool_calls]
        return msg

    def expected_next_prefix(self, prompt, comp):
        """prompt + generation, then `<|eot|>` when the turn ended on it (the server drops that token from the
        text; the next render writes it). Stopped at the token limit there is no end token."""
        text = comp.text
        if comp.stop_type == "limit" or text.endswith(EOT):
            end = ""
        elif not comp.tokens or comp.tokens[-1] == EOT_TOKEN_ID:
            end = EOT
        else:
            end = ""
        return prompt + text + end


def _skip_ws(s: str, p: int) -> int:
    while p < len(s) and s[p].isspace():
        p += 1
    return p


def _coerce(value: str, schema: dict | None):
    return llm._glm_coerce(value, schema)       # strings raw, everything else JSON, by the schema's types


def _parse_invoke(body: str, q: int, props: dict):
    """Parse `<atem:invoke name="N">` parameters at body[q:]. -> (name, args, end, error)."""
    m = _INVOKE_OPEN.match(body, q)
    name, p, args, err = m.group(1), m.end(), {}, None
    schema = props.get(name, {})
    while True:
        p = _skip_ws(body, p)
        if body.startswith(_INVOKE_CLOSE, p):
            return name, args, p + len(_INVOKE_CLOSE), err
        pm = _PARAM_OPEN.match(body, p)
        if not pm:
            return name, args, len(body), err or ("unterminated <atem:invoke>" if p >= len(body) else
                                                  "expected <atem:parameter> or </atem:invoke>")
        vm = _PARAM_VALUE.match(body, pm.end()) or _PARAM_VALUE_AT_END.match(body, pm.end())
        if not vm:
            return name, args, len(body), err or f"unterminated <atem:parameter name={pm.group(1)!r}>"
        key = pm.group(1)
        if key in args:
            err = err or f"duplicate parameter {key!r}"
        args[key] = _coerce(vm.group(1), schema.get(key))
        p = vm.end()


def _parse_calls(body: str, props: dict) -> tuple[list[ToolCall], list[str]]:
    """Every `<atem:function_calls>` block in a segment body -> (calls in order, text outside the blocks)."""
    calls: list[ToolCall] = []
    stray: list[str] = []
    pos = 0
    while True:
        i = body.find(_FC_OPEN, pos)
        if i < 0:
            # an invoke without a function_calls wrapper: still an attempted call
            j = body.find("<atem:invoke", pos)
            if j >= 0:
                stray.append(body[pos:j])
                calls.append(ToolCall(name="", arguments={}, raw=body[j:],
                                      parse_error="<atem:invoke> outside an <atem:function_calls> block"))
            else:
                stray.append(body[pos:])
            return calls, stray
        stray.append(body[pos:i])
        p = i + len(_FC_OPEN)
        found: list[tuple[str, dict, int, int, str | None]] = []     # name, args, start, end, error
        closed, block_err = False, None
        while True:
            q = _skip_ws(body, p)
            if body.startswith(_FC_CLOSE, q):
                p, closed = q + len(_FC_CLOSE), True
                break
            if q >= len(body):
                p = q
                break
            if not _INVOKE_OPEN.match(body, q):
                block_err, p = "expected <atem:invoke> or </atem:function_calls>", len(body)
                found.append(("", {}, q, len(body), block_err))
                break
            name, args, end, err = _parse_invoke(body, q, props)
            found.append((name, args, q, end, err))
            p = end
            if err:
                break
        if not closed and not block_err and not any(f[4] for f in found):
            block_err = "unterminated <atem:function_calls> block"
        if not found:
            found.append(("", {}, i, p, "empty <atem:function_calls> block" if closed else block_err))
        for name, args, s, e, err in found:
            err = err or block_err
            raw = body[i:p] if len(found) == 1 else body[s:e]
            calls.append(ToolCall(name=name, arguments=args, raw=raw, parse_error=err))
        pos = p
