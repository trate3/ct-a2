"""Target-model client: template rendering, raw completion, and tool-call parsing.

The agent talks to an OpenAI-style llama.cpp server, but through the raw
`/completion` endpoint rather than the chat endpoint: we render the prompt with the
model's own chat template (pinned in `data/`), and the server returns the exact
generated token ids. That keeps a token-exact record of every model call, which
the activation tap later re-reads by prefilling `prompt + generation` in the same
model (the hidden state at each generated position is then the on-policy state).

The generation after `<think>\n` is split into private reasoning (before
`</think>`) and the visible completion (content and `<tool_call>` blocks). Parsing
follows the Qwen3.6 XML tool-call format the template documents.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import os
import re
import threading
import time
from dataclasses import dataclass, field

import requests

TEMPLATE_PATH = pathlib.Path(__file__).parent / "data" / "qwen3.6_chat_template.jinja"
DEFAULT_URL = "http://127.0.0.1:8088"
THINK_CLOSE = "</think>"


def template_digest() -> str:
    return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()


def render(messages: list[dict], tools: list[dict], add_generation_prompt: bool = True,
           enable_thinking: bool = True, *, template: pathlib.Path | None = None,
           reasoning_effort: str | None = None) -> str:
    """Render the pinned chat template exactly as the llama.cpp server applies it.

    The server passes `preserve_thinking=true` (checked against `/apply-template`), so
    earlier turns keep their reasoning even after a new user message (e.g. failed-test
    feedback). The context is therefore append-only across the whole rollout.
    """
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda v, **_: json.dumps(v, ensure_ascii=False)
    extra = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
    return env.from_string((template or TEMPLATE_PATH).read_text()).render(
        messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
        preserve_thinking=True, enable_thinking=enable_thinking, **extra)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    raw: str                 # the exact <tool_call>...</tool_call> text generated
    parse_error: str | None = None


@dataclass(frozen=True)
class ParsedTurn:
    reasoning: str           # private reasoning; the whole generation if </think> never closed
    content: str             # visible text outside tool calls
    tool_calls: tuple[ToolCall, ...]
    reasoning_closed: bool = True   # False: generation stopped inside the reasoning channel
    reasoning_raw: str | None = None   # unstripped reasoning, for formats whose template re-renders it verbatim


_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(value: str, schema: dict | None):
    """Typed parameters come back as text; convert them as the server's parser does."""
    kind = (schema or {}).get("type")
    if kind in (None, "string"):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def parse_generation(text: str, tools: list[dict]) -> ParsedTurn:
    """Split a generation (continuing after `<think>\\n`) into reasoning and visible turn."""
    if THINK_CLOSE in text:
        reasoning, visible = text.split(THINK_CLOSE, 1)
        closed = True
    else:   # stopped (e.g. at the token limit) still reasoning: all of it is private
        reasoning, visible, closed = text, "", False
    props = {t["function"]["name"]: t["function"]["parameters"].get("properties", {})
             for t in tools}
    calls = []
    for m in _CALL.finditer(visible):
        name = m.group(1).strip()
        args = {k.strip(): _coerce(v, props.get(name, {}).get(k.strip()))
                for k, v in _PARAM.findall(m.group(2))}
        calls.append(ToolCall(name=name, arguments=args, raw=m.group(0)))
    content = _CALL.sub("", visible)
    # an unterminated <tool_call> is visible output the model meant as an action
    if "<tool_call>" in content:
        start = content.index("<tool_call>")
        calls.append(ToolCall(name="", arguments={}, raw=content[start:],
                              parse_error="unterminated or malformed <tool_call>"))
        content = content[:start]
    return ParsedTurn(reasoning=reasoning.strip(), content=content.strip(), tool_calls=tuple(calls),
                      reasoning_closed=closed)


def assistant_message(turn: ParsedTurn) -> dict:
    """The chat message the template re-renders for this turn in later prompts."""
    msg = {"role": "assistant", "content": turn.content,
           "reasoning_content": turn.reasoning}
    good = [c for c in turn.tool_calls if c.parse_error is None]
    if good:
        msg["tool_calls"] = [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                             for c in good]
    return msg


# ---------------------------------------------------------------------------------------------
# Per-model chat formats
#
# Everything the agent loop needs to know about the model being monitored lives behind `ChatFormat`:
# how a conversation is rendered into the raw prompt, how the generation (which continues after the
# generation prompt's opening `<think>`) is split into reasoning / visible text / tool calls, how the
# parsed turn is written back as a chat message, which tokens end a turn, and what the next prompt must
# start with for the context to stay append-only. The module-level `render` / `parse_generation` /
# `assistant_message` / `template_digest` above ARE the Qwen3.6 format (unchanged); `QwenFormat` only
# forwards to them, so Qwen prompts, parses and manifests are byte-identical to before.

QWEN38_TEMPLATE_PATH = pathlib.Path(__file__).parent / "data" / "qwen3.8-27b_chat_template.jinja"
GLM_TEMPLATE_PATH = pathlib.Path(__file__).parent / "data" / "glm-5.3-flash_chat_template.jinja"
MUSE_TEMPLATE_PATH = pathlib.Path(__file__).parent / "data" / "muse-glimmer-30b_chat_template.jinja"
# Unsloth's GGUF embeds a minja-friendly variant of the official template; llama-server's /props and
# /apply-template reflect THAT file, so it is the server-side pin (render-parity is tested offline).
GLM_GGUF_TEMPLATE_PATH = (pathlib.Path(__file__).parent / "data"
                          / "glm-5.3-flash_unsloth_gguf_chat_template.jinja")
# GLM-5.3-Flash generation_config.json eos_token_id = [<|endoftext|>, <|user|>, <|observation|>]
GLM_STOP_TOKEN_IDS = (154820, 154827, 154829)
GLM_BOUNDARY = {154820: "<|endoftext|>", 154827: "<|user|>", 154829: "<|observation|>"}


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChatFormat:
    """Per-model prompt/parse behaviour. Subclasses: QwenFormat, GLMFormat."""

    name = ""
    think_close = THINK_CLOSE
    stop_token_ids: tuple[int, ...] = ()
    sampling: dict = {}      # the model card's thinking-mode preset, llama.cpp names (see rollout.SAMPLING)
    reasoning_efforts: tuple[str, ...] = ()   # values the template accepts for `reasoning_effort`
    default_effort = ""                       # what the template does when none is passed
    # official checkpoint (activations come from transformers on it; vLLM serves it) -- deploy/second_machine/pins.env
    hf_repo = ""
    hf_revision = ""
    local_dir = ""                            # <MODELS_DIR>/<local_dir>; also the served model name
    num_hidden_layers = 0
    probe_layers: tuple[int, ...] = ()        # 25/50/75 % depth (hidden_states-tuple convention)

    def __init__(self, reasoning_effort: str | None = None):
        if reasoning_effort is not None and reasoning_effort not in self.reasoning_efforts:
            raise ValueError(f"{self.name}: reasoning_effort must be one of {self.reasoning_efforts or '(none)'}, "
                             f"got {reasoning_effort!r}")
        self.reasoning_effort = reasoning_effort      # None: not passed; the template applies its default

    def template_path(self) -> pathlib.Path:
        raise NotImplementedError

    def template_digest(self) -> str:
        return _sha256_file(self.template_path())

    def server_template_digests(self) -> tuple[str, ...]:
        """sha256 values a llama-server's `/props` chat_template may have (the pin, or an equivalent)."""
        return (self.template_digest(),)

    def render(self, messages, tools, add_generation_prompt=True, enable_thinking=True) -> str:
        raise NotImplementedError

    def parse_generation(self, text, tools) -> ParsedTurn:
        raise NotImplementedError

    def assistant_message(self, turn) -> dict:
        raise NotImplementedError

    def expected_next_prefix(self, prompt: str, comp) -> str:
        """What the next call's rendered prompt must start with, given this call's prompt + Completion."""
        raise NotImplementedError

    def describe(self) -> dict:
        d = {"name": self.name, "template_sha256": self.template_digest(),
             "stop_token_ids": list(self.stop_token_ids)}
        if self.reasoning_efforts:
            d["reasoning_effort"] = self.reasoning_effort or f"{self.default_effort} (template default)"
        if self.hf_repo:
            d["hf_repo"], d["hf_revision"] = self.hf_repo, self.hf_revision
        return d


class QwenFormat(ChatFormat):
    """Qwen3.6-35B-A3B (the pilot's target): the module-level functions, byte for byte."""
    name = "qwen3.6"
    hf_repo, hf_revision = "Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"
    local_dir, num_hidden_layers, probe_layers = "qwen3.6-35b-a3b", 40, (10, 20, 30)
    stop_token_ids = (248046, 248044)    # == QWEN_STOP_TOKEN_IDS (tested)
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,     # == rollout.SAMPLING (tested)
                "presence_penalty": 0.0, "repeat_penalty": 1.0}

    def template_path(self):
        return TEMPLATE_PATH          # looked up at call time: smoke/provenance tools swap it

    def template_digest(self):
        return template_digest()

    def render(self, messages, tools, add_generation_prompt=True, enable_thinking=True):
        return render(messages, tools, add_generation_prompt, enable_thinking)

    def parse_generation(self, text, tools):
        return parse_generation(text, tools)

    def assistant_message(self, turn):
        return assistant_message(turn)

    def expected_next_prefix(self, prompt, comp):
        # the next render must extend this: generation plus the end-of-turn token
        return prompt + comp.text + "<|im_end|>\n"


# GLM-5.3-Flash. Rendered by the official template: `[gMASK]<sop><|system|>Reasoning Effort: Max` first
# (the default effort is "max" when `reasoning_effort` is not passed; recorded, not changed), tools in the
# system turn, then `<|user|>text`, `<|assistant|><think>reasoning</think>text<tool_call>...`, and tool
# results as ONE `<|observation|>` followed by consecutive `<tool_response>text</tool_response>`.
# There is no end-of-turn string: a turn ends with the token that begins the next role (`<|observation|>`
# after tool calls, `<|user|>` otherwise; `<|endoftext|>` also stops), and the template has NO whitespace
# between segments. Thinking is always on: the generation prompt is `<|assistant|><think>`. A history turn
# without reasoning renders `<think></think>`; our non-thinking prompt (enable_thinking=False) closes the
# think block the same way: `<|assistant|><think></think>`. That one is our extension, not the template's.
_GLM_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_GLM_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def _schema_types(schema: dict | None) -> set[str]:
    """JSON types a parameter schema allows (`type`, or the union of `anyOf`/`oneOf` branches)."""
    if not schema:
        return set()
    kind = schema.get("type")
    if isinstance(kind, str):
        return {kind}
    if isinstance(kind, list):
        return set(kind)
    out: set[str] = set()
    for branch in schema.get("anyOf", []) + schema.get("oneOf", []):
        out |= _schema_types(branch)
    return out


def _glm_coerce(value: str, schema: dict | None):
    """GLM's template writes string arguments raw and everything else as JSON; invert that with the schema.

    Unknown schema (e.g. an enum without `type`) stays text. A union containing "string" stays text unless
    the value is exactly `null` and null is allowed."""
    types = _schema_types(schema)
    if not types or types == {"string"}:
        return value
    if "string" in types and not (value.strip() == "null" and "null" in types):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


class GLMFormat(ChatFormat):
    name = "glm-5.3-flash"
    reasoning_efforts, default_effort = ("low", "high", "max"), "max"
    hf_repo, hf_revision = "zai-org/GLM-5.3-Flash", "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
    local_dir, num_hidden_layers, probe_layers = "glm-5.3-flash", 45, (11, 22, 34)
    stop_token_ids = GLM_STOP_TOKEN_IDS
    # generation_config.json: temperature 1.0, top_p 0.95 and no top_k (top_k=0 disables llama.cpp's default 40)
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
                "presence_penalty": 0.0, "repeat_penalty": 1.0}

    def template_path(self):
        return GLM_TEMPLATE_PATH

    def server_template_digests(self):
        return (self.template_digest(), _sha256_file(GLM_GGUF_TEMPLATE_PATH))

    def describe(self):
        return {**super().describe(), "gguf_template_sha256": _sha256_file(GLM_GGUF_TEMPLATE_PATH)}

    def render(self, messages, tools, add_generation_prompt=True, enable_thinking=True, *, template=None):
        import jinja2
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(msg):
            raise jinja2.TemplateError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=kw.get("ensure_ascii", False))
        kw = {} if self.reasoning_effort is None else {"reasoning_effort": self.reasoning_effort}
        text = env.from_string((template or GLM_TEMPLATE_PATH).read_text()).render(
            messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kw)
        if add_generation_prompt and not enable_thinking:
            text += "</think>"
        return text

    def parse_generation(self, text, tools):
        """Split a generation (continuing after `<think>`) into reasoning and visible turn."""
        if THINK_CLOSE in text:
            reasoning, visible = text.split(THINK_CLOSE, 1)
            closed = True
        else:
            reasoning, visible, closed = text, "", False
        props = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
        calls = []
        for m in _GLM_CALL.finditer(visible):
            body = m.group(1)
            head, _, _ = body.partition("<arg_key>")
            name = head.strip()
            args = {k.strip(): _glm_coerce(v, props.get(name, {}).get(k.strip()))
                    for k, v in _GLM_ARG.findall(body)}
            if not name or (body.count("<arg_key>") != len(args)):
                calls.append(ToolCall(name=name, arguments=args, raw=m.group(0),
                                      parse_error="malformed <tool_call> (name or <arg_key>/<arg_value> pairs)"))
            else:
                calls.append(ToolCall(name=name, arguments=args, raw=m.group(0)))
        content = _GLM_CALL.sub("", visible)
        if "<tool_call>" in content:   # unterminated: visible output the model meant as an action
            start = content.index("<tool_call>")
            calls.append(ToolCall(name="", arguments={}, raw=content[start:],
                                  parse_error="unterminated or malformed <tool_call>"))
            content = content[:start]
        return ParsedTurn(reasoning=reasoning.strip(), content=content.strip(), tool_calls=tuple(calls),
                          reasoning_closed=closed, reasoning_raw=reasoning)

    def assistant_message(self, turn):
        # the template renders reasoning_content verbatim (no strip), so keep the raw text
        msg = {"role": "assistant", "content": turn.content,
               "reasoning_content": turn.reasoning if turn.reasoning_raw is None else turn.reasoning_raw}
        good = [c for c in turn.tool_calls if c.parse_error is None]
        if good:
            msg["tool_calls"] = [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                 for c in good]
        return msg

    def expected_next_prefix(self, prompt, comp):
        # prompt + generation text, then the token that ended the turn (the server drops its text from
        # `content`): the next render starts the next role with <|observation|> / <|user|>. Stopped at the
        # token limit there is no end token, and nothing follows.
        last = comp.tokens[-1] if comp.tokens else None
        boundary = GLM_BOUNDARY.get(last, "") if comp.stop_type != "limit" else ""
        return prompt + comp.text + boundary


# Qwen3.8-27B (Qwen/Qwen3.8-27B, dense, Apache-2.0). Same `<think>\n` generation prompt, `<|im_end|>` turn end
# and `<tool_call><function=..><parameter=..>` XML as Qwen3.6, so it reuses Qwen3.6's parser unchanged; only the
# pinned (official) template differs. Differences from our Qwen3.6 pin (the Unsloth-fixed template):
#   * `reasoning_effort` (xhigh | medium | low; default xhigh): with thinking on, the system turn starts with
#     "Reasoning effort is set to xhigh. Please think carefully..." (low: a "keep it brief" line; medium: nothing).
#     A system turn is therefore ALWAYS emitted (with tools: before "# Tools"). We pass nothing: default xhigh.
#   * preserve_thinking defaults to true (we pass it anyway), so history keeps all reasoning as for Qwen3.6.
#   * tool-call arguments: bool/None/int render with tojson (`true`/`null`), not Python str (`True`/`None`);
#     our tool schemas have ints/lists/strings only, where both agree.
#   * no developer-role merge; raises if there is no user turn or a system message is not first.
class Qwen38Format(QwenFormat):
    name = "qwen3.8-27b"
    reasoning_efforts, default_effort = ("xhigh", "medium", "low"), "xhigh"
    hf_repo, hf_revision = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    local_dir, num_hidden_layers, probe_layers = "qwen3.8-27b", 64, (16, 32, 48)
    # generation_config.json: eos_token_id [<|im_end|>, <|endoftext|>] (ids verified in tokenizer_config.json)

    def template_path(self):
        return QWEN38_TEMPLATE_PATH

    def template_digest(self):
        return _sha256_file(QWEN38_TEMPLATE_PATH)

    def render(self, messages, tools, add_generation_prompt=True, enable_thinking=True, *, template=None):
        return render(messages, tools, add_generation_prompt, enable_thinking,
                      template=template or QWEN38_TEMPLATE_PATH, reasoning_effort=self.reasoning_effort)


class MuseFormat(ChatFormat):
    """Muse Glimmer 30B. Registry entry for the pinned checkpoint, sampling and probe layers
    (activation extraction and the parity check use these); the prompt rendering and the generation parser
    are written separately and raise until then."""
    name = "muse-glimmer-30b"
    reasoning_efforts, default_effort = ("low", "medium", "high", "xhigh"), "high"
    hf_repo, hf_revision = "meta-models/Muse-Glimmer-30B", "cec94fcfd6512f2cfe708138c32533b346ea4f69"
    local_dir, num_hidden_layers, probe_layers = "muse-glimmer-30b", 52, (13, 26, 39)
    # model card: temperature 1.0, top_p 0.95, top_k 64 (Qwen used top_k 20)
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0,
                "presence_penalty": 0.0, "repeat_penalty": 1.0}

    def template_path(self):
        return MUSE_TEMPLATE_PATH

    def render(self, messages, tools, add_generation_prompt=True, enable_thinking=True):
        raise NotImplementedError("MuseFormat.render is not written yet (a separate piece of work)")

    def parse_generation(self, text, tools):
        raise NotImplementedError("MuseFormat.parse_generation is not written yet (a separate piece of work)")

    def assistant_message(self, turn):
        raise NotImplementedError("MuseFormat.assistant_message is not written yet (a separate piece of work)")

    def expected_next_prefix(self, prompt, comp):
        raise NotImplementedError("MuseFormat.expected_next_prefix is not written yet (a separate piece of work)")


FORMATS = {"qwen3.6": QwenFormat, "qwen3.8-27b": Qwen38Format, "glm-5.3-flash": GLMFormat,
           "muse-glimmer-30b": MuseFormat}
DEFAULT_FORMAT = "qwen3.6"


def get_format(name: str | None = None, **kw) -> ChatFormat:
    name = name or DEFAULT_FORMAT
    if name not in FORMATS:
        raise ValueError(f"unknown target model {name!r}; expected one of {sorted(FORMATS)}")
    if name == "muse-glimmer-30b":     # the working format lives in muse_format.py (it imports this module: no top-level import here)
        from evals.reward_hacking import muse_format
        return muse_format.MuseFormat(**kw)
    return FORMATS[name](**kw)


@dataclass
class Completion:
    text: str
    tokens: list[int]
    stop_type: str           # "eos" | "limit" | "word" | ...
    prompt_tokens: int
    cached_tokens: int
    seconds: float
    timings: dict = field(default_factory=dict)


ENGINES = ("llamacpp", "vllm")
# llama.cpp (unified KV pool) answers HTTP 500 with this message when the pool is exhausted and
# aborts EVERY in-flight request (server-context.cpp: n_batch == 1 && ret == 1). The call produced
# no tokens, so retrying the identical request is safe.
KV_EXHAUSTED = "Context size has been exceeded"


class KVBudget:
    """Client-side admission control for a shared KV pool (llama.cpp `--kv-unified`).

    Every request reserves `estimated prompt tokens + max_tokens` before it is sent and releases
    that when it returns, so the sum over in-flight requests never exceeds the pool and the server
    never reaches its (non-graceful) "Context size has been exceeded" path. One request larger than
    the whole pool is still admitted when it is alone. Thread-safe; shared by all rollout workers
    because they share one client.
    """

    def __init__(self, pool_tokens: int, chars_per_token: float = 3.0):
        self.pool = int(pool_tokens)
        self.chars_per_token = chars_per_token
        self.used = 0
        self.peak = 0
        self.waits = 0
        self._cv = threading.Condition()

    def cost(self, prompt: str, max_tokens: int) -> int:
        return min(self.pool, int(len(prompt) / self.chars_per_token) + 1 + int(max_tokens))

    def acquire(self, n: int) -> None:
        with self._cv:
            if self.used and self.used + n > self.pool:
                self.waits += 1
                while self.used and self.used + n > self.pool:
                    self._cv.wait()
            self.used += n
            self.peak = max(self.peak, self.used)

    def release(self, n: int) -> None:
        with self._cv:
            self.used -= n
            self._cv.notify_all()


@dataclass
class Completion:
    text: str
    tokens: list[int]
    stop_type: str           # "eos" | "limit" | "word" | ...
    prompt_tokens: int
    cached_tokens: int
    seconds: float
    timings: dict = field(default_factory=dict)


class CompletionClient:
    """Minimal client for a llama.cpp server's `/completion` endpoint."""

    engine = "llamacpp"

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 1800.0, retries: int = 5,
                 kv_pool_tokens: int | None = None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.kv_budget = KVBudget(kv_pool_tokens) if kv_pool_tokens else None

    def _retryable_http(self, r) -> bool:
        if r.status_code == 503:           # model (re)loading
            return True
        return r.status_code == 500 and KV_EXHAUSTED in (r.text or "")

    def _post(self, path: str, body: dict) -> dict:
        delay = 5.0
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(self.url + path, json=body, timeout=self.timeout)
                if attempt < self.retries and self._retryable_http(r):
                    raise requests.ConnectionError(f"HTTP {r.status_code}, retrying")
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                if attempt == self.retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
        raise AssertionError("unreachable")

    def props(self) -> dict:
        r = requests.get(self.url + "/props", timeout=60)
        r.raise_for_status()
        return r.json()

    def identity(self) -> dict:
        """Engine + model identity recorded in manifests."""
        props = self.props()
        d = {k: props.get(k) for k in ("model_path", "model_alias", "model_ftype", "build_info", "total_slots")}
        d["engine"] = self.engine
        d["engine_version"] = props.get("build_info")
        d["n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
        d["server_template_sha256"] = hashlib.sha256(props.get("chat_template", "").encode()).hexdigest()
        return d

    def tokenize(self, text: str) -> list[int]:
        return self._post("/tokenize", {"content": text, "add_special": False})["tokens"]

    def complete(self, prompt: str, *, max_tokens: int, seed: int, sampling: dict,
                 extra: dict | None = None) -> Completion:
        t0 = time.perf_counter()
        n = self.kv_budget.cost(prompt, max_tokens) if self.kv_budget else 0
        if self.kv_budget:
            self.kv_budget.acquire(n)
        try:
            d = self._post("/completion", {
                "prompt": prompt, "n_predict": -1 if max_tokens is None else max_tokens, "seed": seed,
                "return_tokens": True,
                "cache_prompt": True, "stream": False, **sampling, **(extra or {})})
        finally:
            if self.kv_budget:
                self.kv_budget.release(n)
        timings = d.get("timings", {})
        return Completion(text=d["content"], tokens=list(d.get("tokens", [])),
                          stop_type=d.get("stop_type", ""),
                          prompt_tokens=int(d.get("tokens_evaluated", 0)),
                          cached_tokens=int(timings.get("cache_n", 0)),
                          seconds=time.perf_counter() - t0, timings=timings)


# ---------------------------------------------------------------------------------------------
# vLLM

# llama.cpp-style sampling names (what rollout.SAMPLING / MonitorConfig use) -> vLLM request fields.
# Anything else raises: a silently dropped sampling parameter would change the study.
_VLLM_SAMPLING = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
                  "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
                  "repeat_penalty": "repetition_penalty", "repetition_penalty": "repetition_penalty"}
# Qwen3.6 generation_config.json: eos_token_id = [<|im_end|>, <|endoftext|>]. config.json's text
# eos is <|endoftext|> only, so the stop ids are sent explicitly instead of trusting the server.
QWEN_STOP_TOKEN_IDS = (248046, 248044)
_VLLM_FINISH = {"stop": "eos", "length": "limit", "abort": "abort"}


def vllm_sampling(sampling: dict) -> dict:
    out = {}
    for k, v in sampling.items():
        if k not in _VLLM_SAMPLING:
            raise ValueError(f"sampling parameter {k!r} has no vLLM mapping")
        out[_VLLM_SAMPLING[k]] = v
    return out


class VLLMCompletionClient(CompletionClient):
    """Client for vLLM's OpenAI-compatible `/v1/completions` with OUR rendered prompt text.

    Verified against vLLM v0.31.0 (`vllm/entrypoints/openai/completion/protocol.py`):
    request `return_token_ids` -> `choices[0].token_ids` (generated ids, INCLUDING the stop token,
    like llama.cpp's `tokens`) and `choices[0].prompt_token_ids`; `skip_special_tokens` /
    `spaces_between_special_tokens` / `stop_token_ids` / `seed` / `top_k` / `min_p` /
    `repetition_penalty` are request fields; the stop token is excluded from `text`;
    `finish_reason` is "stop" | "length"; cached tokens appear in
    `usage.prompt_tokens_details.cached_tokens` only if the server runs `--enable-prompt-tokens-details`.
    `token_mode="logprobs"` is the fallback if `token_ids` is unavailable: `logprobs=0` +
    `return_tokens_as_token_ids` gives "token_id:N" strings for the generated tokens.
    """

    engine = "vllm"

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 1800.0, retries: int = 5,
                 model: str | None = None, stop_token_ids=QWEN_STOP_TOKEN_IDS, token_mode: str = "token_ids",
                 fmt: "ChatFormat | None" = None):
        super().__init__(url, timeout, retries)
        if fmt is not None:          # the target model's own turn-end ids (GLM: <|endoftext|>/<|user|>/<|observation|>)
            stop_token_ids = fmt.stop_token_ids
        assert token_mode in ("token_ids", "logprobs")
        self.model = model
        self.stop_token_ids = list(stop_token_ids)
        self.token_mode = token_mode
        self._max_model_len: int | None = None

    def _get(self, path: str) -> dict:
        r = requests.get(self.url + path, timeout=60)
        r.raise_for_status()
        return r.json()

    def served_model(self) -> dict:
        data = self._get("/v1/models")["data"]
        if self.model:
            data = [m for m in data if m["id"] == self.model] or data
        return data[0]

    def props(self) -> dict:
        m = self.served_model()
        return {"model_path": m.get("root"), "model_alias": m.get("id"), "model_ftype": None,
                "build_info": f"vllm {self._get('/version')['version']}", "total_slots": None,
                "default_generation_settings": {"n_ctx": m.get("max_model_len")}}

    def identity(self) -> dict:
        m = self.served_model()
        ver = self._get("/version")["version"]
        return {"engine": "vllm", "engine_version": ver, "build_info": f"vllm {ver}",
                "model_path": m.get("root"), "model_alias": m.get("id"), "model_ftype": None,
                "total_slots": None, "n_ctx": m.get("max_model_len"),
                "server_template_sha256": None,      # prompts are rendered client-side
                "stop_token_ids": self.stop_token_ids}

    def tokenize(self, text: str) -> list[int]:
        body = {"prompt": text, "add_special_tokens": False}
        if self.model:
            body["model"] = self.model
        return self._post("/tokenize", body)["tokens"]

    def _body(self, prompt, max_tokens, seed, sampling) -> dict:
        if self.model is None:
            self.model = self.served_model()["id"]
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "seed": seed,
                "stream": False, "n": 1, "add_special_tokens": False,
                "skip_special_tokens": False, "spaces_between_special_tokens": False,
                "stop_token_ids": self.stop_token_ids, **vllm_sampling(sampling)}
        if self.token_mode == "token_ids":
            body["return_token_ids"] = True
        else:
            body.update(logprobs=0, return_tokens_as_token_ids=True)
        return body

    def _tokens(self, choice: dict) -> list[int]:
        if self.token_mode == "token_ids":
            ids = choice.get("token_ids")
            if ids is None:
                raise RuntimeError("vLLM response has no choices[0].token_ids; is return_token_ids supported? "
                                   "retry with token_mode='logprobs'")
            return [int(t) for t in ids]
        toks = (choice.get("logprobs") or {}).get("tokens")
        if toks is None or not all(isinstance(t, str) and t.startswith("token_id:") for t in toks):
            raise RuntimeError("vLLM logprobs response lacks 'token_id:N' tokens")
        return [int(t.split(":", 1)[1]) for t in toks]

    def uncapped_max_tokens(self, prompt) -> int:
        """`max_tokens` for "no cap": the server context minus the prompt. vLLM's /v1/completions does NOT do
        this for a missing/null max_tokens; it applies the OpenAI default of 16 tokens (found 2026-10-21)."""
        if self._max_model_len is None:
            self._max_model_len = int(self.served_model()["max_model_len"])
        n = len(prompt) if isinstance(prompt, list) else len(self.tokenize(prompt))
        return max(1, self._max_model_len - n)

    def complete(self, prompt: str, *, max_tokens: int | None, seed: int, sampling: dict,
                 extra: dict | None = None) -> Completion:
        t0 = time.perf_counter()
        if max_tokens is None:
            max_tokens = self.uncapped_max_tokens(prompt)
        d = self._post("/v1/completions", {**self._body(prompt, max_tokens, seed, sampling), **(extra or {})})
        choice = d["choices"][0]
        usage = d.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or len(choice.get("prompt_token_ids") or []))
        finish = choice.get("finish_reason")
        stop_reason = choice.get("stop_reason")
        timings = {"engine": "vllm", "usage": usage, "stop_reason": stop_reason, "finish_reason": finish}
        tokens = self._tokens(choice)
        # vLLM (v1 detokenizer): a stop TOKEN ends the turn, is kept in token_ids and dropped from text, and
        # `stop_reason` is its id. The format's expected_next_prefix reads tokens[-1], so make sure it is there.
        if finish == "stop" and isinstance(stop_reason, int) and stop_reason in self.stop_token_ids:
            timings["stop_token_id"] = stop_reason
            if not tokens:
                tokens = [stop_reason]
            elif tokens[-1] != stop_reason:
                timings["stop_token_mismatch"] = tokens[-1]      # surfaces as a prefix-check failure downstream
        if d.get("metrics"):
            timings["metrics"] = d["metrics"]
        return Completion(text=choice["text"], tokens=tokens,
                          stop_type=_VLLM_FINISH.get(finish, finish or ""),
                          prompt_tokens=prompt_tokens, cached_tokens=int(details.get("cached_tokens") or 0),
                          seconds=time.perf_counter() - t0, timings=timings)


def make_client(engine: str, url: str = DEFAULT_URL, **kw) -> CompletionClient:
    """`engine`: "llamacpp" | "vllm". llama.cpp also honours CT_RH_KV_POOL_TOKENS (shared KV pool size)."""
    if engine == "llamacpp":
        if "kv_pool_tokens" not in kw and os.environ.get("CT_RH_KV_POOL_TOKENS"):
            kw["kv_pool_tokens"] = int(os.environ["CT_RH_KV_POOL_TOKENS"])
        return CompletionClient(url, **kw)
    if engine == "vllm":
        return VLLMCompletionClient(url, **kw)
    raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
