"""Activation capture for the white-box tap: sequence reconstruction, alignment, extraction.

Conventions (frozen here; `tap.py` and `probe.py` import them)
---------------------------------------------------------------
Layer index. `layer = L` means the **hidden_states-tuple index** used by HF
`output_hidden_states=True` and by `util/model_backend`: hidden_states[0] is the embedding
output, hidden_states[L] is the output of decoder block L-1 (0-based module index), i.e.
"after L blocks". We capture it with a forward hook on `layers[L-1]`, so it is the raw
residual stream *before* any final norm. Valid L is 1..num_hidden_layers-1 (HF's last tuple
entry has the final norm applied, which a block hook cannot reproduce; it is refused rather
than silently differing). Tests pin the equality with `output_hidden_states`.

Alignment. The "predictive" convention of `evals/financial/extraction.py`: the state recorded
for a generated token is the state that *produced* it. With P = len(tokenize(prompt)) and
generated ids g_0..g_{n-1}, token j sits at absolute position P+j and its state is the
hidden state at position P+j-1 (the prompt's last token for j = 0). The state after the last
generated token is never used (nothing was predicted from it).

What was actually fed to the model. llama.cpp received each call's prompt as TEXT, tokenized
it itself, and then generated token ids. The true sequence for call k is therefore
`tokenize(prompt_k) + gen_k_ids`, and gen_k_ids need not equal `tokenize(generation_k)`.
Every call is teacher-forced on exactly that sequence. States are reused across calls (KV
cache plus the recurrent state of the linear-attention layers) only when the previous call's
full sequence is a token-prefix of the new one; otherwise the call is recomputed from
scratch (a recurrent state cannot be cropped to a shorter prefix).

Channels of generated tokens (from token ids, not text): REASONING until and including the
`</think>` token (the prompt ends with `<think>\\n` when thinking is enabled; if it does not,
the generation starts in VISIBLE); TOOL_CALL from `<tool_call>` through `</tool_call>`
inclusive; VISIBLE otherwise; END for the stop token (`<|im_end|>`), which the model did emit
but which is not content. The default tap mask excludes END.

Muse Glimmer 30B (ATEM wire format; `muse_marks`, `label_channels_atem`). A generation continues a prompt
that ends with `<|start|>assistant` and is a run of segments ` to=RECIPIENT<|message|>BODY<|eom|>` ending
with `<|eot|>`. The channel of a segment is set by its recipient: `self` = REASONING, `user` = VISIBLE, any
other recipient (a tool name, possibly `name.function`) = TOOL_CALL. The same rule as for Qwen decides which
marker tokens belong to a channel: a marker belongs to the channel it opens or closes (Qwen: `</think>`
is REASONING, `<tool_call>` and `</tool_call>` are TOOL_CALL). So every token of a segment from its first
token through its `<|eom|>` carries the segment's channel: `<|start|>assistant` of a later segment, the
recipient text, `<|message|>`, the body, `<|eom|>`. Those header tokens are labelled once `<|message|>`
(which ends the recipient) has been seen. The first segment's `<|start|>assistant` lies in the prompt and is
not labelled, as Qwen's `<think>`. The final `<|eot|>` (the stop token) is END, whatever the channel.
Labels are computed from token ids; the tokenizer only decodes the recipient text.

Memory. Hooks capture only the requested layers; the prompt is prefilled in chunks
(`chunk_tokens`) with a cache, logits are computed for one position only, and only the rows
that are generated-token states are kept. One `.npz` per rollout, written atomically.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import re
from collections.abc import Sequence

import numpy as np

from evals.reward_hacking import trajectory as tj

REASONING, TOOL_CALL, VISIBLE, END = 0, 1, 2, 3
CHANNEL_NAMES = {REASONING: "reasoning", TOOL_CALL: "tool_call", VISIBLE: "visible", END: "end"}
CHANNEL_CODES = {v: k for k, v in CHANNEL_NAMES.items()}
DEFAULT_CHANNELS = ("reasoning", "tool_call", "visible")

LAYER_CONVENTION = ("hidden_states tuple index L (0 = embeddings, L = output of decoder block L-1, "
                    "pre final norm), captured by forward hook on layers[L-1]")
ALIGNMENT = ("predictive: state of generated token j is hidden state at position P+j-1 "
             "(state that produced it), P = len(tokenize(prompt))")

MUSE_TARGET = "muse-glimmer-30b"
ATEM_START, ATEM_EOM, ATEM_EOT, ATEM_MESSAGE = "<|start|>", "<|eom|>", "<|eot|>", "<|message|>"

IM_END = "<|im_end|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"


# ---------------------------------------------------------------------------
# sequence reconstruction

@dataclasses.dataclass
class CallSeq:
    call_index: int
    prompt_ids: list[int]
    gen_ids: list[int]
    channel: np.ndarray                  # int8 [len(gen_ids)]
    stop_type: str = ""

    @property
    def full_ids(self) -> list[int]:
        return self.prompt_ids + self.gen_ids


def _marker_ids(tokenizer, end_ids: Sequence[int] | None = None) -> dict:
    """Single-token markers; "end" = the turn-ending ids (default `<|im_end|>`; GLM: its stop ids)."""
    out = {}
    for s in ((THINK_OPEN, THINK_CLOSE, CALL_OPEN, CALL_CLOSE) + ((IM_END,) if end_ids is None else ())):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is not a single token for this tokenizer: {ids} (these are the Qwen-style "
                             f"channel markers; a model with other delimiters passes its own `marks`)")
        out[s] = ids[0]
    out["end"] = frozenset([out[IM_END]] if end_ids is None else [int(t) for t in end_ids])
    return out


def label_channels(prompt_ids: Sequence[int], gen_ids: Sequence[int], marks: dict[str, int]) -> np.ndarray:
    """Channel code per generated token (see module docstring)."""
    if marks.get("format") == "atem":
        return label_channels_atem(prompt_ids, gen_ids, marks)
    t_open, t_close = marks[THINK_OPEN], marks[THINK_CLOSE]
    state = VISIBLE
    for tid in reversed(prompt_ids):
        if tid == t_open:
            state = REASONING
            break
        if tid == t_close:
            break
    out = np.empty(len(gen_ids), dtype=np.int8)
    for j, tid in enumerate(gen_ids):
        if tid in marks["end"]:
            out[j] = END
        elif state == REASONING:
            out[j] = REASONING
            if tid == t_close:
                state = VISIBLE
        elif state == TOOL_CALL:
            out[j] = TOOL_CALL
            if tid == marks[CALL_CLOSE]:
                state = VISIBLE
        else:
            if tid == marks[CALL_OPEN]:
                state = TOOL_CALL
                out[j] = TOOL_CALL
            else:
                out[j] = VISIBLE
    return out


# ---------------------------------------------------------------------------
# Muse Glimmer (ATEM wire format) channel labelling

def muse_marks(tokenizer, end_ids: Sequence[int] | None = None) -> dict:
    """Channel marks for the ATEM format (see module docstring); pass as `marks` to `reconstruct`.

    `end_ids` default to `<|eot|>` (the Muse stop token). The dict carries `"format": "atem"`, which makes
    `label_channels` use `label_channels_atem`, and `"decode"`, used only to read the recipient text."""
    out = {"format": "atem"}
    for key, s in (("start", ATEM_START), ("eom", ATEM_EOM), ("eot", ATEM_EOT), ("message", ATEM_MESSAGE)):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is not a single token for this tokenizer: {ids} (Muse ATEM delimiter)")
        out[key] = ids[0]
    out["end"] = frozenset([out["eot"]] if end_ids is None else [int(t) for t in end_ids])
    out["decode"] = lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False)
    return out


def channel_marks(tokenizer, target_model: str | None = None, end_ids: Sequence[int] | None = None) -> dict | None:
    """`marks` for `reconstruct` / `extract` of a target model: Muse -> `muse_marks`; anything else -> None
    (the Qwen-style default, unchanged)."""
    return muse_marks(tokenizer, end_ids) if target_model == MUSE_TARGET else None


_RECIPIENT_RE = re.compile(r"^\s*(?:assistant)?\s*to=([^\s<]+)")


def recipient_channel(header_text: str) -> int:
    """Channel of a segment from the decoded header text (` to=self`, `assistant to=get_weather`, ...):
    self -> REASONING, user -> VISIBLE, another recipient -> TOOL_CALL. A header with no readable recipient
    (no `to=`, or cut off before it) is VISIBLE: plain assistant text."""
    m = _RECIPIENT_RE.match(header_text)
    if not m:
        return VISIBLE
    r = m.group(1)
    return REASONING if r == "self" else VISIBLE if r == "user" else TOOL_CALL


def _atem_initial_state(prompt_ids: Sequence[int], marks: dict):
    """(`body_channel` or None, header ids so far) at the start of the generation.

    Normally the prompt ends with `<|start|>assistant`: a header is open, with no recipient yet. If the prompt
    ends inside an assistant segment (prefill continuation), the generation continues that body; if it ends
    after an `<|eom|>`/`<|eot|>`, a new header starts."""
    start, eom, eot, msg = marks["start"], marks["eom"], marks["eot"], marks["message"]
    for i in range(len(prompt_ids) - 1, -1, -1):
        if prompt_ids[i] != start:
            continue
        seg = list(prompt_ids[i + 1:])
        if eom in seg or eot in seg or marks["end"].intersection(seg):
            return None, []
        if msg in seg:
            k = seg.index(msg)
            text = marks["decode"](seg[:k])
            if text.lstrip().startswith("assistant"):
                return recipient_channel(text), []
            return None, []
        return None, seg          # header still open (`assistant` and maybe part of the recipient)
    return None, []


def label_channels_atem(prompt_ids: Sequence[int], gen_ids: Sequence[int], marks: dict) -> np.ndarray:
    """Channel code per generated token for the ATEM format (rules in the module docstring).

    State machine over ids. In a header (from the segment's first token to `<|message|>`) tokens are collected
    and, when `<|message|>` arrives, all of them plus `<|message|>` get the channel read from the decoded header.
    In a body every token gets the segment's channel; `<|eom|>` (labelled with it) ends the segment and opens
    a new header; a stop token is END. A `<|start|>` seen in a body opens a new header (no `<|eom|>` was
    produced). A generation cut inside a body keeps the channel of that body; one cut inside a header labels
    the header tokens from the recipient text read so far."""
    start, eom, msg, end = marks["start"], marks["eom"], marks["message"], marks["end"]
    decode = marks["decode"]
    out = np.empty(len(gen_ids), dtype=np.int8)
    body, header = _atem_initial_state(prompt_ids, marks)   # body: channel of the open body, else None
    seg_start = 0                                            # index of the first generated token of the header
    pending: list[int] = list(header)                        # header token ids (prompt-side ones included)

    def header_channel() -> int:                             # `<|start|>` is a special token: leave it out
        return recipient_channel(decode([t for t in pending if t != start]))

    def flush(upto: int):                                    # header cut off before `<|message|>`
        if upto > seg_start:
            out[seg_start:upto] = header_channel()

    for j, tid in enumerate(gen_ids):
        tid = int(tid)
        if tid in end:
            if body is None:
                flush(j)
            out[j] = END
            body, pending, seg_start = None, [], j + 1
        elif body is not None:
            if tid == start:                                 # new segment without `<|eom|>`
                body, pending, seg_start = None, [tid], j
                continue
            out[j] = body
            if tid == eom:
                body, pending, seg_start = None, [], j + 1
        elif tid == msg:
            body = header_channel()
            out[seg_start:j + 1] = body
        else:
            if tid == start and pending:                     # a header that never reached `<|message|>`
                flush(j)
                pending, seg_start = [], j
            pending.append(tid)
    if body is None:
        flush(len(gen_ids))
    return out


def reconstruct(traj: tj.Trajectory, tokenizer, end_ids: Sequence[int] | None = None,
                marks: dict | None = None) -> list[CallSeq]:
    """One CallSeq per MODEL_CALL event: tokenize(prompt) + the recorded generated ids.

    Prompt ids recorded by the server (`prompt_token_ids`, e.g. SWE-bench runs via Inspect with vLLM's
    return_token_ids) take precedence over re-tokenizing the prompt string.

    `marks` (default: the Qwen-style markers of `_marker_ids`) lets a model with other channel delimiters
    supply its own dict with the same keys as `_marker_ids` returns, or (Muse Glimmer) `muse_marks(tokenizer)`,
    which switches `label_channels` to the ATEM labelling."""
    marks = marks if marks is not None else _marker_ids(tokenizer, end_ids)
    calls = []
    for e in traj.of(tj.MODEL_CALL):
        d = e.data
        p_ids = ([int(t) for t in d["prompt_token_ids"]] if d.get("prompt_token_ids")
                 else tokenizer.encode(d["prompt"], add_special_tokens=False))
        g_ids = [int(t) for t in d["tokens"]]
        calls.append(CallSeq(call_index=e.call_index, prompt_ids=p_ids, gen_ids=g_ids,
                             channel=label_channels(p_ids, g_ids, marks), stop_type=d.get("stop_type", "")))
    return calls


def is_token_prefix(prev: Sequence[int], new: Sequence[int]) -> bool:
    return len(prev) <= len(new) and np.array_equal(np.asarray(prev), np.asarray(new[:len(prev)]))


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n]), np.asarray(b[:n])
    ne = np.nonzero(a != b)[0]
    return int(ne[0]) if len(ne) else n


def consistency_report(traj: tj.Trajectory, tokenizer) -> dict:
    """Token-prefix and decode-consistency measurements for one trajectory.

    prefix: for each consecutive call pair, is tokenize(prompt_k)+gen_k a token-prefix of
    tokenize(prompt_{k+1}) (the condition for cache reuse)? decode: do the recorded ids decode
    to the recorded generation text (stop token stripped)?
    """
    marks = _marker_ids(tokenizer)
    calls = reconstruct(traj, tokenizer)
    mcs = traj.of(tj.MODEL_CALL)
    prefix, decode = [], []
    for k, (c, e) in enumerate(zip(calls, mcs)):
        gids = c.gen_ids[:-1] if c.gen_ids and c.gen_ids[-1] == marks[IM_END] else c.gen_ids
        text = tokenizer.decode(gids, skip_special_tokens=False)
        decode.append({"call_index": c.call_index, "ok": text == e.data["generation"],
                       "ends_im_end": len(gids) != len(c.gen_ids), "stop_type": c.stop_type,
                       "n_gen": len(c.gen_ids)})
        if k + 1 < len(calls):
            nxt = calls[k + 1].prompt_ids
            full = c.full_ids
            ok = is_token_prefix(full, nxt)
            rec = {"call_index": c.call_index, "ok": ok, "len_prev_full": len(full), "len_next_prompt": len(nxt)}
            if not ok:
                rec["common_prefix"] = common_prefix_len(full, nxt)
                rec["diverges_in_gen"] = rec["common_prefix"] >= len(c.prompt_ids)
            prefix.append(rec)
    return {"rollout_id": traj.meta.get("rollout_id"), "n_calls": len(calls), "prefix": prefix,
            "decode": decode, "total_gen_tokens": sum(len(c.gen_ids) for c in calls)}


# ---------------------------------------------------------------------------
# model side

_NON_TEXT_PARTS = ("vision", "visual", "audio")


def find_decoder_layers(model):
    """The text decoder-block ModuleList (works for CausalLM and ConditionalGeneration wrappers).

    A module named `layers` with exactly `num_hidden_layers` blocks. Anything under a vision/audio tower is
    ignored, so a multimodal wrapper (Muse Glimmer: `model.language_model.layers` next to
    `model.vision_tower.layers`) stays unambiguous even if the two stacks happened to have the same depth.
    """
    n = model.config.get_text_config().num_hidden_layers
    found = [(name, m) for name, m in model.named_modules()
             if name.split(".")[-1] == "layers" and hasattr(m, "__len__") and len(m) == n]
    if len(found) > 1:
        found = [(name, m) for name, m in found
                 if not any(part in seg for seg in name.split(".") for part in _NON_TEXT_PARTS)]
    if len(found) != 1:
        raise ValueError(f"expected exactly one decoder ModuleList with {n} blocks, found {len(found)}")
    return found[0][1]


def check_layers(model, layers: Sequence[int]) -> list[int]:
    n = model.config.get_text_config().num_hidden_layers
    layers = [int(x) for x in layers]
    bad = [x for x in layers if not 1 <= x <= n - 1]
    if bad:
        raise ValueError(f"layers {bad} outside 1..{n - 1} (hidden_states tuple index, pre final norm)")
    return layers


class CallExtractor:
    """Call-synchronous teacher-forced prefill with cross-call cache reuse.

    `process(prompt_ids, gen_ids)` returns {layer: tensor [len(gen_ids), d]} of predictive
    states for the generated tokens and `info` (tokens recomputed, whether the cache was
    reused). This single code path is used for offline extraction (training data) and for
    the online tap, so train/deploy parity holds by construction.
    """

    def __init__(self, model, layers: Sequence[int], *, chunk_tokens: int = 8192, pad_to_chunk: bool = True,
                 pad_id: int = 0):
        import torch
        self._torch = torch
        self.model = model
        self.layers = check_layers(model, layers)
        self.chunk_tokens = int(chunk_tokens)
        # Every forward gets exactly `chunk_tokens` tokens: a short final chunk is padded with
        # `pad_id` AFTER the real tokens (a causal model's states at real positions cannot depend
        # on them) and the padded outputs are dropped. On the cluster's datacenter GPU the fla/causal-conv1d kernels
        # gave garbage for some chunk lengths (kernel_diag --per-call, 2026-10-16) while fixed-size
        # chunks matched a single full forward. Padding pollutes the cache, so no cross-call reuse.
        self.pad_to_chunk = bool(pad_to_chunk)
        self.pad_id = int(pad_id)
        self._blocks = find_decoder_layers(model)
        self._cap: dict[int, object] = {}
        self._handles = [self._blocks[L - 1].register_forward_hook(self._hook(L)) for L in self.layers]
        self.reset()

    def _hook(self, L):
        def fn(_mod, _inp, out):
            self._cap[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self):
        """Forget the cache (start of a rollout)."""
        self._cache = None
        self._cached = np.zeros(0, dtype=np.int64)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _forward(self, ids_chunk):
        kw = dict(past_key_values=self._cache, use_cache=True)
        try:
            out = self.model(input_ids=ids_chunk, logits_to_keep=1, **kw)
        except TypeError:
            out = self.model(input_ids=ids_chunk, **kw)
        self._cache = out.past_key_values

    def process(self, prompt_ids: Sequence[int], gen_ids: Sequence[int]):
        torch = self._torch
        P, full = len(prompt_ids), np.asarray(list(prompt_ids) + list(gen_ids), dtype=np.int64)
        F = len(full)
        if len(gen_ids) == 0:
            return {L: torch.zeros((0, 0)) for L in self.layers}, {"reused": False, "computed": 0}
        # reuse only if the cached sequence is a token-prefix of the new one AND the state at
        # P-1 (predicting the first generated token) is not already behind the cache boundary
        c = len(self._cached)
        reuse = (not self.pad_to_chunk and self._cache is not None and 0 < c <= P - 1
                 and np.array_equal(self._cached, full[:c]))
        if not reuse:
            self.reset()
            c = 0
        need_lo, need_hi = P - 1, F - 1          # keep absolute positions [P-1, F-2]
        parts = {L: [] for L in self.layers}
        with torch.no_grad():
            for s in range(c, F, self.chunk_tokens):
                e = min(F, s + self.chunk_tokens)
                ids = full[s:e]
                if self.pad_to_chunk and len(ids) < self.chunk_tokens:
                    ids = np.concatenate([ids, np.full(self.chunk_tokens - len(ids), self.pad_id, np.int64)])
                chunk = torch.from_numpy(ids).to(self.device)[None]
                self._cap.clear()
                self._forward(chunk)
                lo, hi = max(s, need_lo), min(e, need_hi)
                for L in self.layers:
                    if hi > lo:
                        parts[L].append(self._cap[L][0, lo - s:hi - s].clone())
                self._cap.clear()
        self._cached = full
        states = {L: torch.cat(parts[L], 0) for L in self.layers}
        assert all(v.shape[0] == len(gen_ids) for v in states.values())
        return states, {"reused": bool(reuse), "computed": F - c}


# ---------------------------------------------------------------------------
# extraction to disk

@dataclasses.dataclass
class Extracted:
    call_index: np.ndarray        # int32 [n]   model call each generated token belongs to
    position: np.ndarray          # int32 [n]   index within that call's generation
    token_id: np.ndarray          # int32 [n]
    channel: np.ndarray           # int8  [n]
    hidden: dict[int, np.ndarray]  # layer -> [n, d]
    meta: dict

    def __len__(self):
        return len(self.token_id)


def extract_calls(extractor: CallExtractor, calls: Sequence[CallSeq], *, store_dtype: str = "float32",
                  single_sequence: bool = True):
    """Per-call predictive states for reconstructed calls, in call order.

    With `single_sequence`, every call whose prompt+generation is a token-prefix of the final
    call's sequence (the append-only context; 514/516 recorded call pairs) is read from one pass
    over that final sequence; the rest get their own pass. The states are the same as a per-call
    pass (causal model), at ~1x instead of ~10x the final context.
    """
    extractor.reset()
    if not single_sequence or not calls:
        for c in calls:
            states, info = extractor.process(c.prompt_ids, c.gen_ids)
            yield c, states, info
        return
    last = calls[-1]
    full = list(last.prompt_ids) + list(last.gen_ids)
    inside = [list(c.prompt_ids) + list(c.gen_ids) == full[:len(c.prompt_ids) + len(c.gen_ids)] for c in calls]
    states_full = None
    shared_computed = 0
    if any(inside):
        # one pass: treat the whole final sequence as "generated" after its first token
        st, info = extractor.process(full[:1], full[1:])   # row j = state predicting full[j+1] = state at position j
        shared_computed = info["computed"]
        states_full = st
        extractor.reset()
    for c, ok in zip(calls, inside):
        if ok and c.gen_ids:
            P = len(c.prompt_ids)
            rows = slice(P - 1, P - 1 + len(c.gen_ids))       # predictive: state at P-1+j predicts gen j
            # Attribute the shared forward pass once, rather than reporting zero
            # computation for every call in the rollout.
            yield c, {L: v[rows] for L, v in states_full.items()}, {"reused": False, "computed": shared_computed,
                                                                     "single_sequence": True}
            shared_computed = 0
        else:
            states, info = extractor.process(c.prompt_ids, c.gen_ids)
            yield c, states, {**info, "single_sequence": False}


def _to_store(t, store_dtype: str) -> np.ndarray:
    a = t.float().cpu().numpy()
    if store_dtype == "float16":
        if np.abs(a).max(initial=0.0) > 65000:
            raise OverflowError("activation magnitude exceeds float16 range; use store_dtype='float32'")
        return a.astype(np.float16)
    if store_dtype != "float32":
        raise ValueError(store_dtype)
    return a


def extract(model, tokenizer, traj: tj.Trajectory, layers: Sequence[int], *, store_dtype: str = "float32",
            chunk_tokens: int = 8192, extractor: CallExtractor | None = None,
            calls: Sequence[CallSeq] | None = None, marks: dict | None = None) -> Extracted:
    """Predictive hidden states at `layers` for every generated token of every call.

    `marks`: channel marks for `reconstruct` when `calls` is not given (None = Qwen default; Muse: `muse_marks`)."""
    own = extractor is None
    ex = extractor or CallExtractor(model, layers, chunk_tokens=chunk_tokens)
    if calls is None:
        calls = reconstruct(traj, tokenizer) if marks is None else reconstruct(traj, tokenizer, marks=marks)
    ci, pos, tid, ch, hs, infos = [], [], [], [], {L: [] for L in ex.layers}, []
    try:
        for c, states, info in extract_calls(ex, calls):
            n = len(c.gen_ids)
            ci.append(np.full(n, c.call_index, np.int32))
            pos.append(np.arange(n, dtype=np.int32))
            tid.append(np.asarray(c.gen_ids, np.int32))
            ch.append(c.channel)
            for L in ex.layers:
                hs[L].append(_to_store(states[L], store_dtype))
            infos.append({"call_index": c.call_index, "n_prompt": len(c.prompt_ids), "n_gen": n, **info})
    finally:
        if own:
            ex.close()
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)
    d = {L: (np.concatenate(v) if v else np.zeros((0, 0), np.float32)) for L, v in hs.items()}
    meta = {"rollout_id": traj.meta.get("rollout_id"), "task_id": traj.meta.get("task_id"),
            "layers": list(ex.layers), "layer_convention": LAYER_CONVENTION, "alignment": ALIGNMENT,
            "store_dtype": store_dtype, "calls": infos}
    return Extracted(cat(ci, np.int32), cat(pos, np.int32), cat(tid, np.int32), cat(ch, np.int8), d, meta)


def save_extracted(path: str | os.PathLike, ex: Extracted) -> None:
    """One .npz per rollout, uncompressed, written atomically (tmp + rename)."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    arrays = {"call_index": ex.call_index, "position": ex.position, "token_id": ex.token_id,
              "channel": ex.channel, "meta_json": np.frombuffer(json.dumps(ex.meta).encode(), np.uint8)}
    arrays.update({f"hidden_L{L}": h for L, h in ex.hidden.items()})
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load_extracted(path: str | os.PathLike, layers: Sequence[int] | None = None) -> Extracted:
    """Load one rollout file; `layers` restricts which hidden arrays are read from disk."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta_json"]).decode())
        hidden = {int(k[len("hidden_L"):]): z[k] for k in z.files if k.startswith("hidden_L")
                  and (layers is None or int(k[len("hidden_L"):]) in layers)}
        return Extracted(z["call_index"], z["position"], z["token_id"], z["channel"], hidden, meta)


def tokenizer_digest(tokenizer) -> str:
    """Hash of the vocabulary + merges as the tokenizer exposes them (revision-independent check)."""
    h = hashlib.sha256()
    h.update(json.dumps(sorted(tokenizer.get_vocab().items()), ensure_ascii=False).encode())
    return h.hexdigest()
