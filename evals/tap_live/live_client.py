"""In-process generation engine for ONE sequence at a time, used as the agent's `client` (both arms of the overhead run).

`LiveHFClient.complete(prompt, max_tokens, seed, sampling)` has the contract of `llm.VLLMCompletionClient.complete`
and returns an `llm.Completion`:

  * the rendered prompt is tokenized as `activations.reconstruct` does (`tokenizer.encode(prompt, add_special_tokens=False)`);
  * our own autoregressive loop (not HF `generate`): one prefill forward, then one forward per generated token;
  * sampling is temperature, then min_p, top_k, top_p (on the top-k probabilities), then a multinomial draw with a
    `torch.Generator(device)` seeded with `seed`. Penalties must be 0 / 1 (asserted). The same seed and the same
    model computation give the same tokens, with or without a tap;
  * generation stops at a stop token id (kept in `tokens`, dropped from `text`, as vLLM does) or at `max_tokens`
    (stop_type "limit"). `text` is the decode of the generated ids with special tokens kept (`</think>`,
    `<tool_call>` are needed by the parser);
  * the KV cache (the model's own DynamicCache, which also holds the linear-attention states) is KEPT ACROSS THE CALLS OF
    AN EPISODE. The engine remembers the exact token ids it fed to the model. On a new call, if they are exactly a prefix
    of the new prompt ids, only the new tokens are prefilled; otherwise the cache is reset and everything is prefilled
    (a recurrent state cannot be cropped). Resets are counted;
  * exactly ONE host synchronisation per generated token (the sampled id, needed to test the stop condition), in both
    arms, and nothing else inside the loop. Timing therefore needs no extra synchronisation: `prefill_s` ends when the
    first token is on the host, `decode_s` is the rest of the loop.

A `tap` (see `live_tap.LiveProbeTap`) may be attached: it sees every forward through a hook, and the client calls
`tap.end_call(...)` after the last token is sampled and before returning.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from evals.reward_hacking import activations as act
from evals.reward_hacking import llm

PENALTY_NEUTRAL = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0, "repetition_penalty": 1.0}
SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p"} | set(PENALTY_NEUTRAL)


def check_sampling(sampling: dict) -> dict:
    """Validated settings with defaults filled in. Penalties other than neutral are refused, not ignored."""
    bad = set(sampling) - SAMPLING_KEYS
    if bad:
        raise ValueError(f"sampling parameters {sorted(bad)} are not implemented")
    for k, neutral in PENALTY_NEUTRAL.items():
        if k in sampling and float(sampling[k]) != neutral:
            raise ValueError(f"{k}={sampling[k]} is not implemented (only {neutral})")
    out = {"temperature": float(sampling.get("temperature", 1.0)), "top_p": float(sampling.get("top_p", 1.0)),
           "top_k": int(sampling.get("top_k", 0)), "min_p": float(sampling.get("min_p", 0.0))}
    if out["temperature"] < 0 or not 0.0 < out["top_p"] <= 1.0 or out["top_k"] < 0 or not 0.0 <= out["min_p"] < 1.0:
        raise ValueError(f"invalid sampling settings {out}")
    return out


def sample_token(logits, s: dict, generator):
    """One token id (a 1-element tensor on the logits' device) from float logits [V]; no host synchronisation."""
    logits = logits.float()
    if s["temperature"] == 0.0:
        return logits.argmax().view(1)
    logits = logits / s["temperature"]
    if s["min_p"] > 0.0:           # drop tokens whose probability is below min_p times the largest one
        floor = logits.max() + float(np.log(s["min_p"]))
        logits = torch.where(logits < floor, torch.full_like(logits, float("-inf")), logits)
    k = s["top_k"]
    if 0 < k < logits.shape[0]:
        vals, idx = torch.topk(logits, k)
    else:
        vals, idx = torch.sort(logits, descending=True)
    probs = torch.softmax(vals, dim=-1)
    if s["top_p"] < 1.0:           # keep the smallest prefix whose mass reaches top_p (the first token always stays)
        before = torch.cumsum(probs, dim=-1) - probs
        probs = torch.where(before >= s["top_p"], torch.zeros_like(probs), probs)
    choice = torch.multinomial(probs, 1, generator=generator)
    return idx[choice]


class LiveHFClient:
    """One-sequence engine over a transformers model (see the module docstring)."""

    engine = "live-hf"

    def __init__(self, model, tokenizer, fmt=None, *, stop_token_ids=None, device=None, prefill_chunk: int = 0,
                 logits_processor=None, name: str = "agent", max_context_tokens: int = 262144):
        self.model, self.tok, self.name = model, tokenizer, name
        ids = fmt.stop_token_ids if stop_token_ids is None else stop_token_ids
        self.stop_ids = frozenset(int(t) for t in ids)
        if not self.stop_ids:
            raise ValueError("no stop token ids")
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.cuda = self.device.type == "cuda"
        self.prefill_chunk = int(prefill_chunk)
        self.max_context_tokens = int(max_context_tokens)    # `max_tokens=None` means "up to this context" (the vLLM server's 262,144)
        # test hook only: callable(logits [V] float, generated ids so far, call index) -> logits. Production runs pass None.
        self.logits_processor = logits_processor
        self.tap = None
        self.reset_episode()

    # -- tap
    def attach_tap(self, tap) -> None:
        if self.tap is not None:
            self.detach_tap()
        tap.attach(self.model)
        self.tap = tap
        tap.begin_episode()

    def detach_tap(self) -> None:
        if self.tap is not None:
            self.tap.detach()
        self.tap = None

    # -- episode state
    def reset_episode(self) -> None:
        """Forget the KV cache and the per-episode logs (start of an episode)."""
        self._cache = None
        self._processed: list[int] = []
        self.calls: list[dict] = []          # per call: the timings dict that is also in the Completion
        self.generated: list[list[int]] = []  # per call: generated ids (stop token included)
        self.resets = 0
        if self.tap is not None:
            self.tap.begin_episode()

    def resolve_episode(self) -> None:
        """Turn pending CUDA-event timings of the tap into numbers (after the episode wall time was taken)."""
        if self.tap is not None:
            self.tap.resolve()

    def _drop_cache(self) -> None:
        self._cache, self._processed = None, []

    # -- generation
    def _forward(self, ids):
        out = self.model(input_ids=ids, past_key_values=self._cache, use_cache=True, logits_to_keep=1)
        self._cache = out.past_key_values
        return out.logits[0, -1]

    @torch.no_grad()
    def complete(self, prompt: str, *, max_tokens: int | None, seed: int, sampling: dict, extra=None) -> llm.Completion:
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer or None (no cap: the rest of the context)")
        s = check_sampling(sampling)
        t_start = time.perf_counter()
        ids = self.tok.encode(prompt, add_special_tokens=False)
        n_prompt = len(ids)
        if max_tokens is None:             # as the vLLM client's `uncapped_max_tokens`: server context minus the prompt
            max_tokens = self.max_context_tokens - n_prompt
            if max_tokens < 1:
                raise ValueError(f"the prompt ({n_prompt} tokens) leaves no room in a {self.max_context_tokens}-token context")
        tokenize_s = time.perf_counter() - t_start

        had_cache = bool(self._processed)
        processed_before = len(self._processed)
        common = act.common_prefix_len(self._processed, ids) if had_cache else 0
        reuse = had_cache and common == len(self._processed) and n_prompt > common
        reset_used = had_cache and not reuse
        if reset_used:
            self.resets += 1
        if not reuse:
            self._drop_cache()
        start = len(self._processed)          # tokens already in the cache (0 after a reset)

        call_index = len(self.calls)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        tap = self.tap
        if tap is not None:
            tap.begin_call(max_tokens, call_index)
        gen_ids: list[int] = []
        stopped = False
        try:
            t0 = time.perf_counter()
            new = ids[start:]
            step = self.prefill_chunk or len(new)
            logits = None
            for lo in range(0, len(new), step):
                chunk = torch.tensor(new[lo:lo + step], dtype=torch.long, device=self.device)[None]
                if tap is not None:
                    tap.capture_enabled = lo + step >= len(new)     # only the last chunk's last position predicts a token
                logits = self._forward(chunk)
            if self.logits_processor is not None:
                logits = self.logits_processor(logits, gen_ids, call_index)
            tok = sample_token(logits, s, gen)
            tid = int(tok.item())                                   # the one host synchronisation of this step
            t1 = time.perf_counter()
            gen_ids.append(tid)
            stopped = tid in self.stop_ids
            while not stopped and len(gen_ids) < max_tokens:
                logits = self._forward(tok.view(1, 1))
                if self.logits_processor is not None:
                    logits = self.logits_processor(logits, gen_ids, call_index)
                tok = sample_token(logits, s, gen)
                tid = int(tok.item())
                gen_ids.append(tid)
                stopped = tid in self.stop_ids
            t2 = time.perf_counter()
        except BaseException:
            self._drop_cache()
            if tap is not None:
                tap.abort_call()
            raise
        # every generated token but the last was fed to the model; the last one (stop token or limit) was not
        self._processed = ids + gen_ids[:-1]

        timings = {"engine": self.engine, "call_index": call_index, "prefill_s": t1 - t0, "decode_s": t2 - t1,
                   "tokenize_s": tokenize_s, "n_prompt_tokens": n_prompt, "n_prefill_tokens": n_prompt - start,
                   "n_cached_tokens": start, "n_generated": len(gen_ids), "reset_used": reset_used,
                   "common_prefix": common, "processed_before": processed_before, "stopped_on_stop_token": stopped}
        if tap is not None:
            timings.update(tap.end_call(n_generated=len(gen_ids), last_is_stop=stopped, call_index=call_index,
                                        record=timings))
        t3 = time.perf_counter()
        text_ids = gen_ids[:-1] if stopped else gen_ids
        text = self.tok.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        timings["detok_s"] = time.perf_counter() - t3
        self.calls.append(timings)
        self.generated.append(gen_ids)
        return llm.Completion(text=text, tokens=gen_ids, stop_type="eos" if stopped else "limit",
                              prompt_tokens=n_prompt, cached_tokens=start, seconds=time.perf_counter() - t_start,
                              timings=timings)


def load_model(model_dir: str, tokenizer_dir: str | None = None, *, device: str = "cuda", dtype: str = "bfloat16",
               model_class: str = "conditional"):
    """(model, tokenizer) as `scripts/extract_activations.load_model` loads the agent: AutoModelForImageTextToText
    (the checkpoint's native Qwen3_5ForConditionalGeneration) unless `model_class="auto"` (AutoModelForCausalLM).
    A CUDA model is placed on the device at load time."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir or model_dir)
    if model_class == "conditional":
        from transformers import AutoModelForImageTextToText as cls
    else:
        cls = AutoModelForCausalLM
    kw = {"device_map": {"": device}} if device.startswith("cuda") else {}
    model = cls.from_pretrained(model_dir, dtype=getattr(torch, dtype), **kw)
    return model.to(device).eval(), tok
