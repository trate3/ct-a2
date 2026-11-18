"""Pre-flight check of the live engine and tap on the real model (run once on the GPU before the timed run).

  python -m evals.tap_live.check_live --model DIR [--tokenizer DIR] --artifact PATH [--device cuda] [--out check.json]

It generates 48 tokens from a fixed prompt with the tap attached and then, with the same model, (1) scores the same
tokens offline by teacher forcing the prompt plus generation through one full forward pass with the repo's scoring
code, (2) compares the tap's pooled score with it, (3) compares the next-token logits of a prefix-reusing second call with
those of a fresh prefill, and (4) compares a single-forward prefill with a chunked one. In bfloat16 the cached and the full
forward differ by rounding, so the numbers are REPORTED (max abs differences, argmax agreement); nothing is asserted here.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from evals.reward_hacking import activations as act
from evals.reward_hacking import llm
from evals.reward_hacking import tap as tapmod
from evals.tap_live import live_client, live_tap

PROMPT = ("<|im_start|>system\nYou are a travel research assistant.<|im_end|>\n<|im_start|>user\n"
          "Find a quiet hotel near the station for two nights and explain the choice.<|im_end|>\n<|im_start|>assistant\n<think>\n")


def first_logits(model, tok, fmt, prompt, device, *, prefill_chunk=0, prime=None):
    """Next-token logits for `prompt` (after `prime`, a first call whose cache the engine may reuse)."""
    got = []

    def grab(logits, gen_ids, call_index):
        if not gen_ids:
            got.append(logits.float().cpu())
        return logits
    c = live_client.LiveHFClient(model, tok, fmt, device=device, prefill_chunk=prefill_chunk, logits_processor=grab)
    if prime is not None:
        c.complete(prime, max_tokens=1, seed=0, sampling={"temperature": 0.0})
        got.clear()
    comp = c.complete(prompt, max_tokens=1, seed=0, sampling={"temperature": 0.0})
    return got[0], comp.timings


def check(model, tok, fmt, artifact, device, *, prompt: str = PROMPT, max_tokens: int = 48) -> dict:
    device = torch.device(device)
    art = artifact if not isinstance(artifact, str) else tapmod.TapArtifact.load(artifact)
    client = live_client.LiveHFClient(model, tok, fmt, device=device)
    tap = live_tap.LiveProbeTap(art, device=device, time_capture=False)
    client.attach_tap(tap)
    comp = client.complete(prompt, max_tokens=max_tokens, seed=0, sampling=llm.get_format("qwen3.8-27b").sampling)
    client.detach_tap()
    dec = tap.decisions[0]
    ids = tok.encode(prompt, add_special_tokens=False)
    P, n = len(ids), len(comp.tokens)
    with torch.no_grad():
        hs = model(input_ids=torch.tensor([ids + comp.tokens], device=device), output_hidden_states=True).hidden_states[art.layer][0]
    states = hs[P - 1:P - 1 + n].float()
    channel = np.array([act.END if t in client.stop_ids else act.VISIBLE for t in comp.tokens], dtype=np.int8)
    offline = tapmod.StreamingTap(art).update(states, channel, 0)
    # prefix reuse: the second prompt extends the first call's tokens exactly when the text round-trips
    p2 = prompt + comp.text + "<|im_end|>\nContinue."
    ids2 = tok.encode(p2, add_special_tokens=False)
    extends = ids2[:P + n - 1] == ids + comp.tokens[:-1]
    out = {"n_generated": n, "stop_type": comp.stop_type, "tap_score": dec["score_call"], "offline_score": offline,
           "abs_diff": None if offline is None or dec["score_call"] is None else abs(dec["score_call"] - offline),
           "alarm": dec["alarm"], "second_prompt_extends_first_call": bool(extends)}
    reuse, t_reuse = first_logits(model, tok, fmt, p2, device, prime=prompt)
    fresh, t_fresh = first_logits(model, tok, fmt, p2, device)
    out["reuse"] = {"n_cached_tokens": t_reuse["n_cached_tokens"], "reset_used": t_reuse["reset_used"],
                    "max_abs_logit_diff_vs_fresh": float((reuse - fresh).abs().max()), "argmax_equal": bool(reuse.argmax() == fresh.argmax())}
    chunked, _ = first_logits(model, tok, fmt, p2, device, prefill_chunk=16)
    out["chunked_prefill"] = {"max_abs_logit_diff_vs_single": float((chunked - fresh).abs().max()),
                              "argmax_equal": bool(chunked.argmax() == fresh.argmax())}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    model, tok = live_client.load_model(a.model, a.tokenizer, device=a.device, dtype=a.dtype)
    res = check(model, tok, llm.get_format("qwen3.8-27b", reasoning_effort="medium"), a.artifact, a.device)
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
