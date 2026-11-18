"""CPU tests of the live engine and the probe tap on a TINY random Qwen3_5ForConditionalGeneration (built from the real
Qwen3.8-27B config structure: hybrid linear/full attention; transformers' own torch fallbacks for the gated delta rule)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from evals.reward_hacking import activations as act
from evals.reward_hacking import tap as tapmod
from evals.tap_live import live_client as lc
from evals.tap_live import live_tap as lt
from evals.tap_live import tiny_model as tm
from evals.tap_live.conftest import n_tap_hooks

SP = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
PROMPT = "hello the user wants me to implement a function in func.py"
LAYER = 2


def forced(tok, script_ids):
    """Test hook: force the generated ids of call c (then the stop token)."""
    stop = tok.convert_tokens_to_ids("<|im_end|>")

    def proc(logits, gen_ids, call_index):
        seq = script_ids[min(call_index, len(script_ids) - 1)]
        want = seq[len(gen_ids)] if len(gen_ids) < len(seq) else stop
        out = torch.full_like(logits, float("-inf"))
        out[want] = 0.0
        return out
    return proc


def make_client(model, tok, fmt, tap=None, **kw):
    c = lc.LiveHFClient(model, tok, fmt, device="cpu", **kw)
    if tap is not None:
        c.attach_tap(tap)
    return c


# ---- (a) same seed, same tokens, with and without the tap -----------------------------------------------------------

@pytest.mark.parametrize("pooling,alpha", [("token_mean", 1.0), ("ema_max", 0.3)])
def test_tap_does_not_change_generation(model, tok, fmt, pooling, alpha):
    seen = {}

    def record(key):
        def proc(logits, gen_ids, call_index):
            seen.setdefault(key, []).append(logits.clone())
            return logits
        return proc

    for seed in (1, 2, 3):
        plain = make_client(model, tok, fmt, logits_processor=record(("plain", seed)))
        a = plain.complete(PROMPT, max_tokens=25, seed=seed, sampling=SP)
        tap = lt.LiveProbeTap(tm.random_artifact(LAYER, pooling, alpha=alpha), device="cpu")
        tapped = make_client(model, tok, fmt, tap, logits_processor=record(("tap", seed)))
        b = tapped.complete(PROMPT, max_tokens=25, seed=seed, sampling=SP)
        assert a.tokens == b.tokens and a.text == b.text
        assert len(a.tokens) == 25 and a.stop_type == "limit"
        for x, y in zip(seen[("plain", seed)], seen[("tap", seed)]):
            assert torch.equal(x, y)            # the hook changes no logit, bit for bit
        tapped.detach_tap()
    c = make_client(model, tok, fmt).complete(PROMPT, max_tokens=25, seed=99, sampling=SP)
    assert c.tokens != a.tokens                  # a different seed gives a different stream


def test_tap_removed_hook_leaves_no_trace(model, tok, fmt):
    tap = lt.LiveProbeTap(tm.random_artifact(LAYER), device="cpu")
    c = make_client(model, tok, fmt, tap)
    assert n_tap_hooks(model, LAYER) == 1
    c.detach_tap()
    assert n_tap_hooks(model, LAYER) == 0


def test_sampling_penalties_and_unknown_keys(model, tok, fmt):
    c = make_client(model, tok, fmt)
    with pytest.raises(ValueError):
        c.complete(PROMPT, max_tokens=3, seed=0, sampling={**SP, "presence_penalty": 0.5})
    with pytest.raises(ValueError):
        c.complete(PROMPT, max_tokens=3, seed=0, sampling={**SP, "repeat_penalty": 1.1})
    with pytest.raises(ValueError):
        c.complete(PROMPT, max_tokens=3, seed=0, sampling={**SP, "bogus": 1})
    ok = c.complete(PROMPT, max_tokens=3, seed=0, sampling={**SP, "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0})
    assert len(ok.tokens) == 3


def test_sampling_filters():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([5.0, 4.0, 3.0, -10.0, -10.0, -10.0])
    seen = {int(lc.sample_token(logits, lc.check_sampling({"temperature": 1.0, "top_k": 2}), g)) for _ in range(200)}
    assert seen == {0, 1}                                              # top-k keeps two tokens
    seen = {int(lc.sample_token(logits, lc.check_sampling({"temperature": 1.0, "top_p": 0.5}), g)) for _ in range(200)}
    assert seen == {0}                                                 # the first token carries more than half the mass
    seen = {int(lc.sample_token(logits, lc.check_sampling({"temperature": 1.0, "min_p": 0.2}), g)) for _ in range(300)}
    assert seen <= {0, 1} and 0 in seen                                # token 2 has p/pmax = e^-2 = 0.135 < 0.2
    assert int(lc.sample_token(logits, lc.check_sampling({"temperature": 0.0}), g)) == 0


def test_stop_token_kept_in_tokens_dropped_from_text(model, tok, fmt):
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    think = tok.convert_tokens_to_ids("</think>")
    body = tok.encode(" the user wants", add_special_tokens=False)
    c = make_client(model, tok, fmt, logits_processor=forced(tok, [body + [think]]))
    comp = c.complete(PROMPT, max_tokens=30, seed=0, sampling=SP)
    assert comp.tokens == body + [think, stop] and comp.stop_type == "eos"
    assert comp.text == " the user wants</think>"                      # special token kept for the parser, stop dropped
    assert comp.timings["stopped_on_stop_token"] and comp.timings["n_generated"] == len(comp.tokens)


# ---- (b) tap score equals the offline scoring on the hidden states of a full forward pass ----------------------------

def full_states(model, ids, layer):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    return out.hidden_states[layer][0]          # hidden_states[L] = output of block L-1


def offline_call_states(model, prompt_ids, gen_ids, layer):
    hs = full_states(model, list(prompt_ids) + list(gen_ids), layer)
    P = len(prompt_ids)
    return hs[P - 1:P - 1 + len(gen_ids)]       # predictive: the state that produced token j is at position P+j-1


def offline_channel(gen_ids, stop_ids):
    return np.array([act.END if t in stop_ids else act.VISIBLE for t in gen_ids], dtype=np.int8)


@pytest.mark.parametrize("pooling,alpha,init", [("token_mean", 1.0, "first"), ("ema_max", 1.0, "first"),
                                                 ("ema_max", 0.3, "first"), ("ema_max", 0.05, "zero")])
def test_tap_score_equals_offline_with_stop_token_excluded(model, tok, fmt, pooling, alpha, init):
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    art = tm.random_artifact(LAYER, pooling, alpha=alpha, ema_init=init, seed=3)
    body = tok.encode(" the user wants me to look at both files now", add_special_tokens=False)
    body2 = tok.encode(" and then the tests", add_special_tokens=False)
    tap = lt.LiveProbeTap(art, device="cpu", scope="call")
    c = make_client(model, tok, fmt, tap, logits_processor=forced(tok, [body, body2]))
    p1 = PROMPT
    c1 = c.complete(p1, max_tokens=40, seed=0, sampling=SP)
    assert c1.tokens[-1] == stop and len(c1.tokens) == len(body) + 1
    ids1 = tok.encode(p1, add_special_tokens=False)
    st1 = offline_call_states(model, ids1, c1.tokens, LAYER)
    ch1 = offline_channel(c1.tokens, {stop})
    off1 = tapmod.StreamingTap(art)
    S1 = off1.update(st1, ch1, 0)
    d1 = tap.decisions[0]
    assert d1["n_tokens"] == len(body)                                      # the stop token's state is not pooled
    assert d1["score_call"] == pytest.approx(S1, abs=1e-5)
    assert d1["score_rollout"] == pytest.approx(S1, abs=1e-5)
    assert tap.score_of_call(0) == d1["score"]
    # including the stop token would give another number (the exclusion is real)
    incl = tapmod.StreamingTap(art).update(st1, np.full(len(c1.tokens), act.VISIBLE, np.int8), 0)
    if pooling == "token_mean":            # (a max pool may not move when one more state is added)
        assert abs(incl - S1) > 1e-6
    # second call: the prompt extends the first call's tokens; call scope is a fresh state, rollout scope carries on
    p2 = p1 + c1.text + "<|im_end|>\nnext user turn about the tests"
    c2 = c.complete(p2, max_tokens=40, seed=1, sampling=SP)
    ids2 = tok.encode(p2, add_special_tokens=False)
    st2 = offline_call_states(model, ids2, c2.tokens, LAYER)
    ch2 = offline_channel(c2.tokens, {stop})
    S2_call = tapmod.StreamingTap(art).update(st2, ch2, 1)
    off1.update(st2, ch2, 1)
    S2_roll = off1.S
    d2 = tap.decisions[1]
    assert d2["score_call"] == pytest.approx(S2_call, abs=1e-5)
    assert d2["score_rollout"] == pytest.approx(S2_roll, abs=1e-5)
    assert d2["score"] == d2["score_call"] and d2["scope"] == "call"
    assert d2["alarm"] == (d2["score"] > art.threshold)


def test_rollout_scope_alarm_and_episode_reset(model, tok, fmt):
    art = tm.random_artifact(LAYER, "ema_max", alpha=0.5, threshold=-1e9)
    body = tok.encode(" the user wants me", add_special_tokens=False)
    tap = lt.LiveProbeTap(art, device="cpu", scope="rollout")
    c = make_client(model, tok, fmt, tap, logits_processor=forced(tok, [body]))
    c.complete(PROMPT, max_tokens=20, seed=0, sampling=SP)
    assert tap.decisions[0]["alarm"] and tap.decisions[0]["score"] == tap.decisions[0]["score_rollout"]
    c.reset_episode()
    assert tap.decisions == [] and c.calls == [] and c._cache is None


def test_no_state_scored_when_stop_is_first_token(model, tok, fmt):
    tap = lt.LiveProbeTap(tm.random_artifact(LAYER), device="cpu")
    c = make_client(model, tok, fmt, tap, logits_processor=forced(tok, [[]]))
    comp = c.complete(PROMPT, max_tokens=10, seed=0, sampling=SP)
    assert len(comp.tokens) == 1
    assert tap.decisions[0]["score"] is None and tap.decisions[0]["alarm"] is False and tap.decisions[0]["n_tokens"] == 0


def test_tap_timings_recorded(model, tok, fmt):
    tap = lt.LiveProbeTap(tm.random_artifact(LAYER, "ema_max", alpha=0.3), device="cpu")
    c = make_client(model, tok, fmt, tap)
    comp = c.complete(PROMPT, max_tokens=10, seed=0, sampling=SP)
    t = comp.timings
    assert t["capture_s"] > 0 and t["score_s"] > 0 and t["tap_s"] == pytest.approx(t["capture_s"] + t["score_s"])
    assert c.calls[0] is comp.timings
    off = lt.LiveProbeTap(tm.random_artifact(LAYER), device="cpu", time_capture=False)
    c2 = make_client(model, tok, fmt, off)
    t2 = c2.complete(PROMPT, max_tokens=10, seed=0, sampling=SP).timings
    assert t2["capture_s"] is None and t2["tap_s"] == t2["score_s"]


def test_chunked_prefill_captures_only_the_last_position(model, tok, fmt):
    art = tm.random_artifact(LAYER, "token_mean", seed=4)
    t1, t2 = (lt.LiveProbeTap(art, device="cpu") for _ in range(2))
    a = make_client(model, tok, fmt, t1)
    b = make_client(model, tok, fmt, t2, prefill_chunk=4)
    ca = a.complete(PROMPT, max_tokens=8, seed=0, sampling=SP)
    cb = b.complete(PROMPT, max_tokens=8, seed=0, sampling=SP)
    assert ca.tokens == cb.tokens
    assert t1.decisions[0]["score"] == pytest.approx(t2.decisions[0]["score"], abs=1e-5)


# ---- (c) prefix reuse ---------------------------------------------------------------------------------------------------

def test_prefix_reuse_prefills_only_new_tokens_and_matches_fresh(model, tok, fmt):
    body = tok.encode(" the user wants me to look", add_special_tokens=False)
    grabbed = {}

    def grab(tag):
        def proc(logits, gen_ids, call_index):
            if not gen_ids:
                grabbed[tag] = logits.clone()
            return forced(tok, [body])(logits, gen_ids, call_index)
        return proc

    c = make_client(model, tok, fmt, logits_processor=grab("reuse1"))
    c1 = c.complete(PROMPT, max_tokens=30, seed=0, sampling=SP)
    t1 = c1.timings
    assert t1["n_cached_tokens"] == 0 and t1["reset_used"] is False and t1["n_prefill_tokens"] == t1["n_prompt_tokens"]
    p2 = PROMPT + c1.text + "<|im_end|>\nand then the tests"
    c.logits_processor = grab("reuse2")
    c2 = c.complete(p2, max_tokens=30, seed=0, sampling=SP)
    t2 = c2.timings
    n_before = len(tok.encode(PROMPT, add_special_tokens=False)) + len(body)          # prompt + generated but the stop token
    assert t2["reset_used"] is False and c.resets == 0
    assert t2["n_cached_tokens"] == n_before and t2["n_cached_tokens"] == c2.cached_tokens
    assert t2["n_prefill_tokens"] == t2["n_prompt_tokens"] - n_before
    fresh = make_client(model, tok, fmt, logits_processor=grab("fresh"))
    fresh.complete(p2, max_tokens=30, seed=0, sampling=SP)
    assert fresh.calls[0]["n_cached_tokens"] == 0
    assert torch.allclose(grabbed["reuse2"], grabbed["fresh"], atol=1e-4, rtol=1e-4)  # same next-token distribution
    assert torch.equal(grabbed["reuse2"].argmax(), grabbed["fresh"].argmax())


def test_diverging_prompt_resets_and_is_counted(model, tok, fmt):
    c = make_client(model, tok, fmt)
    c.complete(PROMPT, max_tokens=6, seed=0, sampling=SP)
    other = "something else entirely: " + PROMPT
    comp = c.complete(other, max_tokens=6, seed=0, sampling=SP)
    t = comp.timings
    assert t["reset_used"] is True and c.resets == 1 and t["n_cached_tokens"] == 0
    assert t["n_prefill_tokens"] == t["n_prompt_tokens"]
    fresh = make_client(model, tok, fmt).complete(other, max_tokens=6, seed=0, sampling=SP)
    assert fresh.tokens == comp.tokens                                  # a reset gives the fresh-prefill result
    # the same prompt again has no new tokens to prefill: also a reset (the state cannot be rewound)
    again = c.complete(other, max_tokens=6, seed=0, sampling=SP)
    assert again.timings["reset_used"] is True and c.resets == 2
    c.reset_episode()
    assert c.resets == 0 and c._cache is None
    assert c.complete(PROMPT, max_tokens=3, seed=0, sampling=SP).timings["reset_used"] is False


def test_reused_cache_generates_like_a_fresh_one(model, tok, fmt):
    body = tok.encode(" the user wants me", add_special_tokens=False)
    c = make_client(model, tok, fmt, logits_processor=forced(tok, [body]))
    c1 = c.complete(PROMPT, max_tokens=30, seed=0, sampling=SP)
    p2 = PROMPT + c1.text + "<|im_end|>\nnow what"
    c.logits_processor = None
    reused = c.complete(p2, max_tokens=12, seed=5, sampling=SP)
    assert reused.timings["n_cached_tokens"] > 0
    fresh = make_client(model, tok, fmt).complete(p2, max_tokens=12, seed=5, sampling=SP)
    assert reused.tokens == fresh.tokens


def test_check_live_preflight_runs_on_the_tiny_model(model, tok, fmt):
    from evals.tap_live import check_live
    prompt = "<|im_start|>user\nhello the user wants me<|im_end|>\n<|im_start|>assistant\n<think>\n"
    res = check_live.check(model, tok, fmt, tm.random_artifact(LAYER, "ema_max", alpha=0.4), "cpu", prompt=prompt, max_tokens=20)
    assert res["abs_diff"] < 1e-5
    assert res["reuse"]["max_abs_logit_diff_vs_fresh"] < 1e-4 and res["chunked_prefill"]["argmax_equal"]
    assert res["reuse"]["n_cached_tokens"] > 0 and res["reuse"]["reset_used"] is False


# ---- several heads (the coding run: token-mean and EMA-max in one arm) ------------------------------------------------

@pytest.mark.parametrize("layers", [(2, 2), (1, 3)])
def test_two_heads_share_or_split_the_capture_and_match_offline(model, tok, fmt, layers):
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    mean = tm.random_artifact(layers[0], "token_mean", seed=5)
    ema = tm.random_artifact(layers[1], "ema_max", alpha=0.3, seed=6)
    tap = lt.LiveProbeTap({"token_mean": mean, "ema_max": ema}, device="cpu")
    assert tap.layers == sorted(set(layers))
    body = tok.encode(" the user wants me to look at both files now", add_special_tokens=False)
    c = make_client(model, tok, fmt, tap, logits_processor=forced(tok, [body]))
    comp = c.complete(PROMPT, max_tokens=40, seed=0, sampling=SP)
    assert n_tap_hooks(model, layers[0]) == 1 and n_tap_hooks(model, layers[1]) == 1
    ids = tok.encode(PROMPT, add_special_tokens=False)
    for name, art in (("token_mean", mean), ("ema_max", ema)):
        st = offline_call_states(model, ids, comp.tokens, art.layer)
        off = tapmod.StreamingTap(art).update(st, offline_channel(comp.tokens, {stop}), 0)
        assert tap.score_of_call(0, name) == pytest.approx(off, abs=1e-5)
    assert [d["head"] for d in tap.decisions] == ["token_mean", "ema_max"]
    t = comp.timings
    assert set(t["score_s_by_head"]) == {"token_mean", "ema_max"} and all(v > 0 for v in t["score_s_by_head"].values())
    assert t["score_s"] == pytest.approx(sum(t["score_s_by_head"].values()))
    assert t["tap_s"] == pytest.approx(t["capture_s"] + t["score_s"])
    with pytest.raises(ValueError):
        tap.score_of_call(0)


def test_tap_buffer_grows_beyond_its_initial_size(model, tok, fmt, monkeypatch):
    monkeypatch.setattr(lt, "INITIAL_ROWS", 8)
    art = tm.random_artifact(LAYER, "token_mean", seed=7)
    tap = lt.LiveProbeTap(art, device="cpu")
    body = tok.encode(" the user wants me to look at both files now and then the tests again", add_special_tokens=False)
    assert len(body) > 16
    c = make_client(model, tok, fmt, tap, logits_processor=forced(tok, [body]))
    comp = c.complete(PROMPT, max_tokens=None, seed=0, sampling=SP)       # no cap: the engine's context minus the prompt
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    off = tapmod.StreamingTap(art).update(offline_call_states(model, tok.encode(PROMPT, add_special_tokens=False), comp.tokens, LAYER),
                                          offline_channel(comp.tokens, {stop}), 0)
    assert tap.decisions[0]["n_tokens"] == len(body) and tap.decisions[0]["score"] == pytest.approx(off, abs=1e-5)
    assert tap.bufs[LAYER].shape[0] >= len(body)


def test_max_tokens_none_means_the_rest_of_the_context(model, tok, fmt):
    c = lc.LiveHFClient(model, tok, fmt, device="cpu", max_context_tokens=20)
    n = len(tok.encode(PROMPT, add_special_tokens=False))
    comp = c.complete(PROMPT, max_tokens=None, seed=0, sampling={"temperature": 1.0, "top_k": 0, "top_p": 1.0})
    assert len(comp.tokens) <= 20 - n or comp.stop_type == "eos"
    tight = lc.LiveHFClient(model, tok, fmt, device="cpu", max_context_tokens=n)
    with pytest.raises(ValueError):
        tight.complete(PROMPT, max_tokens=None, seed=0, sampling=SP)
