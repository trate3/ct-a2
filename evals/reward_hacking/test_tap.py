"""Streaming tap: EMA semantics, offline/online parity, artifact identity."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from evals.reward_hacking import activations as act
from evals.reward_hacking import tap as T
from evals.reward_hacking import tiny_qwen as tq

LAYER = 2


def make_art(d, seed=0, **kw):
    rng = np.random.default_rng(seed)
    base = dict(layer=LAYER, w=rng.normal(size=d), b=0.3, mu=rng.normal(size=d), sigma=rng.uniform(0.5, 2, d),
                alpha=0.2, threshold=0.1, model_id="m", model_revision="rev", tokenizer_revision="trev",
                method="test")
    base.update(kw)
    return T.TapArtifact(**base)


@pytest.fixture(scope="module")
def stack():
    tok = tq.tiny_tokenizer()
    model = tq.tiny_model(len(tok))
    traj = tq.synthetic_trajectory(tok, n_calls=5, noncanonical_call=2)
    calls = act.reconstruct(traj, tok)
    ex = act.extract(model, tok, traj, [LAYER])
    return tok, model, traj, calls, ex


def test_ema_scan_matches_definition_and_is_chunk_invariant():
    rng = np.random.default_rng(1)
    s = rng.normal(size=1000)
    a = 0.07
    e, ref = 0.5, []
    for x in s:
        e = a * x + (1 - a) * e
        ref.append(e)
    np.testing.assert_array_equal(T.ema_scan(s, a, 0.5), ref)
    parts, ep = [], 0.5
    for ch in np.array_split(s, 7):
        o = T.ema_scan(ch, a, ep)
        ep = o[-1]
        parts.append(o)
    np.testing.assert_array_equal(np.concatenate(parts), ref)


def test_initialization_and_max_and_strict_threshold():
    art = make_art(4, mu=None, sigma=None, w=np.array([1.0, 0, 0, 0]), b=0.0, alpha=0.5)
    h = np.zeros((3, 4))
    h[:, 0] = [2.0, 0.0, 4.0]
    ch = np.full(3, act.VISIBLE, np.int8)
    tap = T.StreamingTap(art)
    S = tap.update(h, ch, 0)
    # first-init: e = [2, 1, 2.5]; max 2.5
    assert S == pytest.approx(2.5)
    z = T.StreamingTap(T.TapArtifact(**{**art.__dict__, "ema_init": "zero"}))
    assert z.update(h, ch, 0) == pytest.approx(max(1.0, 0.5, 2.25))
    assert T.StreamingTap(art.with_threshold(2.5)).update(h, ch, 0) == 2.5
    t2 = T.StreamingTap(art.with_threshold(2.5)); t2.update(h, ch, 0)
    assert t2.predict is False                                    # strict >
    t3 = T.StreamingTap(art.with_threshold(2.4999)); t3.update(h, ch, 0)
    assert t3.predict is True


def test_token_mean_pooling_matches_offline_and_streaming():
    art = make_art(2, w=np.array([1.0, 0.0]), mu=None, sigma=None, b=0.5, pooling="token_mean")
    h = np.array([[1.0, 0.0], [3.0, 0.0], [-1.0, 0.0]])
    ch = np.full(3, act.VISIBLE, np.int8)
    tap = T.StreamingTap(art)
    assert tap.update(h[:2], ch[:2], 0) == pytest.approx(2.5)
    assert tap.update(h[2:], ch[2:], 1) == pytest.approx(1.5)
    assert tap.n_scored == 3
    assert art.to_dict()["pooling"] == "token_mean"


def test_call_scoped_ema_resets_each_invocation_and_legacy_remains_rollout_scoped():
    art = make_art(1, w=np.ones(1), mu=None, sigma=None, b=0.0, alpha=0.5, ema_scope="call")
    ex = act.Extracted(np.array([0, 1, 1], np.int32), np.zeros(3, np.int32), np.zeros(3, np.int32),
                       np.full(3, act.VISIBLE, np.int8), {LAYER: np.array([[-100.0], [-50.0], [-40.0]], np.float32)}, {})
    # Call reset: scores are [-100] and [-50,-45], so max over calls is -45.
    off = T.score_extracted(art, ex)
    assert off.S == pytest.approx(-45.0)
    stream = T.StreamingTap(art)
    stream.update(ex.hidden[LAYER][:1], ex.channel[:1], 0)
    stream.update(ex.hidden[LAYER][1:2], ex.channel[1:2], 1)
    stream.update(ex.hidden[LAYER][2:], ex.channel[2:], 1)  # second chunk of same invocation
    assert stream.S == pytest.approx(off.S)
    # Old artifacts omit ema_scope and retain continuous rollout EMA semantics.
    legacy = T.TapArtifact(**{**art.__dict__, "ema_scope": "rollout"})
    assert T.score_extracted(legacy, ex).S == pytest.approx(-57.5)


def test_channel_mask_skips_end_and_unscored_calls_carry_score():
    art = make_art(3, mu=None, sigma=None, w=np.array([1.0, 0, 0]), b=0.0, alpha=1.0, channels=("tool_call",))
    tap = T.StreamingTap(art)
    h = np.array([[5.0, 0, 0], [1.0, 0, 0], [9.0, 0, 0]])
    assert tap.update(h, np.array([act.VISIBLE, act.TOOL_CALL, act.END], np.int8), 0) == 1.0
    assert tap.update(h, np.array([act.VISIBLE] * 3, np.int8), 1) == 1.0       # nothing scored: carried
    assert tap.call_scores == {0: 1.0, 1: 1.0}
    fresh = T.StreamingTap(art)
    assert fresh.update(h, np.array([act.VISIBLE] * 3, np.int8), 0) is None
    assert fresh.predict is False


def test_offline_online_parity_across_calls_and_resets(stack):
    tok, model, traj, calls, ex = stack
    d = ex.hidden[LAYER].shape[1]
    for kw in ({}, {"ema_init": "zero", "alpha": 0.05}, {"channels": ("tool_call",), "alpha": 1.0}):
        art = make_art(d, **kw)
        off = T.score_extracted(art, ex)
        # (a) online scorer fed stored states call by call
        tap = T.StreamingTap(art)
        for c in sorted(set(ex.call_index.tolist())):
            m = ex.call_index == c
            tap.update(ex.hidden[LAYER][m], ex.channel[m], c)
        assert tap.S == pytest.approx(off.S, abs=1e-9) and tap.n_scored == off.n_scored
        assert tap.call_scores.keys() == off.call_scores.keys()
        for c, v in off.call_scores.items():
            assert tap.call_scores[c] == pytest.approx(v, abs=1e-9)
        # (b) the true online path: teacher-forced prefill per call, KV reused across calls
        cx = act.CallExtractor(model, [LAYER])
        on = T.score_online(art, cx, calls)
        cx.close()
        assert on.S == pytest.approx(off.S, abs=1e-4)                  # model recompute noise only
        # reset: re-running after reset reproduces a fresh tap exactly (nothing carried between rollouts)
        first = T.StreamingTap(art)
        for c in sorted(set(ex.call_index.tolist())):
            m = ex.call_index == c
            first.update(ex.hidden[LAYER][m], ex.channel[m], c)
        first.reset()
        assert first.S is None and first.e is None and first.call_scores == {}
        for c in sorted(set(ex.call_index.tolist())):
            m = ex.call_index == c
            first.update(ex.hidden[LAYER][m], ex.channel[m], c)
        assert first.S == tap.S and first.call_scores == tap.call_scores


def test_carry_across_calls_equals_single_pass(stack):
    _, _, _, _, ex = stack
    art = make_art(ex.hidden[LAYER].shape[1])
    whole = T.StreamingTap(art)
    whole.update(ex.hidden[LAYER], ex.channel, 0)                       # one big "call"
    split = T.StreamingTap(art)
    for c in sorted(set(ex.call_index.tolist())):
        m = ex.call_index == c
        split.update(ex.hidden[LAYER][m], ex.channel[m], c)
    assert split.S == pytest.approx(whole.S, abs=1e-12) and split.e == pytest.approx(whole.e, abs=1e-12)


def test_torch_input_matches_numpy(stack):
    _, _, _, _, ex = stack
    art = make_art(ex.hidden[LAYER].shape[1])
    h = ex.hidden[LAYER]
    np.testing.assert_allclose(art.token_scores(torch.from_numpy(h)), art.token_scores(h), atol=1e-6)


def test_max_call_restricts_to_pre_violation_calls(stack):
    _, _, _, _, ex = stack
    art = make_art(ex.hidden[LAYER].shape[1])
    full = T.score_extracted(art, ex)
    pre = T.score_extracted(art, ex, max_call=2)
    assert set(pre.call_scores) == {0, 1}
    assert pre.call_scores[1] == full.call_scores[1]                     # causal: unchanged by later calls
    assert pre.S == full.call_scores[1]


def test_decision_point_scores_track_calls(stack):
    _, _, traj, _, ex = stack
    art = make_art(ex.hidden[LAYER].shape[1])
    off = T.score_extracted(art, ex)
    pts = T.decision_point_scores(traj, off.call_scores)
    assert [p["call_index"] for p in pts] == [0, 1, 2, 3, 4]
    assert [p["score"] for p in pts] == [off.call_scores[c] for c in range(5)]
    vals = [p["score"] for p in pts]
    assert vals == sorted(vals)                                          # max-so-far is monotone


def test_artifact_roundtrip_and_digest(tmp_path):
    a = make_art(16, threshold=float("-inf"), extra={"grid": [1, 2], "x": 1.5})
    p = tmp_path / "a.json"
    d = a.save(p)
    b = T.TapArtifact.load(p)
    assert b.digest == a.digest == d
    for f in ("w", "mu", "sigma"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    assert b.threshold == float("-inf") and b.channels == a.channels and b.extra == a.extra
    assert make_art(16, seed=0, threshold=float("-inf"), extra={"grid": [1, 2], "x": 1.5}).digest == a.digest
    assert a.with_threshold(0.0).digest != a.digest
    assert make_art(16, model_revision="other", threshold=float("-inf"), extra=a.extra).digest != a.digest
    assert make_art(16, channels=("visible", "reasoning", "tool_call")).digest == make_art(16).digest   # order-free
    p.write_text(p.read_text().replace('"alpha": 0.2', '"alpha": 0.3'))
    with pytest.raises(ValueError, match="digest"):
        T.TapArtifact.load(p)


def test_artifact_digest_golden():
    """Pins the canonical serialization; change ARTIFACT_VERSION if this must move."""
    a = T.TapArtifact(layer=3, w=np.arange(4.0), b=0.5, mu=np.zeros(4), sigma=np.ones(4), alpha=0.25,
                      threshold=1.0, model_id="m", model_revision="r", tokenizer_revision="t", method="dom")
    assert a.digest == GOLDEN


GOLDEN = "a0c78dcd85918a530a373cc61676996f4c35715a0c59ad380f93cd57e114b631"
