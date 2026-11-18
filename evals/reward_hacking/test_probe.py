"""Probe fitting and selection on synthetic separable data (no model needed)."""
from __future__ import annotations

import numpy as np
import pytest

from evals.reward_hacking import activations as act
from evals.reward_hacking import metrics
from evals.reward_hacking import probe as P
from evals.reward_hacking import tap as T

D, LAYER = 16, 2
U = np.zeros(D)
U[:3] = [1, 1, 1]
U /= np.linalg.norm(U)


def make_sample(rid, task, label, partition, rng, n=300, frac=0.5, strength=5.0, calls=4):
    h = rng.normal(size=(n, D)) * rng.uniform(0.5, 2, D) + rng.normal(size=D)
    if label:                                        # weak labels: only a fraction of tokens carry the signal
        sel = rng.random(n) < frac
        h[sel] += strength * U
    ci = np.sort(rng.integers(0, calls, n)).astype(np.int32)
    ch = np.full(n, act.REASONING, np.int8)
    ch[::17] = act.END
    ex = act.Extracted(ci, np.zeros(n, np.int32), np.zeros(n, np.int32), ch, {LAYER: h.astype(np.float32)}, {})
    return P.RolloutSample(rid, task, label, partition, None, None, ex)


def make_split(prefix, partition, n_tasks, rng, per=2, pos_frac=0.4):
    out = []
    for t in range(n_tasks):
        for r in range(per):
            y = int(rng.random() < pos_frac)
            out.append(make_sample(f"{prefix}{t}_{r}", f"{prefix}{t}", y, partition, rng,
                                   n=int(rng.integers(100, 600))))
    return out


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return make_split("tr", "train", 30, rng), make_split("va", "val", 16, rng)


def test_dom_recovers_direction_and_is_unit_norm_with_midpoint_bias():
    rng = np.random.default_rng(1)
    neg = rng.normal(size=(500, D))
    pos = rng.normal(size=(500, D)) + 3 * U
    r = P.fit_dom(pos, neg, standardize=False)
    assert np.linalg.norm(r["w"]) == pytest.approx(1.0)
    assert float(r["w"] @ U) > 0.95
    sp, sn = pos @ r["w"] + r["b"], neg @ r["w"] + r["b"]
    assert sp.mean() > 0 > sn.mean() and sp.mean() == pytest.approx(-sn.mean(), rel=0.2)
    assert metrics.auroc([1] * 500 + [0] * 500, np.r_[sp, sn]) > 0.97
    r2 = P.fit_dom(pos, neg, standardize=True)
    assert r2["mu"].shape == (D,) and float(r2["w"] @ U / np.linalg.norm(r2["w"])) > 0.9
    with pytest.raises(ValueError):
        P.fit_dom(pos[:5], neg)


def test_normalizer_weights_rollouts_equally(data):
    train, _ = data
    n = P.fit_normalizer(train, LAYER)
    per = [np.asarray(s.data.hidden[LAYER][P._eligible(s.data, act.DEFAULT_CHANNELS)], np.float64).mean(0) for s in train]
    np.testing.assert_allclose(n.mu, np.mean(per, 0), atol=1e-9)
    assert (n.sigma >= 1e-6).all()


def test_terminal_cutoff_is_applied_to_normalizer_and_train_pass():
    def sample(rid, label, values):
        h = np.zeros((4, D), np.float32)
        h[:, 0] = values
        ex = act.Extracted(np.array([0, 0, 1, 1], np.int32), np.arange(4, dtype=np.int32),
                           np.zeros(4, np.int32), np.full(4, act.REASONING, np.int8), {LAYER: h}, {})
        return P.RolloutSample(rid, rid, label, "train", 1, None, ex)

    samples = [sample("neg", 0, [-2, -1, -1000, -900]), sample("pos", 1, [1, 2, 900, 1000])]
    n = P.fit_normalizer(samples, LAYER)
    assert n.mu[0] == pytest.approx(0.0)  # only call 0 contributes
    assert P._eligible(samples[0].data, act.DEFAULT_CHANNELS, max_call=1).tolist() == [0, 1]
    n2, pre = P.train_pass(samples, LAYER, act.DEFAULT_CHANNELS, max_tokens=100, seed=0)
    assert n2.mu[0] == pytest.approx(0.0)
    assert pre[1] == [2, 2]


def test_terminal_cutoff_is_applied_to_mean_fit_and_validation_score():
    def sample(rid, label, terminal, later):
        h = np.zeros((4, D), np.float32)
        h[:2, 0] = terminal
        h[2:, 0] = later
        ex = act.Extracted(np.array([0, 0, 1, 1], np.int32), np.arange(4, dtype=np.int32),
                           np.zeros(4, np.int32), np.full(4, act.REASONING, np.int8), {LAYER: h}, {})
        return P.RolloutSample(rid, rid, label, "train", 1, None, ex)

    train = [sample("neg", 0, -2, 500), sample("pos", 1, 2, -500)]
    fit = P.fit_logistic_mean(train, LAYER, max_iter=20)
    assert fit["w"][0] > 0  # post-submission tokens carry the opposite, excluded signal
    val = [P.RolloutSample("v", "v", 1, "val", 1, None,
                           act.Extracted(np.array([0, 0, 1, 1], np.int32), np.arange(4, dtype=np.int32),
                                         np.zeros(4, np.int32), np.full(4, act.REASONING, np.int8),
                                         {LAYER: np.array([[2] + [0] * (D - 1), [2] + [0] * (D - 1),
                                                           [-100] + [0] * (D - 1), [-100] + [0] * (D - 1)],
                                                          np.float32)}, {}))]
    art = T.TapArtifact(layer=LAYER, w=np.r_[1.0, np.zeros(D - 1)], b=0.0, mu=None, sigma=None,
                        alpha=1.0, threshold=0.0)
    assert P.score_samples(art, val)[0] == pytest.approx(2.0)


def test_rollout_weights():
    w = P._rollout_weights(np.array([1, 0, 0, 0]), False)
    assert w.sum() == pytest.approx(1) and np.allclose(w, 0.25)
    w = P._rollout_weights(np.array([1, 0, 0, 0]), True)
    assert w[0] == pytest.approx(0.5) and w.sum() == pytest.approx(1)


def test_token_logistic_equal_rollout_weight_and_separates(data):
    train, val = data
    norm = P.fit_normalizer(train, LAYER)
    fit = P.fit_logistic_token(train, LAYER, norm=norm, l2=1e-2, max_tokens_per_rollout=64, seed=0)
    assert float(fit["w"] @ U) / np.linalg.norm(fit["w"]) > 0.6          # finds the signal direction
    art = T.TapArtifact(layer=LAYER, **fit, alpha=0.2, threshold=0.0)
    assert P.val_auroc(art, val) > 0.9
    # determinism: same seed, same fit
    fit2 = P.fit_logistic_token(train, LAYER, norm=norm, l2=1e-2, max_tokens_per_rollout=64, seed=0)
    np.testing.assert_allclose(fit["w"], fit2["w"])


def test_mean_pooled_logistic_fits_one_vector_per_rollout(data):
    train, val = data
    fit = P.fit_logistic_mean(train, LAYER, l2=1e-2)
    assert float(fit["w"] @ U) / np.linalg.norm(fit["w"]) > 0.6
    art = T.TapArtifact(layer=LAYER, **fit, alpha=1.0, threshold=0.0, pooling="token_mean")
    assert P.val_auroc(art, val) > 0.9


def test_length_matched_validation_threshold_is_deterministic_with_ties_and_reuse():
    def smp(rid, label, length, score):
        h = np.zeros((length, D), np.float32)
        h[:, 0] = score
        ex = act.Extracted(np.zeros(length, np.int32), np.arange(length, dtype=np.int32),
                           np.zeros(length, np.int32), np.full(length, act.REASONING, np.int8),
                           {LAYER: h}, {})
        return P.RolloutSample(rid, rid, label, "val", data=ex)
    val = [smp("p10", 1, 10, 0), smp("p11", 1, 11, 0), smp("p100", 1, 100, 0),
           smp("a", 0, 10, 1), smp("b", 0, 10, 9), smp("z", 0, 95, 3)]
    art = T.TapArtifact(layer=LAYER, w=np.r_[1.0, np.zeros(D - 1)], b=0.0, mu=None, sigma=None,
                        alpha=1.0, threshold=0.0)
    rep = P.length_matched_threshold(art, val, target_fpr=0.34)
    assert rep["threshold"] == pytest.approx(1.0)
    assert rep["n_val_positive_matched"] == rep["n_val_negative_matches"] == 3
    assert rep["n_unique_val_negatives_matched"] == 2
    assert [m["negative_rollout_id"] for m in rep["matches"]] == ["a", "z", "a"]


def test_equal_weight_means_long_rollout_does_not_dominate():
    rng = np.random.default_rng(3)
    long_neg = make_sample("n_long", "a", 0, "train", rng, n=5000)
    short_neg = make_sample("n_short", "b", 0, "train", rng, n=20)
    pos = make_sample("p", "c", 1, "train", rng, n=20, frac=1.0)
    X, sizes, labels = P._gather_tokens([long_neg, short_neg, pos], LAYER, P.fit_normalizer([long_neg, short_neg, pos], LAYER),
                                        act.DEFAULT_CHANNELS, 4096, 0)
    assert sizes[0] == 4096 > sizes[1]                                    # subsampled cap
    w = P._rollout_weights(labels, False)
    assert w.sum() == pytest.approx(1) and np.allclose(w, 1 / 3)


def test_ema_torch_matches_numpy_scan():
    import torch
    rng = np.random.default_rng(2)
    s = rng.normal(size=1500)
    for a in (0.01, 0.3, 1.0):
        got = P._ema_torch(torch.from_numpy(s), a, torch.tensor(s[0]), block=128).numpy()
        np.testing.assert_allclose(got, T.ema_scan(s, a, s[0]), atol=1e-10)


def test_trajectory_objective_trains_pooled_scores(data):
    train, val = data
    norm = P.fit_normalizer(train, LAYER)
    init = P.fit_logistic_token(train, LAYER, norm=norm, l2=1e-2, max_tokens_per_rollout=32, seed=0)
    for pool in ("max", "lse"):
        fit = P.fit_logistic_trajectory(train, LAYER, norm=norm, alpha=0.2, init=init, epochs=5, lr=5e-3,
                                        pool=pool, seed=0, max_tokens=200)
        art = T.TapArtifact(layer=LAYER, **fit, alpha=0.2, threshold=0.0)
        assert P.val_auroc(art, val) > 0.85
    cold = P.fit_logistic_trajectory(train, LAYER, norm=norm, alpha=0.2, epochs=5, lr=5e-2, seed=0, max_tokens=200)
    assert float(cold["w"] @ U) > 0


@pytest.mark.parametrize("method", ["dom", "logistic_token", "logistic_trajectory"])
def test_select_probe_end_to_end(data, method):
    train, val = data
    rng = np.random.default_rng(5)
    kw = {}
    if method == "dom":
        kw["dom_pairs"] = {LAYER: (rng.normal(size=(200, D)) + 3 * U, rng.normal(size=(200, D)))}
    if method == "logistic_trajectory":
        kw.update(epochs=2, max_tokens=150)
    art = P.select_probe(train, val, layers=[LAYER], method=method, alphas=(0.05, 0.5), l2s=(1e-2,),
                         target_fpr=0.2, model_id="tiny", max_tokens_per_rollout=64, **kw)
    assert art.method == method and art.extra["n_train"] == len(train) and len(art.extra["grid"]) >= 2
    scores = P.score_samples(art, val)
    labels = [s.label for s in val]
    neg = scores[np.array(labels) == 0]
    assert (neg > art.threshold).mean() <= 0.2 + 1e-9                    # FPR constraint on val
    assert art.extra["val_auroc"] > 0.8
    assert T.TapArtifact.from_json(art.to_json()).digest == art.digest


def test_select_probe_guards_against_leaks(data):
    train, val = data
    with pytest.raises(ValueError, match="partitions"):
        P.select_probe(val, train, layers=[LAYER])
    import dataclasses
    leaky = [dataclasses.replace(val[0], partition="val", task_id=train[0].task_id)] + list(val[1:])
    with pytest.raises(ValueError, match="share tasks"):
        P.select_probe(train, leaky, layers=[LAYER])


def test_splits_file_and_build_samples(tmp_path):
    splits = P.load_splits()
    assert set(splits.values()) == {"train", "val", "test"} and len(splits) == 103


def test_build_samples_labels_and_partitions(tmp_path):
    from evals.reward_hacking import trajectory as tj
    rng = np.random.default_rng(0)
    edir, tdir = tmp_path / "e", tmp_path / "t"
    edir.mkdir(), tdir.mkdir()
    splits = {"tA": "train", "tB": "val"}
    for rid, task, passed in [("r1", "tA", True), ("r2", "tB", False), ("r3", "tZ", True)]:
        t = tj.Trajectory(meta={"task_id": task, "rollout_id": rid})
        t.add(tj.SUBMISSION, {"attempt": 1, "func_py": "", "test_py": "", "answer": ""}, 0)
        t.add(tj.EVALUATION, {"attempt": 1, "passed": passed, "explanation": ""}, 0)
        (tdir / f"{rid}.json").write_text(t.to_json())
        act.save_extracted(edir / f"{rid}.npz", make_sample(rid, task, 0, "", rng, n=20).data)
    tr = P.build_samples(edir, tdir, splits, partition="train")
    assert [(s.rollout_id, s.label) for s in tr] == [("r1", 1)]
    assert tr[0].max_call == 1
    allv = P.build_samples(edir, tdir, splits)
    assert {s.rollout_id: (s.label, s.partition) for s in allv} == {"r1": (1, "train"), "r2": (0, "val")}   # r3: unsplit task dropped
    assert allv[0].load([LAYER]).hidden[LAYER].shape[1] == D


def test_train_pass_cache_reproduces_uncached_fit():
    """train_pass + pre= gives the exact normalizer and logistic fit of the per-l2 re-reading path."""
    rng = np.random.default_rng(3)
    samples = []
    for i in range(6):
        n = 300 + 50 * i
        hid = {7: rng.normal(size=(n, 8)).astype(np.float32) + (i % 2)}
        ex = act.Extracted(np.zeros(n, np.int32), np.arange(n, dtype=np.int32), np.zeros(n, np.int32),
                           np.full(n, act.REASONING, np.int8), hid, {})
        samples.append(P.RolloutSample(f"r{i}", f"t{i}", i % 2, "train", data=ex))
    norm_a = P.fit_normalizer(samples, 7)
    norm_b, pre = P.train_pass(samples, 7, act.DEFAULT_CHANNELS, 100, 0)
    np.testing.assert_array_equal(norm_a.mu, norm_b.mu)
    np.testing.assert_array_equal(norm_a.sigma, norm_b.sigma)
    fa = P.fit_logistic_token(samples, 7, norm=norm_a, max_tokens_per_rollout=100)
    fb = P.fit_logistic_token(samples, 7, norm=norm_b, max_tokens_per_rollout=100, pre=pre)
    np.testing.assert_array_equal(fa["w"], fb["w"])
    assert fa["b"] == fb["b"]


def test_load_ordered_keeps_order_under_concurrency():
    import time

    class S:
        def __init__(self, i):
            self.i = i

        def load(self, layers):
            time.sleep(0.01 * ((7 * self.i) % 5))      # finish out of order
            return self.i

    got = [(s.i, ex) for s, ex in P.load_ordered([S(i) for i in range(40)], [0], workers=6)]
    assert got == [(i, i) for i in range(40)]


def test_fast_selection_scores_match_tap_reference():
    rng = np.random.default_rng(5)
    d, layer = 6, 3
    p = {"w": rng.normal(size=d), "b": 0.3, "mu": rng.normal(size=d), "sigma": rng.uniform(0.5, 2, size=d)}
    samples = []
    for i in range(5):
        n = 1500 + 37 * i
        ex = act.Extracted(np.repeat(np.arange(3), n // 3 + 1)[:n].astype(np.int32), np.arange(n, dtype=np.int32),
                           np.zeros(n, np.int32), rng.choice([act.REASONING, act.VISIBLE, act.END], n).astype(np.int8),
                           {layer: rng.normal(size=(n, d)).astype(np.float32)}, {})
        samples.append(P.RolloutSample(f"r{i}", f"t{i}", i % 2, "val", max_call=2 if i == 3 else None, data=ex))
    sc = P.val_token_scores(p, samples, layer, act.DEFAULT_CHANNELS)
    for alpha in (0.02, 0.2, 1.0):
        art = T.TapArtifact(layer=layer, w=p["w"], b=p["b"], mu=p["mu"], sigma=p["sigma"], alpha=alpha,
                            threshold=0.0, channels=tuple(act.DEFAULT_CHANNELS), ema_init="first")
        ref = P.score_samples(art, samples)
        fast = [float(P.ema_blocked(s, alpha, s[0]).max()) for s, _calls in sc]
        np.testing.assert_allclose(fast, ref, rtol=1e-9, atol=1e-9)
        call_art = T.TapArtifact(layer=layer, w=p["w"], b=p["b"], mu=p["mu"], sigma=p["sigma"], alpha=alpha,
                                 threshold=0.0, channels=tuple(act.DEFAULT_CHANNELS), ema_init="first", ema_scope="call")
        fast_call = [P.fast_val_auroc(sc, [s.label for s in samples], alpha, "first", "ema_max", "call")]
        call_scores = P.score_samples(call_art, samples)
        assert fast_call[0] == pytest.approx(metrics.auroc([s.label for s in samples], call_scores))
