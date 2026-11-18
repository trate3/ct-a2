"""The optional `device` of the probe fits: CPU results unchanged, CUDA matches CPU, no silent fallback."""

from __future__ import annotations

import numpy as np
import pytest

from evals.reward_hacking import probe as P
from evals.reward_hacking.test_probe import LAYER, make_split
from evals_runner import rh_probe as RP

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="torch.cuda.is_available() is False: no CUDA device on this machine")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    train, val = make_split("tr", "train", 10, rng), make_split("va", "val", 6, rng)
    norm, pre = P.train_pass(train, LAYER, P.act.DEFAULT_CHANNELS, 48, 0)
    return train, val, norm, pre


def same(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    return type(a) is type(b) and a == b


def fits(train, norm, pre, **dev):
    """Every fit entry point once; `dev` is {} (the default path) or {"device": ...}."""
    tok = P.fit_logistic_token(train, LAYER, norm=norm, l2=1e-2, max_tokens_per_rollout=48, seed=0, pre=pre,
                               class_balance=True, **dev)
    mean = P.fit_logistic_mean(train, LAYER, l2=1e-2, class_balance=True, **dev)
    rng = np.random.default_rng(1)
    mean_x = P.fit_logistic_mean_X(rng.normal(size=(40, 16)), rng.integers(0, 2, 40), l2=1e-2, **dev)
    traj = P.fit_logistic_trajectory(train, LAYER, norm=norm, alpha=0.2, init=tok, epochs=2, lr=5e-3, seed=0,
                                     max_tokens=120, **dev)
    lse = P.fit_logistic_trajectory(train, LAYER, norm=norm, alpha=0.2, pool="lse", epochs=2, seed=0,
                                    max_tokens=120, **dev)
    return {"tok": tok, "mean": mean, "mean_x": mean_x, "traj": traj, "lse": lse}


# 1. explicit device="cpu" is the existing default path, exactly

def test_explicit_cpu_is_identical_to_default(data):
    train, val, norm, pre = data
    default, explicit = fits(train, norm, pre), fits(train, norm, pre, device="cpu")
    for k in default:
        assert same(default[k], explicit[k]), k
    assert all(isinstance(v["w"], np.ndarray) and isinstance(v["b"], float) for v in explicit.values())
    assert explicit["tok"]["w"].dtype == np.float64 and explicit["traj"]["w"].dtype == np.float64


@pytest.mark.parametrize("method,kw", [("logistic_token", {}), ("logistic_trajectory", {"epochs": 2, "max_tokens": 80})])
def test_select_probe_explicit_cpu_is_identical_and_device_not_recorded(data, method, kw):
    train, val, _, _ = data
    args = dict(layers=[LAYER], method=method, alphas=(0.05, 0.5), l2s=(1e-2,), target_fpr=0.2,
                max_tokens_per_rollout=48, **kw)
    a, b = P.select_probe(train, val, **args), P.select_probe(train, val, device="cpu", **args)
    assert same(a.w, b.w) and a.b == b.b and a.threshold == b.threshold and a.alpha == b.alpha
    assert a.extra == b.extra and "device" not in a.extra["fit_kw"]


def test_ema_torch_and_loss_run_on_the_input_device():
    s = torch.from_numpy(np.random.default_rng(2).normal(size=700))
    assert P._ema_torch(s, 0.3, s[0], block=128).device == s.device


# 2. CUDA agrees with CPU

def _problem(n=300, d=16, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (rng.random(n) < 0.4).astype(np.float64)
    X[y == 1, :3] += 1.0
    wt = np.full(n, 1.0 / n)
    return X, y, wt, rng.normal(size=d) * 0.1, 0.2


@needs_cuda
def test_objective_and_gradient_match_cpu_in_float64():
    X, y, wt, w0, b0 = _problem()
    out = {}
    for dev in ("cpu", "cuda"):
        Xt, yt, wtt = (torch.from_numpy(a).to(dev) for a in (X, y, wt))
        w = torch.tensor(w0, dtype=torch.float64, device=dev, requires_grad=True)
        b = torch.tensor([b0], dtype=torch.float64, device=dev, requires_grad=True)
        loss = P._logistic_loss(Xt, yt, wtt, w, b, 1e-2)
        loss.backward()
        out[dev] = (loss.item(), w.grad.cpu().numpy(), b.grad.cpu().numpy())
    assert out["cuda"][0] == pytest.approx(out["cpu"][0], rel=1e-9)
    np.testing.assert_allclose(out["cuda"][1], out["cpu"][1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(out["cuda"][2], out["cpu"][2], rtol=1e-9, atol=1e-12)


@needs_cuda
def test_ema_matches_cpu_in_float64():
    s = np.random.default_rng(4).normal(size=1300)
    cpu = P._ema_torch(torch.from_numpy(s), 0.05, torch.tensor(s[0]), block=256).numpy()
    gpu = P._ema_torch(torch.from_numpy(s).cuda(), 0.05, torch.tensor(s[0]).cuda(), block=256).cpu().numpy()
    np.testing.assert_allclose(gpu, cpu, rtol=1e-9, atol=1e-12)


@needs_cuda
def test_fitted_weights_and_scores_match_cpu(data):
    train, val, norm, pre = data
    cpu, gpu = fits(train, norm, pre, device="cpu"), fits(train, norm, pre, device="cuda")
    # The trajectory fit is float64 throughout: 1e-5 as asked. The token and mean fits are float32 by design
    # (dtypes are kept), so L-BFGS on a different device can differ at float32 level: 1e-3 there.
    tol = {"traj": 1e-5, "lse": 1e-5}
    X = np.random.default_rng(5).normal(size=(200, 16))
    for k in cpu:
        t = tol.get(k, 1e-3)
        np.testing.assert_allclose(gpu[k]["w"], cpu[k]["w"], rtol=t, atol=t, err_msg=k)
        assert gpu[k]["b"] == pytest.approx(cpu[k]["b"], rel=t, abs=t), k
        np.testing.assert_allclose(X @ gpu[k]["w"] + gpu[k]["b"], X @ cpu[k]["w"] + cpu[k]["b"], rtol=t, atol=t,
                                   err_msg=k)
        assert isinstance(gpu[k]["w"], np.ndarray) and gpu[k]["w"].dtype == np.float64 and isinstance(gpu[k]["b"], float)


# 3. no silent fallback

@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_unavailable_cuda_raises_not_falls_back(no_cuda, data):
    train, val, norm, pre = data
    for call in (
        lambda: P.fit_logistic_token(train, LAYER, norm=norm, pre=pre, max_tokens_per_rollout=48, device="cuda"),
        lambda: P.fit_logistic_mean(train, LAYER, device="cuda"),
        lambda: P.fit_logistic_mean_X(np.zeros((4, 3)), np.array([0, 1, 0, 1]), device="cuda:0"),
        lambda: P.fit_logistic_trajectory(train, LAYER, norm=norm, alpha=0.2, epochs=1, device="cuda"),
        lambda: P.select_probe(train, val, layers=[LAYER], device="cuda"),
    ):
        with pytest.raises(RuntimeError, match="CUDA is not available.*refusing to fall back"):
            call()


@pytest.mark.parametrize("bad", ["mps", "xpu", "gpu", "cuda:x", ""])
def test_unsupported_or_malformed_device_raises(bad):
    with pytest.raises(ValueError, match="device"):
        P.resolve_device(bad)


@needs_cuda
def test_out_of_range_cuda_index_raises():
    with pytest.raises(RuntimeError, match="only .* CUDA device"):
        P.resolve_device(f"cuda:{torch.cuda.device_count()}")


def test_resolve_device_cpu():
    assert P.resolve_device("cpu").type == "cpu" and P.resolve_device().type == "cpu"


def test_train_cv_cli_has_device_option_and_fails_before_reading_inputs(no_cuda, tmp_path):
    ap = __import__("argparse").ArgumentParser()
    RP.configure_parser(ap)
    base = ["train-cv", "--acts", str(tmp_path / "none"), "--rollouts", str(tmp_path / "none"), "--out", str(tmp_path / "o")]
    assert ap.parse_args(base).device == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        RP.run(ap.parse_args(base + ["--device", "cuda"]))
    assert not (tmp_path / "o").exists()
