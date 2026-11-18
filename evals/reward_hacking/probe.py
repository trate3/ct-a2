"""Probe training and selection for the white-box tap.

Inputs are the per-rollout activation files of `activations` (predictive states of generated
tokens). Everything is fit on the *train* partition of `data/splits.json` (by task_id);
layer, regularization, EMA alpha and the threshold are chosen on *val*; test is never read
here. The output is a `tap.TapArtifact`.

Normalization. Per-dimension standardization, mean/std taken over the eligible generated
tokens of the train rollouts with every rollout given equal total weight (so long rollouts do
not dominate), std floored at 1e-6. (For the DoM probe it can instead be fit on the pooled
paired activations.)

Probes
1. `fit_dom` -- difference of means from paired (minimal-pair) activations, the primary probe
   of the memo's addendum (replicating Bergen et al. 2609.19101). In normalized space
   w = (m_pos - m_neg) / ||m_pos - m_neg||_2 (unit norm, so s is in units of one train std
   along the direction) and b = -w . (m_pos + m_neg) / 2 (s = 0 at the midpoint of the two
   class means). With `standardize=False` the direction is in raw space.
2. `fit_logistic_token` -- weakly supervised token classification: every eligible token of a
   rollout carries the rollout label. This is NOT token ground truth; it is a weak-label
   surrogate. Each rollout gets total weight 1/R (optionally class-balanced), spread
   uniformly over a seeded random subsample of at most `max_tokens_per_rollout` of its
   eligible tokens; L2 penalty `l2` on w (not b), full-batch L-BFGS in torch.
3. `fit_logistic_trajectory` -- trajectory-level objective: the rollout logit is the pooled
   EMA score S = max_t e_t (or a log-sum-exp soft max with temperature `tau`), the loss is
   the rollout-level BCE(S, y) averaged over rollouts (class-balanced optional), same L2,
   alpha fixed per fit and selected on val. Initialized from the token-level solution, Adam,
   rollouts streamed from disk. Exploratory per the memo.

Every fit, validation score, and terminal score is restricted to model calls through the
`labelled_submission` boundary (`RolloutSample.max_call`, exclusive). The boundary call is
included in full because the trajectory records its submission after processing the full turn.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import zlib
from collections.abc import Sequence

import os

import numpy as np

from evals.reward_hacking import activations as act
from evals.reward_hacking import metrics
from evals.reward_hacking import tap as tapmod

SPLITS_PATH = pathlib.Path(__file__).parent / "data" / "splits.json"


def load_splits(path=SPLITS_PATH) -> dict[str, str]:
    """task_id -> 'train' | 'val' | 'test'."""
    return dict(json.loads(pathlib.Path(path).read_text())["assignment"])


@dataclasses.dataclass
class RolloutSample:
    rollout_id: str
    task_id: str
    label: int
    partition: str = ""
    max_call: int | None = None       # exclusive terminal cutoff: eligible calls have call_index < max_call
    path: str | None = None           # .npz from activations.save_extracted
    data: "act.Extracted | None" = None

    def load(self, layers: Sequence[int]) -> "act.Extracted":
        if self.data is not None:
            return self.data
        return act.load_extracted(self.path, layers=list(layers))


def build_samples(extracted_dir, traj_dir, splits: dict[str, str], *, partition: str | None = None,
                  max_call_by_rollout: dict[str, int | None] | None = None) -> list[RolloutSample]:
    """Samples for rollouts that have both an extracted file and a labelled trajectory."""
    from evals.reward_hacking import trajectory as tj
    out = []
    for p in sorted(pathlib.Path(extracted_dir).glob("*.npz")):
        tp = pathlib.Path(traj_dir) / (p.stem + ".json")
        if not tp.exists():
            continue
        traj = tj.Trajectory.from_json(tp.read_text())
        y = tj.label(traj)
        tid = traj.meta["task_id"]
        part = splits.get(tid)
        if y is None or part is None or (partition and part != partition):
            continue
        mc = ((max_call_by_rollout or {}).get(p.stem)
              if max_call_by_rollout is not None and p.stem in max_call_by_rollout
              else tj.labelled_submission_max_call(traj))
        out.append(RolloutSample(p.stem, tid, int(y), part, mc, str(p)))
    return out


# ---------------------------------------------------------------------------
# normalization

@dataclasses.dataclass
class Normalizer:
    mu: np.ndarray
    sigma: np.ndarray

    def __call__(self, h):
        return (np.asarray(h, dtype=np.float64) - self.mu) / self.sigma


def _eligible(ex: "act.Extracted", channels: Sequence[str], max_call: int | None = None) -> np.ndarray:
    codes = [act.CHANNEL_CODES[c] for c in channels]
    keep = np.isin(ex.channel, codes)
    if max_call is not None:
        keep &= ex.call_index < max_call
    return np.nonzero(keep)[0]


def fit_normalizer(samples: Sequence[RolloutSample], layer: int, channels=act.DEFAULT_CHANNELS) -> Normalizer:
    """Per-dim mean/std over eligible tokens, equal total weight per rollout (train rollouts only)."""
    s1 = s2 = None
    R = 0
    for smp in samples:
        ex = smp.load([layer])
        idx = _eligible(ex, channels, smp.max_call)
        if len(idx) == 0:
            continue
        h = np.asarray(ex.hidden[layer][idx], dtype=np.float64)
        m1, m2 = h.mean(0), (h ** 2).mean(0)
        s1 = m1 if s1 is None else s1 + m1
        s2 = m2 if s2 is None else s2 + m2
        R += 1
    if R == 0:
        raise ValueError("no eligible tokens to fit a normalizer")
    mu = s1 / R
    var = np.maximum(s2 / R - mu ** 2, 0.0)
    return Normalizer(mu, np.maximum(np.sqrt(var), 1e-6))


# ---------------------------------------------------------------------------
# (i) difference of means

def fit_dom(pos: np.ndarray, neg: np.ndarray, *, standardize: bool = True,
            norm: Normalizer | None = None) -> dict:
    """DoM probe from paired activations pos[i] / neg[i] (equal counts, one row per pair or token).

    Returns {w, b, mu, sigma}; mu/sigma are None when not standardizing. `norm` overrides the
    default normalizer (fit on the pooled pairs) -- pass a rollout-fit one to match deployment.
    """
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if pos.shape != neg.shape or pos.ndim != 2:
        raise ValueError("pos and neg must be paired arrays of the same [n, d] shape")
    if standardize:
        if norm is None:
            both = np.concatenate([pos, neg])
            norm = Normalizer(both.mean(0), np.maximum(both.std(0), 1e-6))
        pos, neg = norm(pos), norm(neg)
        mu, sigma = norm.mu, norm.sigma
    else:
        mu = sigma = None
    mp, mn = pos.mean(0), neg.mean(0)
    d = mp - mn
    n = np.linalg.norm(d)
    if n == 0:
        raise ValueError("class means coincide")
    w = d / n
    b = -float(w @ (mp + mn) / 2)
    return {"w": w, "b": b, "mu": mu, "sigma": sigma}


# ---------------------------------------------------------------------------
# (ii) logistic probes

def _torch():
    import torch
    return torch


def resolve_device(device: str = "cpu"):
    """The torch device for a fit. "cpu" (the default) is always valid and changes nothing. Any other value must be an
    available CUDA device ("cuda", "cuda:1"); an unavailable or unsupported device raises instead of falling back."""
    torch = _torch()
    try:
        dev = torch.device(device)
    except (RuntimeError, TypeError) as e:
        raise ValueError(f"invalid torch device {device!r}: {e}") from None
    if dev.type == "cpu":
        return dev
    if dev.type != "cuda":
        raise ValueError(f"unsupported device {device!r}: only 'cpu' and CUDA devices are supported")
    if not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is not available (torch {torch.__version__}); "
                           f"refusing to fall back to the CPU")
    if dev.index is not None and dev.index >= torch.cuda.device_count():
        raise RuntimeError(f"device {device!r} requested but only {torch.cuda.device_count()} CUDA device(s) exist")
    return dev


def _logistic_loss(Xt, yt, wt, w, b, l2):
    """The (weighted) logistic objective with L2 on w only, shared by the token and mean-pooled fits."""
    torch = _torch()
    z = Xt @ w + b
    return (wt * torch.nn.functional.softplus(-(2 * yt - 1) * z)).sum() + 0.5 * l2 * (w ** 2).sum()


def _lbfgs_fit(X: np.ndarray, y: np.ndarray, wt: np.ndarray, l2: float, max_iter: int, device) -> tuple:
    """Full-batch L-BFGS on float32 tensors (as always); only the placement changes with `device`."""
    torch = _torch()
    dev = resolve_device(device)
    Xt, yt, wt = (torch.from_numpy(a).to(dev) for a in (X, y, wt))
    w = torch.zeros(X.shape[1], requires_grad=True, device=dev)
    b = torch.zeros(1, requires_grad=True, device=dev)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = _logistic_loss(Xt, yt, wt, w, b, l2)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().double().cpu().numpy(), float(b.item())


def _rng(seed: int, rollout_id: str):
    return np.random.default_rng([seed, zlib.crc32(rollout_id.encode())])


def _rollout_weights(labels: np.ndarray, class_balance: bool) -> np.ndarray:
    """Per-rollout weight summing to 1; optionally each class gets half the mass."""
    labels = np.asarray(labels)
    w = np.full(len(labels), 1.0 / len(labels))
    if class_balance and 0 < labels.sum() < len(labels):
        npos = labels.sum()
        w = np.where(labels == 1, 0.5 / npos, 0.5 / (len(labels) - npos))
    return w


READ_WORKERS = int(os.environ.get("CT_RH_READ_WORKERS", "12"))


def load_ordered(samples, layers, workers: int = READ_WORKERS):
    """Yield (sample, Extracted) in order while up to `workers` files are read concurrently. A single
    sequential reader got ~50 MB/s from the cluster's NFS; the shared filesystem serves parallel readers far faster.
    At most 2 x workers results are held at once (bounded memory)."""
    from concurrent.futures import ThreadPoolExecutor
    samples = list(samples)
    if workers <= 1 or len(samples) <= 1:
        for smp in samples:
            yield smp, smp.load(layers)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        window, nxt = [], 0
        while nxt < len(samples) and len(window) < 2 * workers:
            window.append((samples[nxt], pool.submit(samples[nxt].load, layers))); nxt += 1
        while window:
            smp, fut = window.pop(0)
            if nxt < len(samples):
                window.append((samples[nxt], pool.submit(samples[nxt].load, layers))); nxt += 1
            yield smp, fut.result()


def compact_one(smp, ex, layer, channels, max_tokens, seed):
    """Everything `train_pass` keeps from ONE rollout, nothing else: (m1, m2, raw token subsample, label), or None
    if the rollout has no eligible token. The subsample depends only on (seed, rollout_id), never on other rollouts,
    so it can be taken while streaming and reused by every fold."""
    idx = _eligible(ex, channels, smp.max_call)
    if len(idx) == 0:
        return None
    h = np.asarray(ex.hidden[layer][idx], dtype=np.float64)
    m1, m2 = h.mean(0), (h ** 2).mean(0)
    del h
    if len(idx) > max_tokens:
        idx = np.sort(_rng(seed, smp.rollout_id).choice(idx, max_tokens, replace=False))
    return m1, m2, np.ascontiguousarray(ex.hidden[layer][idx]), smp.label


def compact_pass(samples, layer, channels, max_tokens, seed, workers: int = READ_WORKERS):
    """ONE streaming read of `samples` at `layer`; returns a list aligned with `samples` of `compact_one` items.
    Memory: n x max_tokens x d (raw dtype) plus the in-flight files, instead of every rollout's full states."""
    samples = list(samples)
    out = [None] * len(samples)
    for i, (smp, ex) in enumerate(load_ordered(samples, [layer], workers)):
        out[i] = compact_one(smp, ex, layer, channels, max_tokens, seed)
        del ex
    return out


def train_pass_from_compact(items):
    """`train_pass` on precomputed `compact_one` items (None entries skipped), same accumulation order, so the
    result is bit-identical to `train_pass` over the same samples in the same order."""
    s1 = s2 = None
    R = 0
    raw, sizes, labels = [], [], []
    for it in items:
        if it is None:
            continue
        m1, m2, r, lab = it
        s1 = m1 if s1 is None else s1 + m1
        s2 = m2 if s2 is None else s2 + m2
        R += 1
        raw.append(r)
        sizes.append(len(r))
        labels.append(lab)
    if R == 0:
        raise ValueError("no eligible tokens to fit a normalizer")
    mu = s1 / R
    var = np.maximum(s2 / R - mu ** 2, 0.0)
    return Normalizer(mu, np.maximum(np.sqrt(var), 1e-6)), (raw, sizes, np.asarray(labels))


def train_pass(samples, layer, channels, max_tokens, seed):
    """ONE read of each train rollout at `layer`: the normalizer (exactly `fit_normalizer`) and the raw seeded
    token subsample `_gather_tokens` would take, so every l2 of the grid reuses them. Reading the per-layer
    arrays once instead of once per l2 plus once for the normalizer is what bounds the run time on NFS."""
    items = []
    for smp, ex in load_ordered(samples, [layer]):
        items.append(compact_one(smp, ex, layer, channels, max_tokens, seed))
        del ex
    return train_pass_from_compact(items)


def _gather_tokens(samples, layer, norm: Normalizer, channels, max_tokens, seed, pre=None):
    if pre is not None:                     # (raw subsamples, sizes, labels) from train_pass
        raw, sizes, labels = pre
        return np.concatenate([norm(r).astype(np.float32) for r in raw]), list(sizes), np.asarray(labels)
    X, sizes, labels = [], [], []
    for smp in samples:
        ex = smp.load([layer])
        idx = _eligible(ex, channels, smp.max_call)
        if len(idx) == 0:
            continue
        if len(idx) > max_tokens:
            idx = np.sort(_rng(seed, smp.rollout_id).choice(idx, max_tokens, replace=False))
        X.append(norm(ex.hidden[layer][idx]).astype(np.float32))
        sizes.append(len(idx))
        labels.append(smp.label)
    if not X:
        raise ValueError("no eligible tokens")
    return np.concatenate(X), sizes, np.asarray(labels)


def fit_logistic_token(samples: Sequence[RolloutSample], layer: int, *, norm: Normalizer,
                       l2: float = 1e-2, channels=act.DEFAULT_CHANNELS, max_tokens_per_rollout: int = 4096,
                       class_balance: bool = False, seed: int = 0, max_iter: int = 200, pre=None,
                       device: str = "cpu") -> dict:
    """Weak token supervision (rollout label on every token); returns {w, b, mu, sigma}.
    `pre`: the cached subsample from `train_pass` (same seed and max_tokens_per_rollout).
    `device`: where the L-BFGS runs ("cpu" by default); same float32 arithmetic, only the placement differs."""
    resolve_device(device)
    X, sizes, labels = _gather_tokens(samples, layer, norm, channels, max_tokens_per_rollout, seed, pre)
    rw = _rollout_weights(labels, class_balance)
    tok_w = np.concatenate([np.full(n, r / n) for r, n in zip(rw, sizes)])       # sums to 1
    y = np.concatenate([np.full(n, l, np.float32) for n, l in zip(sizes, labels)])
    w, b = _lbfgs_fit(X, y, tok_w.astype(np.float32), l2, max_iter, device)
    return {"w": w, "b": b, "mu": norm.mu, "sigma": norm.sigma}


def fit_logistic_mean(samples: Sequence[RolloutSample], layer: int, *, l2: float = 1e-2,
                      channels=act.DEFAULT_CHANNELS, class_balance: bool = False,
                      max_iter: int = 200, device: str = "cpu") -> dict:
    """Logistic regression on one mean-pooled activation vector per training rollout."""
    resolve_device(device)
    X, labels = [], []
    for smp, ex in load_ordered(samples, [layer]):
        idx = _eligible(ex, channels, smp.max_call)
        if len(idx):
            X.append(np.asarray(ex.hidden[layer][idx], dtype=np.float64).mean(0))
            labels.append(smp.label)
    if not X or len(set(labels)) < 2:
        raise ValueError("mean-pooled logistic fit requires eligible rollouts from both classes")
    return fit_logistic_mean_X(np.asarray(X, dtype=np.float64), np.asarray(labels, dtype=np.float32), l2=l2,
                               class_balance=class_balance, max_iter=max_iter, device=device)


def fit_logistic_mean_X(X: np.ndarray, y: np.ndarray, *, l2: float = 1e-2, class_balance: bool = False,
                        max_iter: int = 200, device: str = "cpu") -> dict:
    """The mean-pooled logistic fit on a precomputed (n_rollouts, d) matrix of mean activations."""
    resolve_device(device)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float32)
    mu = X.mean(0)
    sigma = np.maximum(X.std(0), 1e-6)
    X = ((X - mu) / sigma).astype(np.float32)
    wt = _rollout_weights(y, class_balance).astype(np.float32)
    w, b = _lbfgs_fit(X, y, wt, l2, max_iter, device)
    return {"w": w, "b": b, "mu": mu, "sigma": sigma}


def _ema_torch(s, alpha: float, e_prev, block: int = 512):
    """Differentiable EMA via blocked lower-triangular Toeplitz products (stable for any alpha)."""
    torch = _torch()
    idx = torch.arange(block, dtype=s.dtype, device=s.device)
    diff = idx[:, None] - idx[None, :]
    T = torch.where(diff >= 0, alpha * (1 - alpha) ** diff.clamp(min=0), torch.zeros((), dtype=s.dtype, device=s.device))
    dec = (1 - alpha) ** (idx + 1)
    out = []
    for i in range(0, len(s), block):
        n = min(block, len(s) - i)
        e = T[:n, :n] @ s[i:i + n] + dec[:n] * e_prev
        out.append(e)
        e_prev = e[-1]
    return torch.cat(out)


def fit_logistic_trajectory(samples: Sequence[RolloutSample], layer: int, *, norm: Normalizer, alpha: float,
                            init: dict | None = None, l2: float = 1e-2, channels=act.DEFAULT_CHANNELS,
                            pool: str = "max", tau: float = 0.5, epochs: int = 20, lr: float = 1e-2,
                            batch_rollouts: int = 8, class_balance: bool = True, ema_init: str = "first",
                            seed: int = 0, max_tokens: int | None = None, device: str = "cpu") -> dict:
    """Trajectory-level objective on pooled EMA scores (module docstring item 3). `device`: where the float64
    parameters and the per-rollout activations live during the fit ("cpu" by default)."""
    torch = _torch()
    dev = resolve_device(device)
    d = len(norm.mu)
    w = (torch.zeros(d, dtype=torch.float64, device=dev) if init is None
         else torch.as_tensor(init["w"], dtype=torch.float64).clone().to(dev))
    b = (torch.zeros((), dtype=torch.float64, device=dev) if init is None
         else torch.tensor(float(init["b"]), dtype=torch.float64, device=dev))
    w.requires_grad_(True), b.requires_grad_(True)
    opt = torch.optim.Adam([w, b], lr=lr)
    labels = np.asarray([s.label for s in samples])
    rw = _rollout_weights(labels, class_balance)
    rng = np.random.default_rng(seed)

    def pooled(smp):
        ex = smp.load([layer])
        idx = _eligible(ex, channels, smp.max_call)
        if max_tokens is not None:
            idx = idx[:max_tokens]
        if len(idx) == 0:
            return None
        h = torch.from_numpy(norm(ex.hidden[layer][idx])).to(dev)
        s = h @ w + b
        e = _ema_torch(s, alpha, s[0] if ema_init == "first" else torch.zeros((), dtype=s.dtype, device=dev))
        return e.max() if pool == "max" else tau * torch.logsumexp(e / tau, 0)

    for _ in range(epochs):
        order = rng.permutation(len(samples))
        for i in range(0, len(order), batch_rollouts):
            opt.zero_grad()
            loss = 0.5 * l2 * (w ** 2).sum()
            batch = order[i:i + batch_rollouts]
            for j in batch:
                S = pooled(samples[j])
                if S is None:
                    continue
                y = float(samples[j].label)
                # unbiased minibatch estimate of sum_j rw_j * BCE_j
                loss = loss + float(rw[j] * len(samples) / len(batch)) * torch.nn.functional.softplus(-(2 * y - 1) * S)
            loss.backward()
            opt.step()
    return {"w": w.detach().cpu().numpy().copy(), "b": float(b.item()), "mu": norm.mu, "sigma": norm.sigma}


# ---------------------------------------------------------------------------
# selection on val, threshold, artifact

def score_samples(art: tapmod.TapArtifact, samples: Sequence[RolloutSample], *, use_max_call: bool = True):
    """Offline S per sample (None -> -inf: nothing scored means the tap cannot fire)."""
    out = []
    for smp, ex in load_ordered(samples, [art.layer]):
        r = tapmod.score_extracted(art, ex, max_call=smp.max_call if use_max_call else None)
        out.append(-np.inf if r.S is None else r.S)
    return np.asarray(out)


def ema_blocked(s: np.ndarray, alpha: float, e_prev: float, block: int = 512) -> np.ndarray:
    """`tap.ema_scan` vectorized: per 512-token block a lower-triangular Toeplitz product plus the carried
    state (float64, stable for any alpha; equals the sequential scan to ~1e-12). Selection only: the tap and
    the frozen-threshold scoring keep the sequential reference."""
    s = np.asarray(s, dtype=np.float64)
    c = 1.0 - alpha
    k = np.arange(block)
    diff = k[:, None] - k[None, :]
    T = np.where(diff >= 0, alpha * c ** np.clip(diff, 0, None), 0.0)
    dec = c ** (k + 1)
    out = np.empty_like(s)
    e = float(e_prev)
    for i in range(0, len(s), block):
        n = min(block, len(s) - i)
        out[i:i + n] = T[:n, :n] @ s[i:i + n] + dec[:n] * e
        e = out[i + n - 1]
    return out


def val_token_scores(p: dict, samples: Sequence[RolloutSample], layer: int, channels) -> list:
    """Token scores s_t of probe params `p` for each (loaded) sample, as `TapArtifact.token_scores` on the
    tap's token mask (channels, max_call), with the normalization folded into the weights:
    ((h - mu) / sigma) . w + b == h . (w / sigma) + (b - mu . (w / sigma)). Computed once per fitted probe
    and reused for every alpha (s does not depend on alpha)."""
    wv = np.asarray(p["w"], np.float64) / np.asarray(p["sigma"], np.float64)
    b0 = float(p["b"]) - float(np.dot(np.asarray(p["mu"], np.float64), wv))
    codes = [act.CHANNEL_CODES[c] for c in channels]
    out = []
    for smp in samples:
        ex = smp.load([layer])
        keep = np.isin(ex.channel, codes)
        if smp.max_call is not None:
            keep &= ex.call_index < smp.max_call
        idx = np.nonzero(keep)[0]
        if len(idx) == 0:
            out.append(None)
            continue
        h, sc = ex.hidden[layer], np.empty(len(idx))
        for j in range(0, len(idx), 65536):
            sc[j:j + 65536] = h[idx[j:j + 65536]].astype(np.float64) @ wv + b0
        out.append((sc, ex.call_index[idx].copy()))
    return out


def fast_val_auroc(scores: list, labels, alpha: float, ema_init: str, pooling: str = "ema_max",
                   ema_scope: str = "rollout") -> float:
    if pooling == "token_mean":
        S = [-np.inf if s is None else float(np.mean(s[0])) for s in scores]
    else:
        S = []
        for item in scores:
            if item is None:
                S.append(-np.inf)
                continue
            s, call_ids = item
            if ema_scope == "call":
                vals = [ema_blocked(s[call_ids == c], alpha,
                                    s[call_ids == c][0] if ema_init == "first" else 0.0).max()
                        for c in np.unique(call_ids)]
                S.append(float(max(vals)))
            else:
                S.append(float(ema_blocked(s, alpha, s[0] if ema_init == "first" else 0.0).max()))
    a = metrics.auroc(list(labels), np.asarray(S))
    return -1.0 if a is None else a


def val_auroc(art, val: Sequence[RolloutSample]) -> float:
    a = metrics.auroc([s.label for s in val], score_samples(art, val))
    return -1.0 if a is None else a


def length_matched_threshold(art: tapmod.TapArtifact, val: Sequence[RolloutSample],
                             target_fpr: float = 0.10) -> dict:
    """Threshold sensitivity using val negatives matched to positive scored-token lengths.

    Each eligible positive is paired with its nearest eligible negative by absolute log token-count
    distance. Negative reuse is allowed; ties resolve by rollout_id. Matching uses lengths only,
    then the matched negative score multiset calibrates the requested FPR.
    """
    rows = []
    for smp, ex in load_ordered(val, [art.layer]):
        r = tapmod.score_extracted(art, ex, max_call=smp.max_call)
        if r.S is not None and r.n_scored > 0:
            rows.append({"rollout_id": smp.rollout_id, "label": int(smp.label),
                         "n_scored": int(r.n_scored), "score": float(r.S)})
    positives = sorted((r for r in rows if r["label"] == 1), key=lambda r: r["rollout_id"])
    negatives = sorted((r for r in rows if r["label"] == 0), key=lambda r: r["rollout_id"])
    if not positives or not negatives:
        return {"method": "nearest_absolute_log_scored_token_length; negative_reuse=true; ties=rollout_id",
                "target_fpr": float(target_fpr), "threshold": None,
                "n_val_positive_matched": len(positives), "n_val_negative_candidates": len(negatives),
                "n_val_negative_matches": 0, "n_unique_val_negatives_matched": 0,
                "matches": [], "reason": "no scored positive or no scored negative validation rollouts"}
    matches = []
    for pos in positives:
        neg = min(negatives, key=lambda r: (abs(math.log(r["n_scored"]) - math.log(pos["n_scored"])),
                                            r["rollout_id"]))
        matches.append({"positive_rollout_id": pos["rollout_id"], "positive_n_scored": pos["n_scored"],
                        "negative_rollout_id": neg["rollout_id"], "negative_n_scored": neg["n_scored"]})
    matched_scores = [next(r["score"] for r in negatives if r["rollout_id"] == m["negative_rollout_id"])
                      for m in matches]
    unique_negatives = {m["negative_rollout_id"] for m in matches}
    return {"method": "nearest_absolute_log_scored_token_length; negative_reuse=true; ties=rollout_id",
            "target_fpr": float(target_fpr), "threshold": float(metrics.threshold_at_fpr(matched_scores, target_fpr)),
            "n_val_positive_matched": len(positives), "n_val_negative_candidates": len(negatives),
            "n_val_negative_matches": len(matches), "n_unique_val_negatives_matched": len(unique_negatives),
            "matches": matches}


def select_probe(train: Sequence[RolloutSample], val: Sequence[RolloutSample], *, layers: Sequence[int],
                 method: str = "logistic_token", alphas: Sequence[float] = (0.02, 0.05, 0.2, 1.0),
                 pooling: str = "ema_max",
                 ema_scope: str = "rollout",
                 l2s: Sequence[float] = (1e-3, 1e-2, 1e-1), target_fpr: float = 0.05,
                 channels=act.DEFAULT_CHANNELS, ema_init: str = "first", dom_pairs: dict | None = None,
                 model_id: str = "", model_revision: str = "", tokenizer_revision: str = "",
                 seed: int = 0, device: str = "cpu", **fit_kw) -> tapmod.TapArtifact:
    """Choose layer / l2 / alpha by val AUROC of S, then the threshold at `target_fpr` on val negatives.

    Grid order is layers x l2 x alpha; a candidate must strictly beat the incumbent (ties keep
    the earlier, i.e. lower-layer / smaller-l2 / smaller-alpha candidate). `method`:
    'dom' (needs dom_pairs={layer: (pos, neg)}), 'logistic_token', 'logistic_trajectory'.
    The grid is recorded in the artifact's `extra`. `device` only places the probe fits ("cpu" by default; a CUDA
    device that is not available raises); it is not recorded, and scoring/selection stay on the CPU.
    """
    resolve_device(device)
    if method not in ("dom", "logistic_token", "logistic_trajectory"):
        raise ValueError(method)
    if pooling not in ("ema_max", "token_mean"):
        raise ValueError(pooling)
    if ema_scope not in ("rollout", "call"):
        raise ValueError(ema_scope)
    selection_alphas = (1.0,) if pooling == "token_mean" else tuple(alphas)
    if any(s.partition not in ("train", "") for s in train) or any(s.partition not in ("val", "") for s in val):
        raise ValueError("train/val samples must come from the train/val partitions")
    if {s.task_id for s in train} & {s.task_id for s in val}:
        raise ValueError("train and val share tasks")
    grid, best, best_auc = [], None, -2.0
    for layer in layers:
        # Load each val rollout's states for this layer once: every grid cell scores all of val,
        # and re-reading the .npz files per cell dominated the run time on NFS.
        val_layer = [dataclasses.replace(s, data=ex) for s, ex in load_ordered(val, [layer])]
        mtok = fit_kw.get("max_tokens_per_rollout", 4096)
        if method == "dom":
            norm, pre = fit_normalizer(train, layer, channels), None
        elif pooling == "token_mean" and method == "logistic_token":
            norm, pre = None, None
        else:
            norm, pre = train_pass(train, layer, channels, mtok, seed)
        fits = []                                      # (l2, alpha_or_None, params)
        if method == "dom":
            pos, neg = dom_pairs[layer]
            fits.append((None, None, fit_dom(pos, neg, norm=norm)))
        elif pooling == "token_mean" and method == "logistic_token":
            for l2 in l2s:
                fits.append((l2, None, fit_logistic_mean(train, layer, l2=l2, channels=channels,
                                                         class_balance=bool(fit_kw.get("class_balance", False)),
                                                         device=device)))
        else:
            for l2 in l2s:
                tok = fit_logistic_token(train, layer, norm=norm, l2=l2, channels=channels, seed=seed, pre=pre,
                                         device=device, **{k: v for k, v in fit_kw.items() if k in ("max_tokens_per_rollout", "class_balance")})
                if method == "logistic_token":
                    fits.append((l2, None, tok))
                else:
                    for a in alphas:
                        fits.append((l2, a, fit_logistic_trajectory(
                            train, layer, norm=norm, alpha=a, init=tok, l2=l2, channels=channels, ema_init=ema_init,
                            seed=seed, device=device, **{k: v for k, v in fit_kw.items() if k in (
                                "pool", "tau", "epochs", "lr", "batch_rollouts", "class_balance", "max_tokens")})))
        labels = [s.label for s in val_layer]
        for l2, a_fixed, p in fits:
            sc = val_token_scores(p, val_layer, layer, channels)          # once per fit, every alpha reuses it
            for a in ([a_fixed] if a_fixed is not None else selection_alphas):
                art = tapmod.TapArtifact(layer=layer, w=p["w"], b=p["b"], mu=p["mu"], sigma=p["sigma"], alpha=a,
                                         threshold=0.0, channels=tuple(channels), ema_init=ema_init,
                                         ema_scope=ema_scope)
                auc = fast_val_auroc(sc, labels, a, ema_init, pooling, ema_scope)
                grid.append({"layer": layer, "l2": l2, "alpha": a, "pooling": pooling, "val_auroc": auc})
                if auc > best_auc:
                    best, best_auc = art, auc
        best_val = {s.rollout_id: s for s in val_layer} if best is not None and best.layer == layer else None
        if best_val is not None:
            best_layer_val = [best_val[s.rollout_id] for s in val]
        del val_layer, pre
    # the chosen layer's val states were kept above, so the threshold does not re-read val
    best = dataclasses.replace(best, pooling=pooling, ema_scope=ema_scope)
    if pooling == "token_mean":
        scores = [(-np.inf if s is None else float(np.mean(s[0]))) for s in
                  val_token_scores({"w": best.w, "b": best.b, "mu": best.mu, "sigma": best.sigma},
                                   best_layer_val, best.layer, best.channels)]
    else:
        scores = score_samples(best, best_layer_val)
    neg = [s for s, smp in zip(scores, val) if smp.label == 0]
    thr = metrics.threshold_at_fpr(neg, target_fpr)
    extra = {"method": method, "target_fpr": target_fpr, "val_auroc": best_auc, "grid": grid,
             "n_train": len(train), "n_val": len(val), "n_train_pos": int(sum(s.label for s in train)),
             "n_val_pos": int(sum(s.label for s in val)), "seed": seed, "fit_kw": fit_kw}
    return dataclasses.replace(best, threshold=thr, method=method, pooling=pooling, ema_scope=ema_scope,
                               extra={**extra, "pooling": pooling, "ema_scope": ema_scope}, model_id=model_id,
                               model_revision=model_revision, tokenizer_revision=tokenizer_revision)
