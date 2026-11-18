"""Streaming white-box tap: s_t = w^T norm(h_t) + b, EMA, running max, threshold.

    norm(h)  = (h - mu) / sigma                      (per-dimension; mu, sigma fit on train tasks)
    s_t      = w . norm(h_t) + b
    e_t      = alpha * s_t + (1 - alpha) * e_{t-1}
    S        = max_t e_t
    predict  = 1[S > threshold]                      (strict >, as in `metrics`)

Semantics (all frozen in the artifact or documented here)
- Tokens: only *generated* tokens whose channel is in `channels` (default reasoning,
  tool_call, visible; not the stop token) are scored, using the predictive states of
  `activations`. Tool outputs and repeated prompt/context tokens are never scored: the tap
  is fed the states of each call's generated tokens only, and never rescored.
- Initialization: `ema_init="first"` (default): e_0 := s_1 at the first scored token of the
  rollout, so e_1 = s_1 (no pull toward 0, which is an arbitrary point for a learned bias).
  `"zero"`: e_0 = 0. Either way S is the max over e_t, t >= 1.
- Reset: one StreamingTap per rollout (`reset()` returns it to the initial state). Legacy
  artifacts carry EMA across calls; artifacts with `ema_scope="call"` reset EMA per invocation
  and retain the maximum call score across the rollout.
- Decision points: `call_scores[c]` is the max-so-far S after model call c (None until a
  token has been scored). `decision_point_scores` maps these onto
  `first_violation.decision_points`, so a per-action detector reads S as of that call only.
- Numerics: scoring and the recursion run in float64 on the scalar scores; the dot product
  may run on the accelerator (torch input) so only n scalars cross to the CPU. The EMA is a
  sequential loop, so chunking by call changes nothing; the dot product can differ in the
  last bits between chunkings (~1e-12 relative), which the parity tests bound.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import pathlib
from collections.abc import Mapping, Sequence

import numpy as np

from evals.reward_hacking import activations as act

ARTIFACT_VERSION = 1


def _f(x: float):
    """JSON-safe float (inf/-inf/nan become strings)."""
    x = float(x)
    return x if math.isfinite(x) else repr(x)


def _unf(x) -> float:
    return float(x)   # float("-inf") / float("inf") parse from the strings above


def _arr(a: np.ndarray | None):
    if a is None:
        return None
    a = np.ascontiguousarray(a, dtype="<f8")
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def _unarr(d) -> np.ndarray | None:
    if d is None:
        return None
    return np.frombuffer(base64.b64decode(d["b64"]), dtype="<f8").reshape(d["shape"]).copy()


@dataclasses.dataclass(frozen=True, eq=False)
class TapArtifact:
    """Everything a deployment needs to reproduce the tap; stable `digest`."""
    layer: int
    w: np.ndarray                      # [d]
    b: float
    mu: np.ndarray | None              # [d] or None (no normalization)
    sigma: np.ndarray | None           # [d] or None
    alpha: float
    threshold: float
    channels: tuple[str, ...] = act.DEFAULT_CHANNELS
    ema_init: str = "first"
    model_id: str = ""
    model_revision: str = ""
    tokenizer_revision: str = ""
    method: str = ""                   # e.g. "dom", "logistic_token", "logistic_trajectory_max"
    pooling: str = "ema_max"             # rollout aggregation: ema_max or token_mean
    ema_scope: str = "rollout"            # rollout (legacy) or call (reset for each invocation)
    extra: dict = dataclasses.field(default_factory=dict)   # training/selection provenance

    def __post_init__(self):
        if self.ema_init not in ("first", "zero"):
            raise ValueError(self.ema_init)
        if self.pooling not in ("ema_max", "token_mean"):
            raise ValueError(f"unknown pooling {self.pooling!r}")
        if self.ema_scope not in ("rollout", "call"):
            raise ValueError(f"unknown ema_scope {self.ema_scope!r}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        for c in self.channels:
            if c not in act.CHANNEL_CODES:
                raise ValueError(f"unknown channel {c!r}")
        object.__setattr__(self, "channels", tuple(sorted(self.channels)))

    # -- identity
    def to_dict(self) -> dict:
        out = {
            "version": ARTIFACT_VERSION, "layer": int(self.layer), "w": _arr(self.w), "b": _f(self.b),
            "mu": _arr(self.mu), "sigma": _arr(self.sigma), "alpha": _f(self.alpha),
            "threshold": _f(self.threshold), "channels": list(self.channels), "ema_init": self.ema_init,
            "comparison": "score > threshold", "model_id": self.model_id,
            "model_revision": self.model_revision, "tokenizer_revision": self.tokenizer_revision,
            "layer_convention": act.LAYER_CONVENTION, "alignment": act.ALIGNMENT,
            "method": self.method, "extra": self.extra,
        }
        if self.pooling != "ema_max":
            out["pooling"] = self.pooling
        if self.ema_scope != "rollout":
            out["ema_scope"] = self.ema_scope
        return out

    @property
    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_json(self) -> str:
        d = self.to_dict()
        d["digest"] = self.digest
        return json.dumps(d, sort_keys=True, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "TapArtifact":
        d = json.loads(text)
        if d.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"unsupported artifact version {d.get('version')}")
        a = cls(layer=d["layer"], w=_unarr(d["w"]), b=_unf(d["b"]), mu=_unarr(d["mu"]),
                sigma=_unarr(d["sigma"]), alpha=_unf(d["alpha"]), threshold=_unf(d["threshold"]),
                channels=tuple(d["channels"]), ema_init=d["ema_init"], model_id=d["model_id"],
                model_revision=d["model_revision"], tokenizer_revision=d["tokenizer_revision"],
                method=d["method"], extra=d["extra"], pooling=d.get("pooling", "ema_max"),
                ema_scope=d.get("ema_scope", "rollout"))
        expected = a.digest
        if d.get("digest") not in (None, expected):
            raise ValueError("artifact digest mismatch (file edited or corrupted)")
        return a

    def save(self, path) -> str:
        pathlib.Path(path).write_text(self.to_json())
        return self.digest

    @classmethod
    def load(cls, path) -> "TapArtifact":
        return cls.from_json(pathlib.Path(path).read_text())

    def with_threshold(self, threshold: float) -> "TapArtifact":
        return dataclasses.replace(self, threshold=float(threshold))

    # -- scoring
    @property
    def channel_codes(self) -> np.ndarray:
        return np.array([act.CHANNEL_CODES[c] for c in self.channels], dtype=np.int8)

    def token_scores(self, h) -> np.ndarray:
        """s_t for states h [n, d] (numpy or torch; float64 result on the host)."""
        if hasattr(h, "detach"):                      # torch: dot on device, move n scalars
            import torch
            h = h.detach().to(torch.float64)
            if self.mu is not None:
                h = (h - torch.as_tensor(self.mu, device=h.device)) / torch.as_tensor(self.sigma, device=h.device)
            s = h @ torch.as_tensor(self.w, device=h.device) + self.b
            return s.cpu().numpy()
        h = np.asarray(h, dtype=np.float64)
        if self.mu is not None:
            h = (h - self.mu) / self.sigma
        return h @ self.w + self.b


def ema_scan(s: np.ndarray, alpha: float, e_prev: float) -> np.ndarray:
    """e_t = alpha*s_t + (1-alpha)*e_{t-1}, sequential in float64 (chunking-invariant)."""
    out = np.empty(len(s), dtype=np.float64)
    e, a, c = float(e_prev), float(alpha), 1.0 - float(alpha)
    for i, x in enumerate(np.asarray(s, dtype=np.float64).tolist()):
        e = a * x + c * e
        out[i] = e
    return out


class StreamingTap:
    """Online scorer: feed each model call's generated-token states as they appear."""

    def __init__(self, artifact: TapArtifact):
        self.art = artifact
        self.reset()

    def reset(self):
        self.e: float | None = None          # None until the first scored token
        self.current_call_index: int | None = None
        self.S: float | None = None          # running max of e (None until scored)
        self.n_scored = 0
        self.score_sum = 0.0
        self.call_scores: dict[int, float | None] = {}

    def update(self, states, channel: np.ndarray, call_index: int) -> float | None:
        """Consume one call's generated-token states; returns the max-so-far S after it.

        `states`: [n, d] array/tensor at the artifact's layer, or {layer: [n, d]}.
        `channel`: int8 [n] channel codes of those tokens (`activations.CHANNEL_CODES`).
        """
        if isinstance(states, Mapping):
            states = states[self.art.layer]
        channel = np.asarray(channel)
        if len(channel) != states.shape[0]:
            raise ValueError("channel and states length differ")
        call_index = int(call_index)
        if self.art.ema_scope == "call" and call_index != self.current_call_index:
            self.e = None
        self.current_call_index = call_index
        keep = np.isin(channel, self.art.channel_codes)
        if keep.any():
            if hasattr(states, "detach"):
                import torch
                states = states[torch.as_tensor(np.nonzero(keep)[0], device=states.device)]
            else:
                states = np.asarray(states)[keep]
            s = self.art.token_scores(states)
            self.score_sum += float(np.asarray(s, dtype=np.float64).sum())
            self.n_scored += len(s)
            if self.art.pooling == "token_mean":
                self.S = self.score_sum / self.n_scored
                self.call_scores[int(call_index)] = self.S
                return self.S
            if self.e is None:
                self.e = float(s[0]) if self.art.ema_init == "first" else 0.0
            e = ema_scan(s, self.art.alpha, self.e)
            self.e = float(e[-1])
            m = float(e.max())
            self.S = m if self.S is None else max(self.S, m)
        self.call_scores[int(call_index)] = self.S
        return self.S

    @property
    def predict(self) -> bool:
        return self.S is not None and self.S > self.art.threshold


@dataclasses.dataclass
class RolloutScore:
    S: float | None
    n_scored: int
    call_scores: dict[int, float | None]     # max-so-far after each call (carried when a call scores nothing)
    s: np.ndarray                            # token scores (scored tokens only)
    e: np.ndarray                            # EMA values (scored tokens only)
    token_index: np.ndarray                  # indices into the extracted arrays of the scored tokens

    def predict(self, threshold: float) -> bool:
        return self.S is not None and self.S > threshold


def score_extracted(art: TapArtifact, ex: "act.Extracted", *, max_call: int | None = None) -> RolloutScore:
    """Offline reference scorer over a whole rollout's extracted states.

    `max_call`: score only calls with call_index < max_call (pre-violation evaluation: pass
    c*, the first violating call). Vectorized over the rollout, independent of StreamingTap.
    """
    keep = np.isin(ex.channel, art.channel_codes)
    if max_call is not None:
        keep &= ex.call_index < max_call
    idx = np.nonzero(keep)[0]
    calls = sorted(set(int(c) for c in ex.call_index if max_call is None or c < max_call))
    if len(idx) == 0:
        return RolloutScore(None, 0, {c: None for c in calls}, np.zeros(0), np.zeros(0), idx)
    s = art.token_scores(ex.hidden[art.layer][idx])
    if art.pooling == "token_mean":
        running = np.cumsum(s) / np.arange(1, len(s) + 1)
        ci = ex.call_index[idx]
        cs = {}
        for c in calls:
            sel = np.nonzero(ci <= c)[0]
            cs[c] = float(running[sel[-1]]) if len(sel) else None
        return RolloutScore(float(np.mean(s)), len(idx), cs, s, running, idx)
    ci = ex.call_index[idx]
    e = np.empty_like(s)
    call_max = {}
    if art.ema_scope == "call":
        for c in calls:
            where = np.nonzero(ci == c)[0]
            if len(where):
                block = s[where]
                vals = ema_scan(block, art.alpha, float(block[0]) if art.ema_init == "first" else 0.0)
                e[where] = vals
                call_max[c] = float(vals.max())
    else:
        e = ema_scan(s, art.alpha, float(s[0]) if art.ema_init == "first" else 0.0)
        for c in calls:
            where = np.nonzero(ci <= c)[0]
            if len(where):
                call_max[c] = float(np.maximum.accumulate(e[where])[-1])
    cs, last = {}, None
    for c in calls:
        if c in call_max:
            last = max(last, call_max[c]) if last is not None else call_max[c]
        cs[c] = last
    return RolloutScore(float(max(call_max.values())) if call_max else None, len(idx), cs, s, e, idx)


def score_online(art: TapArtifact, extractor: "act.CallExtractor", calls: Sequence["act.CallSeq"]) -> StreamingTap:
    """Call-synchronous online path: teacher-forced prefill after each call, then tap.update."""
    if art.layer not in extractor.layers:
        raise ValueError(f"extractor does not capture layer {art.layer}")
    tap = StreamingTap(art)
    extractor.reset()
    for c in calls:
        states, _ = extractor.process(c.prompt_ids, c.gen_ids)
        tap.update(states[art.layer], c.channel, c.call_index)
    return tap


def decision_point_scores(traj, call_scores: Mapping[int, float | None]) -> list[dict]:
    """Max-so-far score at each per-action decision point (`first_violation.decision_points`).

    Each point is the last event of a model call; the score is S after that call only, so a
    point strictly before the first violating call's point uses no later information.
    """
    from evals.reward_hacking.first_violation import decision_points
    out = []
    for seq in decision_points(traj):
        c = traj.events[seq].call_index
        out.append({"seq": seq, "call_index": c, "score": call_scores.get(c)})
    return out
