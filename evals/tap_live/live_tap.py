"""Probe tap attached to the live engine: capture during generation, score at the end of every call.

What runs where
---------------
  * a forward hook on decoder block L-1 (L = the artifact's layer, hidden_states-tuple convention of
    `activations.py`) copies, at EVERY forward pass (the prefill and each decode step), the hidden state of the LAST
    position into a GPU float32 buffer row. That state is the one that produces the next token, the "predictive"
    alignment of `activations.py`. No host synchronisation, no change to the layer output (read-only). The buffer grows
    by doubling when a call is longer than expected (a device-side copy, no host sync);
  * when the call's last token has been sampled, and before `complete()` returns, `end_call` runs on the GPU, in the
    agent loop's critical path, once for each HEAD (an artifact: layer, normalisation, weights, pooling): stack the
    call's states (the state that produced the stop token is left out, as the repo leaves out the END token),
    normalise ((h - mu) / sigma), dot with w plus b (float32), pool (token_mean, or the repo's EMA-max recurrence over
    the token scores in order, `probe._ema_torch`, with the artifact's alpha and `ema_init`), then ONE read-back
    (one synchronisation) per head, compare with the frozen threshold (strict `>`), append the decision to
    `decisions` (one entry per call and head).

Several heads share the capture when they read the same layer (the capture is then done and timed once); each head's
scoring is timed separately, so the cost of one pooling alone is `capture_s + score_s_by_head[head]`.

Scope. `scope="call"` (default): the pooled score belongs to the current call (state reset at call start).
`scope="rollout"`: the artifact's declared legacy scope (`extra.ema_scope == "rollout"`): token-mean sum and count and the
EMA / running max carry across the calls of the episode (reset by `begin_episode`). Both scores are computed and
logged on every call (one read-back of both); the alarm uses `scope`.

Timing. `capture_s` = CUDA-event time around the hooks' copies, summed over the call's forwards (resolved to numbers by
`resolve()` after the episode, so reading the events costs nothing inside the timed episode; on CPU perf_counter).
`score_s_by_head[h]` = perf_counter around head h's call-end scoring including its read-back; `score_s` is their sum;
`tap_s = capture_s + score_s`. `time_capture=False` skips the event records (an uninstrumented capture;
`capture_s` is then None and `tap_s` is `score_s`).
"""
from __future__ import annotations

import hashlib
import pathlib
import time

import torch

from evals.reward_hacking import activations as act
from evals.reward_hacking import probe as probe_mod
from evals.reward_hacking import tap as tapmod

INITIAL_ROWS = 4096


class _Head:
    def __init__(self, name, artifact, device):
        if isinstance(artifact, (str, pathlib.Path)):
            self.path = str(artifact)
            self.sha256 = hashlib.sha256(pathlib.Path(artifact).read_bytes()).hexdigest()
            artifact = tapmod.TapArtifact.load(artifact)
        else:
            self.path, self.sha256 = None, None
        self.name, self.art = name, artifact
        f32 = dict(dtype=torch.float32, device=device)
        self.w = torch.as_tensor(artifact.w, **f32)
        self.b = float(artifact.b)
        self.mu = None if artifact.mu is None else torch.as_tensor(artifact.mu, **f32)
        self.sigma = None if artifact.sigma is None else torch.as_tensor(artifact.sigma, **f32)
        self.d = int(self.w.shape[0])
        self.reset_episode()

    def reset_episode(self):
        self.r_sum = None
        self.r_n = 0
        self.r_e = None
        self.r_max = None
        self.r_last = None

    def describe(self) -> dict:
        a = self.art
        return {"name": self.name, "path": self.path, "sha256": self.sha256, "digest": a.digest, "layer": a.layer,
                "pooling": a.pooling, "alpha": a.alpha, "ema_init": a.ema_init, "threshold": a.threshold,
                "channels": list(a.channels), "model_id": a.model_id, "model_revision": a.model_revision,
                "artifact_ema_scope": a.ema_scope, "artifact_extra_ema_scope": (a.extra or {}).get("ema_scope")}


class LiveProbeTap:
    """`artifacts`: one TapArtifact (or path), or {head name: TapArtifact or path}."""

    def __init__(self, artifacts, *, device, scope: str = "call", time_capture: bool = True, name: str | None = None):
        if scope not in ("call", "rollout"):
            raise ValueError(f"scope must be 'call' or 'rollout', got {scope!r}")
        self._torch = torch
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        if not isinstance(artifacts, dict):
            artifacts = {None: artifacts}
        self.heads: dict[str, _Head] = {}
        for key, art in artifacts.items():
            h = _Head(key, art, self.device)
            h.name = key if key is not None else f"{h.art.pooling}_L{h.art.layer}"
            self.heads[h.name] = h
        self.name = name or "+".join(self.heads)
        self.scope = scope
        self.time_capture = bool(time_capture)
        self.layers = sorted({h.art.layer for h in self.heads.values()})
        dims = {L: {h.d for h in self.heads.values() if h.art.layer == L} for L in self.layers}
        if any(len(v) != 1 for v in dims.values()):
            raise ValueError(f"heads on one layer disagree on the hidden size: {dims}")
        self.d = {L: next(iter(v)) for L, v in dims.items()}
        self.bufs = {L: torch.zeros((0, self.d[L]), dtype=torch.float32, device=self.device) for L in self.layers}
        self._n = {L: 0 for L in self.layers}
        self.capture_enabled = True
        self._in_call = False
        self._handles: list = []
        self._pool: list = []                 # CUDA event pairs reused across the calls of an episode
        self._pool_next = 0
        self._call_pair_start = 0
        self._cpu_capture = 0.0
        self._pending: list = []              # (timings dict, first pair, last pair) waiting for `resolve`
        self.decisions: list[dict] = []
        self.begin_episode()

    # -- model attachment
    @property
    def art(self):
        """The artifact of a single-head tap."""
        if len(self.heads) != 1:
            raise AttributeError("a multi-head tap has no single artifact; use .heads[name].art")
        return next(iter(self.heads.values())).art

    def attach(self, model) -> None:
        self.detach()
        layers = act.check_layers(model, self.layers)
        blocks = act.find_decoder_layers(model)
        for L in layers:
            self._handles.append(blocks[L - 1].register_forward_hook(self._make_hook(L)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _pair(self, i: int):
        while len(self._pool) <= i:
            self._pool.append((self._torch.cuda.Event(enable_timing=True), self._torch.cuda.Event(enable_timing=True)))
        return self._pool[i]

    def _make_hook(self, L: int):
        def hook(_module, _inputs, out):
            """Read-only: copies the last position's state into the buffer; returns None (the output is untouched)."""
            if not self._in_call or not self.capture_enabled:     # forwards outside a call (a test's reference pass) are not tapped
                return None
            h = out[0] if isinstance(out, tuple) else out
            i = self._n[L]
            buf = self.bufs[L]
            if i >= buf.shape[0]:
                grown = self._torch.zeros((max(2 * buf.shape[0], INITIAL_ROWS), buf.shape[1]), dtype=buf.dtype, device=buf.device)
                grown[:buf.shape[0]] = buf
                buf = self.bufs[L] = grown
            if self.time_capture:
                if self.cuda:
                    a, z = self._pair(self._pool_next)
                    a.record()
                    buf[i].copy_(h[0, -1])
                    z.record()
                    self._pool_next += 1
                else:
                    t = time.perf_counter()
                    buf[i].copy_(h[0, -1])
                    self._cpu_capture += time.perf_counter() - t
            else:
                buf[i].copy_(h[0, -1])
            self._n[L] = i + 1
            return None
        return hook

    # -- lifecycle
    def begin_episode(self) -> None:
        for h in self.heads.values():
            h.reset_episode()
        self._pool_next = 0
        self._pending = []
        self.decisions = []

    def begin_call(self, max_tokens, call_index: int) -> None:
        for L in self.layers:
            if self.bufs[L].shape[0] == 0:
                rows = INITIAL_ROWS if max_tokens is None else max(1, min(int(max_tokens), INITIAL_ROWS))
                self.bufs[L] = torch.zeros((rows, self.d[L]), dtype=torch.float32, device=self.device)
            self._n[L] = 0
        self._in_call = True
        self.capture_enabled = True
        self._cpu_capture = 0.0
        self._call_pair_start = self._pool_next

    def abort_call(self) -> None:
        self._n = {L: 0 for L in self.layers}
        self._in_call = False
        self.capture_enabled = True

    # -- scoring
    def _token_scores(self, head: _Head, h):
        z = h if head.mu is None else (h - head.mu) / head.sigma
        return z @ head.w + head.b

    def _ema(self, head: _Head, s, e0):
        return probe_mod._ema_torch(s, float(head.art.alpha), e0)

    def _score_head(self, head: _Head, m: int, call_index: int) -> tuple[dict, float]:
        t0 = time.perf_counter()
        art = head.art
        call_S = roll_S = None
        if m > 0:
            s = self._token_scores(head, self.bufs[art.layer][:m])
            first0 = s[0] if art.ema_init == "first" else torch.zeros((), dtype=s.dtype, device=s.device)
            if art.pooling == "token_mean":
                tot = s.sum()
                call_t = tot / m
                head.r_sum = tot if head.r_sum is None else head.r_sum + tot
                head.r_n += m
                roll_t = head.r_sum / head.r_n
            else:
                e = self._ema(head, s, first0)
                call_t = e.max()
                e_r = e if head.r_e is None else self._ema(head, s, head.r_e)
                head.r_e = e_r[-1]
                mx = e_r.max()
                head.r_max = mx if head.r_max is None else torch.maximum(head.r_max, mx)
                roll_t = head.r_max
            vals = torch.stack([call_t, roll_t]).tolist()            # the one read-back / synchronisation of this head
            call_S, roll_S = float(vals[0]), float(vals[1])
            head.r_last = roll_S
        else:
            roll_S = head.r_last                                     # nothing scored: the episode score is unchanged
        dt = time.perf_counter() - t0
        score = call_S if self.scope == "call" else roll_S
        dec = {"call_index": int(call_index), "head": head.name, "n_tokens": int(m), "score": score, "score_call": call_S,
               "score_rollout": roll_S, "alarm": bool(score is not None and score > art.threshold),
               "threshold": float(art.threshold), "scope": self.scope, "score_s": dt}
        return dec, dt

    @torch.no_grad()
    def end_call(self, *, n_generated: int, last_is_stop: bool, call_index: int, record: dict | None = None) -> dict:
        """Score the call that just finished with every head. Returns the timing fields for the call's record; the
        decisions are appended to `self.decisions`."""
        for L in self.layers:
            if self._n[L] != n_generated:
                raise RuntimeError(f"tap captured {self._n[L]} states at layer {L} for {n_generated} generated tokens")
        self._in_call = False
        m = n_generated - 1 if last_is_stop else n_generated        # the state that produced the stop token is END
        by_head, total = {}, 0.0
        for head in self.heads.values():
            dec, dt = self._score_head(head, m, call_index)
            self.decisions.append(dec)
            by_head[head.name] = dt
            total += dt
        out = {"score_s": total, "score_s_by_head": by_head, "tap_n_scored": int(m)}
        if self.time_capture and not self.cuda:
            out["capture_s"] = self._cpu_capture
            out["tap_s"] = self._cpu_capture + total
        elif self.time_capture:
            out["capture_s"], out["tap_s"] = None, None              # filled by resolve()
            self._pending.append((record, self._call_pair_start, self._pool_next))
        else:
            out["capture_s"], out["tap_s"] = None, total
        if record is not None:
            record.update(out)
        return out

    def resolve(self) -> None:
        """Read the CUDA events recorded during the episode (after the episode's wall time was taken)."""
        if not self.cuda or not self._pending:
            self._pending = []
            return
        self._torch.cuda.synchronize()
        for record, lo, hi in self._pending:
            ms = sum(self._pool[i][0].elapsed_time(self._pool[i][1]) for i in range(lo, hi))
            if record is not None:
                record["capture_s"] = ms / 1000.0
                record["tap_s"] = record["capture_s"] + record["score_s"]
        self._pending = []

    def score_of_call(self, call_index: int, head: str | None = None):
        """Pooled score (per `scope`) logged for a call (and head), None if nothing was scored."""
        if head is None:
            if len(self.heads) != 1:
                raise ValueError("a multi-head tap needs a head name")
            head = next(iter(self.heads))
        for d in self.decisions:
            if d["call_index"] == call_index and d["head"] == head:
                return d["score"]
        raise KeyError((call_index, head))

    def describe(self) -> dict:
        return {"name": self.name, "scope": self.scope, "layers": self.layers, "time_capture": self.time_capture,
                "heads": {n: h.describe() for n, h in self.heads.items()}}
