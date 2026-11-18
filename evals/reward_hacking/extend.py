"""Raise the per-call token cap (and/or the context rule) without resampling: continue the rollouts it cut.

A rollout in which no model call reached the old cap is, token for token, a valid rollout under any larger
cap: the cap never bound, so it is reused unchanged. A rollout that stopped because call j hit the old cap
(`model_length`) is continued, not resampled (resampling only those would be outcome-dependent selection):

  calls 0..j-1  are replayed from the record (their recorded completions are returned) while the tools run
                for real in a fresh sandbox, for their side effects; the model is given each tool's RECORDED
                output (what it actually saw: a replayed `ls -la` prints new timestamps), and after every
                tool call the replayed func.py/test.py must hash to the recorded workspace state. With
                recorded completions and outputs, each prompt the loop renders must then equal the recorded
                prompt byte for byte (checked);
  call j        is resumed: the server gets the recorded prompt's token ids plus the j-th call's recorded
                generated ids and samples up to (new cap - recorded length) more tokens; the merged
                completion is what one uninterrupted call under the new cap would have produced
                (autoregressive sampling: the continuation is a draw from the same conditional);
  calls > j     are sampled normally.

Raising `max_context_tokens` works the same way for a rollout that stopped `context_exhausted`: every recorded
call is replayed (none was capped), and the loop then carries on from the last one, running its tool calls,
which the original never executed, live. A cap of 0 means no per-call cap (the server's remaining context).

A prompt or workspace mismatch during replay raises `ReplayMismatch` (the rollout ends as an infra error
and is reported, never silently resampled).
"""

from __future__ import annotations

import dataclasses
import math

from evals.reward_hacking import llm
from evals.reward_hacking import trajectory as tj

REUSE, EXTEND, FRESH = "reuse", "extend", "fresh"


class ReplayMismatch(RuntimeError):
    """The replayed loop rendered a prompt that differs from the recorded one."""


def _cap(c: dict) -> float:
    return c["max_tokens_per_call"] or math.inf


def plan(traj: tj.Trajectory | None, old: dict, new: dict) -> str:
    """REUSE (no old limit that grew ever bound), EXTEND (stopped by a limit that grew), FRESH (missing or
    infra error). `old`/`new` are rollout configs (`max_tokens_per_call`, `max_context_tokens`)."""
    if traj is None:
        return FRESH
    end = traj.of(tj.ROLLOUT_END)
    status = end[-1].data["status"] if end else "infra_error"
    if status == "infra_error":
        return FRESH
    rid = traj.meta.get("rollout_id")
    calls = traj.of(tj.MODEL_CALL)
    capped = [c for c in calls if c.data["stop_type"] == "limit"]
    if status == "model_length":
        if not capped or capped[-1] is not calls[-1] or len(calls[-1].data["tokens"]) != _cap(old):
            raise ValueError(f"{rid}: model_length without a final call at the old cap")
        return EXTEND if _cap(new) > _cap(old) else REUSE
    if capped or any(len(c.data["tokens"]) >= _cap(old) for c in calls):
        raise ValueError(f"{rid}: a call hit the cap but status is {status!r}")
    if status == "context_exhausted" and new["max_context_tokens"] > old["max_context_tokens"]:
        return EXTEND
    return REUSE


class ExtendingClient:
    """Wraps a completion client for one rollout; see the module docstring."""

    def __init__(self, inner, recorded_calls: list[dict], recorded: tj.Trajectory | None = None):
        self.inner = inner
        self.recorded = recorded_calls          # MODEL_CALL data, in call order; the last one is capped
        self.n = 0
        self.info: dict = {}
        self.tool_results, self.expected_ws = {}, {}
        self.outputs_differed = 0
        if recorded is not None:
            cur = None
            for e in recorded.events:           # the workspace hash after each tool call of the record
                if e.kind == tj.TOOL_RESULT:
                    self.tool_results[e.data["call_id"]] = e.data
                    self.expected_ws[e.data["call_id"]] = cur
                elif e.kind == tj.WORKSPACE:
                    cur = e.data["sha256"]
                    if e.data["after_call_id"] is not None:
                        self.expected_ws[e.data["after_call_id"]] = cur

    def replay_tool_outcome(self, call_id: str, out):
        rec = self.tool_results.get(call_id)
        if rec is None:                         # a call made after the extended call: live output
            return out
        self.outputs_differed += out.output_text != rec["output"]
        trunc = rec.get("truncated")
        return dataclasses.replace(out, output_text=rec["output"], error_type=rec.get("error_type"),
                                   truncated=tuple(trunc) if trunc else None)

    def replay_check_workspace(self, call_id: str, sha: str | None) -> None:
        if call_id in self.tool_results and self.expected_ws.get(call_id) != sha:
            raise ReplayMismatch(f"workspace after {call_id} differs from the record")

    def complete(self, prompt: str, *, max_tokens: int, seed: int, sampling: dict) -> llm.Completion:
        k, self.n = self.n, self.n + 1
        last = len(self.recorded) - 1
        if k > last:
            return self.inner.complete(prompt, max_tokens=max_tokens, seed=seed, sampling=sampling)
        rec = self.recorded[k]
        if prompt != rec["prompt"]:
            at = next((i for i, (a, b) in enumerate(zip(prompt, rec["prompt"])) if a != b),
                      min(len(prompt), len(rec["prompt"])))
            raise ReplayMismatch(f"call {k}: rendered prompt differs from the record at char {at} "
                                 f"(len {len(prompt)} vs {len(rec['prompt'])})")
        timings = dict(rec.get("timings") or {})
        if k < last or rec["stop_type"] != "limit":       # replayed (a resumed rollout replays every call)
            if k == last:
                self.info = {"resumed_after_call": k, "replayed_tool_outputs_differed": self.outputs_differed}
            return llm.Completion(text=rec["generation"], tokens=list(rec["tokens"]), stop_type=rec["stop_type"],
                                  prompt_tokens=rec["prompt_tokens"], cached_tokens=rec["cached_tokens"],
                                  seconds=0.0, timings={**timings, "replayed": True})
        partial = list(rec["tokens"])
        remaining = None if max_tokens is None else max_tokens - len(partial)   # None: no cap
        if remaining is not None and remaining <= 0:
            raise ValueError(f"new cap {max_tokens} does not exceed the recorded generation ({len(partial)})")
        ids = self.inner.tokenize(prompt)
        if len(ids) != rec["prompt_tokens"]:
            raise ReplayMismatch(f"call {k}: prompt tokenizes to {len(ids)} ids, record says {rec['prompt_tokens']}")
        cont = self.inner.complete(ids + partial, max_tokens=remaining, seed=seed, sampling=sampling)
        text = rec["generation"] + cont.text
        self.info = {"extended_call": k, "recorded_tokens": len(partial), "continued_tokens": len(cont.tokens),
                     "replayed_tool_outputs_differed": self.outputs_differed,
                     "boundary_replacement_char": "�" in rec["generation"][-4:] + cont.text[:4]}
        return llm.Completion(text=text, tokens=partial + list(cont.tokens), stop_type=cont.stop_type,
                              prompt_tokens=rec["prompt_tokens"], cached_tokens=cont.cached_tokens,
                              seconds=cont.seconds, timings={**cont.timings, "extension": self.info})
