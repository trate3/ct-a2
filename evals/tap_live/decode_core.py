"""Shared pieces of the focused live-decoding measurement: a base cache, exact cache copies, and one timed decode.

`decode_run` is the timed region of `decode_bench`: starting from a cache that holds every prompt token but the last, it
feeds the last prompt token (one forward), decodes up to `max_new_tokens` with the agent's sampling and stops at the
format's stop tokens, and, when a tap is attached, scores the call at the end. Nothing inside it uses CUDA events unless
`step_events=True` (the separate profile pass): the primary timing is wall clock with one host sync per generated token and
nothing else, identically in both arms.
"""
from __future__ import annotations

import copy
import hashlib
import time

import torch

from evals.tap_live.live_client import check_sampling, sample_token


def iter_tensors(obj, path: str = ""):
    """(path, tensor) for every tensor reachable from a cache object (layers, dicts of states), in a fixed order."""
    if torch.is_tensor(obj):
        yield path, obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from iter_tensors(obj[k], f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_tensors(v, f"{path}/{i}")
    elif hasattr(obj, "__dict__"):
        for k, v in sorted(vars(obj).items()):
            if isinstance(v, (torch.device, torch.dtype, int, float, bool, str, type(None))):
                continue
            yield from iter_tensors(v, f"{path}/{k}")


def caches_equal(a, b) -> bool:
    """Exact equality of every cache tensor (same paths, shapes, dtypes, values)."""
    ta, tb = list(iter_tensors(a)), list(iter_tensors(b))
    return len(ta) == len(tb) and all(pa == pb and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
                                      for (pa, x), (pb, y) in zip(ta, tb))


def cache_fingerprint(cache) -> str:
    """sha256 over the bytes of every cache tensor (moves them to the host: tests and checks, not the timed path)."""
    h = hashlib.sha256()
    for path, t in iter_tensors(cache):
        h.update(path.encode())
        h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def prefill_base(model, ids: list[int], device, chunk: int = 0):
    """Fresh model-native cache holding `ids` (the hybrid model's linear-attention state cannot be cropped, so this is
    built once per history and copied per run). Untimed."""
    if not ids:
        raise ValueError("nothing to prefill")
    cache, step = None, chunk or len(ids)
    for lo in range(0, len(ids), step):
        x = torch.tensor(ids[lo:lo + step], dtype=torch.long, device=device)[None]
        out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
    sync(device)
    return cache


def copy_cache(cache, device) -> tuple[object, float]:
    """An equivalent fresh copy and the seconds it took (untimed by the benchmark; reported)."""
    sync(device)
    t0 = time.perf_counter()
    c = copy.deepcopy(cache)
    sync(device)
    return c, time.perf_counter() - t0


def seed_for(kind: str, history_id: str, repeat: int) -> int:
    return int.from_bytes(hashlib.sha256(f"tap-live-decode:{kind}:{history_id}:{repeat}".encode()).digest()[:4], "big")


def ids_hash(ids) -> str:
    return hashlib.sha256(",".join(str(int(t)) for t in ids).encode()).hexdigest()


@torch.no_grad()
def decode_run(model, cache, first_ids, *, stop_ids, sampling: dict, seed: int, max_new_tokens: int, device, tap=None,
               processor=None, step_events: bool = False, logits_sink=None, sink_steps: int = 0) -> dict:
    """The timed region. `first_ids`: LongTensor [1, T] (the last prompt token on a base-cache copy; or a whole prompt with
    `cache=None` for a fresh-cache generation). Returns elapsed_s, generated ids (the stop token included), whether a stop
    token ended it, the tap's decision record (None without a tap) and, with `step_events`, per-step milliseconds."""
    s = check_sampling(sampling)
    stop_ids = frozenset(int(t) for t in stop_ids)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    cuda = device.type == "cuda"
    gen_ids: list[int] = []
    pairs = []
    rec: dict | None = None
    score_ev = None
    sync(device)
    t0 = time.perf_counter()
    if tap is not None:
        tap.begin_call(max_new_tokens, 0)
    inp, stopped = first_ids, False
    while True:
        if step_events and cuda:
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
        elif step_events:
            ts = time.perf_counter()
        out = model(input_ids=inp, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        logits = out.logits[0, -1]
        if logits_sink is not None and len(gen_ids) < sink_steps:
            logits_sink.append(logits.float().cpu())
        if processor is not None:
            logits = processor(logits, gen_ids, 0)
        tok = sample_token(logits, s, gen)
        if step_events and cuda:
            b.record()
            pairs.append((a, b))
        tid = int(tok.item())                      # the one host synchronisation of this step
        if step_events and not cuda:
            pairs.append((time.perf_counter() - ts) * 1000.0)
        gen_ids.append(tid)
        stopped = tid in stop_ids
        if stopped or len(gen_ids) >= max_new_tokens:
            break
        inp = tok.view(1, 1)
    if tap is not None:
        rec = {}
        if step_events and cuda:
            score_ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            score_ev[0].record()
        tap.end_call(n_generated=len(gen_ids), last_is_stop=stopped, call_index=0, record=rec)
        if score_ev is not None:
            score_ev[1].record()
    sync(device)
    t1 = time.perf_counter()
    res = {"elapsed_s": t1 - t0, "gen_ids": gen_ids, "stopped_naturally": stopped, "record": rec}
    if step_events:
        if cuda:
            res["step_ms"] = [x.elapsed_time(y) for x, y in pairs]
            if score_ev is not None:
                res["score_event_ms"] = score_ev[0].elapsed_time(score_ev[1])
        else:
            res["step_ms"] = pairs
    if tap is not None:
        res["decisions"] = [dict(d) for d in tap.decisions]
    return res
