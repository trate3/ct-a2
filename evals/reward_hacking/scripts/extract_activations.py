"""Extract predictive hidden states for every rollout in trajectory directories (resumable).

    python -m evals.reward_hacking.scripts.extract_activations \
        --model Qwen/Qwen3.6-35B-A3B --revision 995ad96e... TRAJ_DIR [TRAJ_DIR ...]

Output (production): `<out-root>/<dir-name>/<rollout_id>.npz` per input directory, where `out-root`
defaults to `<RH_HOME>/activations` (code default `~/rh/activations`) and `dir-name` is the directory's own name (for a `.../<run>/trajectories`
directory, the run's name). `--out DIR` instead writes one explicit directory (single input directory only).
Layers default to {10, 20, 30} (hidden_states-tuple convention of
`activations`, valid 1..num_hidden_layers-1), stored as float32 (Qwen3.6 hidden states exceed the float16 range).

Resumable and preemption-safe: a `<rollout_id>.npz` is written atomically (tmp + rename) once per rollout and
existing files are skipped; stale `*.tmp` files are deleted at start. SIGTERM/SIGUSR1 finish the rollout in
flight (when it fits in the grace window of the batch scheduler) and exit 75 so the job is requeued/rerun.
Per-run files: `manifest.<i>of<n>.json` (a resume with a different model/revision/layers/dtype is refused),
`progress.<i>of<n>.jsonl` (one record per rollout: tokens, seconds, tokens/s, peak GPU memory) and
`throughput.<i>of<n>.json` (cumulative totals, rewritten after every rollout).

Sizing: one sequence at a time (the hybrid gated-delta-net cache has no padded batching), prefilled in
`--chunk-tokens` chunks (default 16384; weights of the bf16 35B-A3B are ~70 GB, a large GPU has far more). A CUDA
out-of-memory error halves the chunk size (floor 1024) and retries the rollout, so the default errs large.
Run unchanged on CPU with a tiny model (`--model` a local dir, `--dtype float32`) or on a GPU with bf16.

For the real model install `flash-linear-attention` and `causal-conv1d` first: without them
transformers falls back to a slow reference implementation of the gated-delta-net layers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import signal
import time

from evals.reward_hacking import activations as act
from evals.reward_hacking import llm
from evals.reward_hacking import trajectory as tj

DEFAULT_LAYERS = (10, 20, 30)     # Qwen3.6-35B-A3B (40 layers); see default_layers()
DEFAULT_OUT_ROOT = "~/rh/activations"
EXIT_INTERRUPTED = 75
MIN_CHUNK = 1024
# manifest fields that must not change between runs writing into the same directory
PIN_KEYS = ("model", "revision", "tokenizer_revision", "tokenizer_sha256", "layers", "dtype", "store_dtype")


def default_layers(target_model: str = llm.DEFAULT_FORMAT) -> list[int]:
    """The 25/50/75 %-depth layers of the target's decoder stack (qwen3.6: 10, 20, 30 of 40)."""
    return list(llm.FORMATS[target_model].probe_layers)


def dir_name(d: str | os.PathLike) -> str:
    """Output sub-directory name for an input directory: its own name, or its parent's for `trajectories`."""
    p = pathlib.Path(d).expanduser().resolve()
    return p.parent.name if p.name == "trajectories" else p.name


def add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="model revision (recorded; pin it)")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--model-class", default="conditional", choices=["conditional", "auto"],
                    help="conditional: AutoModelForImageTextToText, the checkpoint's native layout (Qwen3_5Moe"
                         "ForConditionalGeneration; loads the vision tower "
                         "too, idle for token-id input); auto: AutoModelForCausalLM "
                         "(text-only class; check for missing-key warnings)")
    ap.add_argument("--target-model", choices=sorted(llm.FORMATS), default=llm.DEFAULT_FORMAT,
                    help="agent model the trajectories came from: selects the default --layers (25/50/75 %% depth); "
                         "the model path itself is --model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--store-dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-tokens", type=int, default=16384)


def load_model(a, layers) -> tuple[object, object, dict]:
    """(model, tokenizer, manifest) from the arguments of `add_model_args`."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, revision=a.tokenizer_revision or a.revision)
    cls = AutoModelForCausalLM
    if a.model_class == "conditional":
        from transformers import AutoModelForImageTextToText as cls   # -> Qwen3_5MoeForConditionalGeneration
    model = cls.from_pretrained(a.model, revision=a.revision, dtype=getattr(torch, a.dtype))
    model.to(a.device).eval()
    man = {"model": a.model, "revision": a.revision, "tokenizer_revision": a.tokenizer_revision or a.revision,
           "tokenizer_sha256": act.tokenizer_digest(tok), "layers": list(layers), "dtype": a.dtype,
           "store_dtype": a.store_dtype, "layer_convention": act.LAYER_CONVENTION, "alignment": act.ALIGNMENT,
           "transformers": transformers.__version__, "torch": torch.__version__, "model_class": type(model).__name__}
    return model, tok, man


def check_manifest(out: pathlib.Path, man: dict) -> None:
    """Refuse to add to a directory written with a different model/layers/dtype."""
    for f in sorted(out.glob("manifest.*.json")):
        old = json.loads(f.read_text())
        diff = {k: (old.get(k), man.get(k)) for k in PIN_KEYS if old.get(k) != man.get(k)}
        if diff:
            raise SystemExit(f"{f} was written with a different configuration {diff}; use a new --out")


class Interrupt:
    """SIGTERM/SIGUSR1 set a flag; the loop stops between rollouts (a second signal exits at once)."""

    def __init__(self):
        self.stop = False
        self._old = {}

    def __enter__(self):
        def handler(signum, _frame):
            if self.stop:
                os._exit(EXIT_INTERRUPTED)
            self.stop = True
            print(f"signal {signum}: finishing the current item, then exiting {EXIT_INTERRUPTED}", flush=True)
        for s in (signal.SIGTERM, signal.SIGUSR1):
            self._old[s] = signal.signal(s, handler)
        return self

    def __exit__(self, *exc):
        for s, h in self._old.items():
            signal.signal(s, h)


STALE_TMP_SECONDS = 3600     # several shards write into ONE output directory: only a tmp file untouched for this long is stale


def remove_stale_tmp(out: pathlib.Path, min_age_s: float = STALE_TMP_SECONDS) -> int:
    """Delete `*.tmp` files that nobody has written for `min_age_s` seconds. A younger one may belong to a shard that is running now
    (shards of one split share the output directory; deleting a live tmp file made that shard fail at os.replace)."""
    import time
    n, now = 0, time.time()
    for p in out.glob("*.tmp"):
        try:
            if now - p.stat().st_mtime < min_age_s:
                continue
        except FileNotFoundError:
            continue
        p.unlink(missing_ok=True)
        n += 1
    return n


def with_oom_backoff(ex, fn):
    """Run `fn()`; on CUDA OOM halve `ex.chunk_tokens` (floor MIN_CHUNK), reset the cache and retry."""
    import torch
    while True:
        try:
            return fn()
        except torch.cuda.OutOfMemoryError:
            if ex.chunk_tokens <= MIN_CHUNK:
                raise
            ex.chunk_tokens = max(MIN_CHUNK, ex.chunk_tokens // 2)
            ex.reset()
            torch.cuda.empty_cache()
            print(f"CUDA OOM: chunk_tokens -> {ex.chunk_tokens}, retrying", flush=True)


class Throughput:
    """Cumulative token/second accounting written next to the activations."""

    def __init__(self, path: pathlib.Path):
        self.path, self.rollouts, self.prefilled, self.generated, self.seconds = path, 0, 0, 0, 0.0

    def add(self, prefilled: int, generated: int, seconds: float, chunk_tokens: int, peak_gb: float | None):
        self.rollouts += 1
        self.prefilled += prefilled
        self.generated += generated
        self.seconds += seconds
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({
            "rollouts": self.rollouts, "tokens_prefilled": self.prefilled, "tokens_generated": self.generated,
            "seconds": round(self.seconds, 2), "prefill_tok_per_s": round(self.prefilled / max(self.seconds, 1e-9), 1),
            "chunk_tokens": chunk_tokens, "peak_gpu_gb": peak_gb}, indent=1))
        tmp.replace(self.path)


def extract_dir(paths, out: pathlib.Path, model, tok, man: dict, a, *, ex=None, stop=None) -> int:
    """Extract the rollouts in `paths` into `out`; returns the number of rollouts written."""
    import torch
    out.mkdir(parents=True, exist_ok=True)
    i, n = map(int, a.shard.split("/"))
    check_manifest(out, man)
    stale = remove_stale_tmp(out)
    paths = sorted(paths)[i::n][:a.limit]
    todo = [p for p in paths if not (out / (pathlib.Path(p).stem + ".npz")).exists()]
    print(f"{out}: {len(paths)} rollouts, {len(paths) - len(todo)} already done, {stale} stale tmp removed", flush=True)
    if not todo:
        return 0
    (out / f"manifest.{i}of{n}.json").write_text(json.dumps(man, indent=1))
    own = ex is None
    ex = ex or act.CallExtractor(model, man["layers"], chunk_tokens=a.chunk_tokens)
    cuda = a.device.startswith("cuda")
    tput = Throughput(out / f"throughput.{i}of{n}.json")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    done = 0
    try:
        with open(out / f"progress.{i}of{n}.jsonl", "a") as log:
            for p in todo:
                if stop is not None and stop.stop:
                    break
                traj = tj.Trajectory.from_json(open(p).read())
                if any("tokens" not in e.data for e in traj.of(tj.MODEL_CALL)):
                    # No recorded generated-token ids (e.g. a SWE log without raw API calls): exact extraction is
                    # impossible, so skip and report instead of failing the whole shard.
                    rec = {"rollout_id": pathlib.Path(p).stem, "skipped": "model calls without recorded token ids"}
                    with open(out / f"skipped.{i}of{n}.jsonl", "a") as sk:
                        sk.write(json.dumps(rec) + "\n")
                    print(rec, flush=True)
                    continue
                t0 = time.perf_counter()
                res = with_oom_backoff(ex, lambda: act.extract(model, tok, traj, man["layers"],
                                                               store_dtype=a.store_dtype, extractor=ex))
                if cuda:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                act.save_extracted(out / (pathlib.Path(p).stem + ".npz"), res)
                computed = sum(c["computed"] for c in res.meta["calls"])
                peak = round(torch.cuda.max_memory_allocated() / 2**30, 1) if cuda else None
                rec = {"rollout_id": traj.meta.get("rollout_id"), "n_calls": len(res.meta["calls"]),
                       "n_generated": len(res), "tokens_prefilled": computed,
                       "calls_recomputed": sum(not c["reused"] for c in res.meta["calls"]),
                       "seconds": round(dt, 2), "prefill_tok_per_s": round(computed / max(dt, 1e-9), 1),
                       "chunk_tokens": ex.chunk_tokens, "peak_gpu_gb": peak}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                tput.add(computed, len(res), dt, ex.chunk_tokens, peak)
                done += 1
                print(rec, f"cumulative {tput.prefilled / max(tput.seconds, 1e-9):.0f} tok/s "
                      f"({done}/{len(todo)})", flush=True)
    finally:
        if own:
            ex.close()
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", help="trajectory directories (*.json)")
    add_model_args(ap)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="hidden_states-tuple layers (default: the --target-model's probe layers: "
                         "qwen3.6 10 20 30, qwen3.8-27b 16 32 48)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--out", default=None, help="one explicit output directory (single input directory)")
    g.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="<out-root>/<dir-name>/ per input directory")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th rollout starting at i")
    a = ap.parse_args(argv)

    if a.out and len(a.dirs) != 1:
        ap.error("--out takes exactly one input directory; use --out-root for several")
    jobs = [(pathlib.Path(a.out) if a.out else pathlib.Path(a.out_root).expanduser() / dir_name(d),
             sorted(glob.glob(f"{d}/*.json"))) for d in a.dirs]
    names = [str(o) for o, _ in jobs]
    if len(set(names)) != len(names):
        ap.error(f"input directories map to the same output directory: {names}")
    if all((o / (pathlib.Path(p).stem + ".npz")).exists() for o, ps in jobs for p in
           sorted(ps)[int(a.shard.split("/")[0])::int(a.shard.split("/")[1])][:a.limit]):
        print("nothing to do", flush=True)
        return 0

    a.layers = a.layers or default_layers(a.target_model)
    model, tok, man = load_model(a, a.layers)
    act.check_layers(model, a.layers)          # refuse layers outside 1..num_hidden_layers-1 before any rollout
    man["target_model"] = a.target_model
    ex = act.CallExtractor(model, a.layers, chunk_tokens=a.chunk_tokens)
    rc = 0
    with Interrupt() as stop:
        for out, paths in jobs:
            extract_dir(paths, out, model, tok, man, a, ex=ex, stop=stop)
            if stop.stop:
                rc = EXIT_INTERRUPTED
                break
    ex.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
