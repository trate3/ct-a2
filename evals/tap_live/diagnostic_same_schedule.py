"""Same-schedule reference diagnostic. ADDED AFTER OBSERVING THE a2 FAILURE of the correctness trial; untimed; separate from it.

  python -m evals.tap_live.diagnostic_same_schedule --recheck diagnostic_same_schedule.json     (no GPU: recompute the verdicts of a saved file)
  python -m evals.tap_live.diagnostic_same_schedule --model DIR --tokenizer DIR --histories histories.json --artifact PROBE.json \
      --out diagnostic_same_schedule.json [--device cuda] [--histories-limit 2] [--budgets 64,256] [--prefill-chunk N] \
      [--correctness results/tap_live/correctness.json]

The correctness trial's check a2 (live tap score versus the offline score of ONE full forward pass) failed the declared 1e-4 tolerance, and
its check b (cached versus fresh generation) was not token-identical. Both results and thresholds stay as they are. This script asks a
narrower question: does the live tap capture and score the states of its OWN decode schedule that it is meant to?
For each history (the first --histories-limit available ones, as in the correctness trial) and budget:
 1. the base cache is built as `decode_bench` builds it (`prefill_base`, same chunking; the actual chunk size, the number of prefilled
    tokens and the id of the held-back last prompt token are recorded);
 2. a clone is decoded through the real `LiveHFClient` + `LiveProbeTap` path (the clone is injected as the client's cache, so exactly the
    held-back last prompt token is prefilled), recording the generated ids and the tap's captured states and score;
 3. an INDEPENDENT reference from a fresh clone of the same base cache: the last prompt token, then the recorded generated ids teacher-forced one
    token at a time (one forward per token, same cache type), with its own forward hook on the target layer (not the tap object); documented
    alignment (state of generated token j = state at P+j-1, the prompt-final state for j = 0; stop token excluded), scored with the repo's
    offline scorer. Done twice to check that the reference itself repeats bit for bit;
 4. comparison: eligible counts, aligned states (max / mean absolute difference, bitwise equal or not), pooled scores, alarm; the full-forward
    score of the correctness trial's a2 is computed too and all three scores are reported;
 5. cache checks after every decoded clone: the base cache equals a pristine copy taken before (every tensor, recurrent and conv state
    included), the clone has distinct storage, and a mutate-and-restore test on a separate clone leaves the base unchanged.
Verdicts use "consistent with / not consistent with". Same-schedule states are expected to be equal exactly; pass flags are given against 0 and
against the 1e-4 tolerance separately. Nothing here says why the full-forward result differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from evals.reward_hacking import activations as act
from evals.tap_live import correctness_trial as ct
from evals.tap_live import decode_bench as db
from evals.tap_live import decode_core as dc
from evals.tap_live import decode_setup as ds
from evals.tap_live import live_client

LABEL = "added after observing the a2 failure"
TOL = 1e-4


def make_clone(base, device):
    """The clone a decode runs on (a module-level function so that a test can break it)."""
    return dc.copy_cache(base, device)[0]


def storage_distinct(base, clone) -> bool:
    a, b = list(dc.iter_tensors(base)), list(dc.iter_tensors(clone))
    return len(a) == len(b) and all(x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr() for (_, x), (_, y) in zip(a, b) if x.numel())


def mutate_and_restore_test(base, pristine, device) -> dict:
    c = dc.copy_cache(base, device)[0]
    path, t = next((p, x) for p, x in dc.iter_tensors(c) if x.numel())
    idx = (0,) * t.dim()
    orig = t[idx].clone()
    t[idx] = orig + 1
    changed_clone = not dc.caches_equal(base, c)
    base_unchanged = dc.caches_equal(base, pristine)
    t[idx] = orig
    restored = dc.caches_equal(base, c)
    return {"tensor": path, "clone_changed": changed_clone, "base_unchanged_by_clone_mutation": base_unchanged,
            "clone_equals_base_after_restore": restored, "pass": changed_clone and base_unchanged and restored}


def cache_report(base) -> dict:
    paths = [p for p, _ in dc.iter_tensors(base)]
    return {"n_tensors": len(paths), "n_conv_state_tensors": sum("conv_states" in p for p in paths),
            "n_recurrent_state_tensors": sum("recurrent_states" in p for p in paths),
            "n_key_value_tensors": sum(p.endswith("/keys") or p.endswith("/values") for p in paths), "type": "model-native DynamicCache"}


@torch.no_grad()
def reference_states(model, base, last_ids, gen_ids, layer, device) -> torch.Tensor:
    """Teacher-forced one-token-at-a-time reference from a fresh clone of `base`: forward 0 feeds the held-back last prompt token, forward j
    feeds generated token j-1; the last generated token is never fed. Own hook on the target layer. Rows: state that produced token j."""
    clone = make_clone(base, device)
    rows = []
    blk = act.find_decoder_layers(model)[layer - 1]
    handle = blk.register_forward_hook(lambda m, i, o: rows.append((o[0] if isinstance(o, tuple) else o)[0, -1].detach().float().clone()))
    try:
        inp = last_ids
        for j in range(len(gen_ids)):
            out = model(input_ids=inp, past_key_values=clone, use_cache=True, logits_to_keep=1)
            clone = out.past_key_values
            inp = torch.tensor([[gen_ids[j]]], dtype=torch.long, device=device)
    finally:
        handle.remove()
    return torch.stack(rows)


SUCCESS = ("consistent with the live tap capturing and scoring the intended states of its own decode schedule (same-schedule states, "
           "eligible count, score and alarm equal; base cache unchanged, clone independent; the client execution matches the benchmark's decode "
           "loop; the reference repeats bit for bit)")


def core_ok(c: dict) -> bool:
    """The capture / indexing / cache checks of one case, recomputed from its fields."""
    return bool(c["states"]["bitwise_equal"] and c["eligible_tokens"]["equal_tap_vs_reference"] and c["alarm"]["equal_tap_vs_reference"]
                and c["scores"]["tap_vs_reference_pass_vs_1e-4"] and c["scores"]["tap_states_offline_vs_reference_pass_vs_0"]
                and c["live_call"]["prefilled_only_the_held_back_token"] and c["cache_checks_pass"])


def bridge_flags(c: dict) -> dict:
    return {"decode_run_same_tokens_as_live_client": c.get("decode_run_same_tokens_as_live_client"),
            "reference_repeats_bitwise": c.get("reference_repeats_bitwise")}


def verdict_of(c: dict) -> str:
    """Success only if the core checks AND both bridge checks hold. A false or missing bridge check gives an explicit 'unresolved bridge'
    text (never the success string); failing core checks give 'not consistent with'."""
    br = bridge_flags(c)
    if not core_ok(c):
        return ("not consistent with the live tap capturing and scoring the intended states of its own decode schedule; see the failing fields"
                + ("" if all(v is True for v in br.values()) else f" (bridge checks: {br})"))
    if all(v is True for v in br.values()):
        return SUCCESS
    return ("unresolved bridge: the capture, indexing and cache checks of the live-client call pass, but " +
            "; ".join(f"{k} is {v}" for k, v in br.items() if v is not True) +
            ", so the checked client execution is not shown to be the loop used by the cost benchmark / the reference is not shown to repeat; "
            "not a success")


def overall_of(cases: list[dict]) -> str:
    vs = [c["verdict"] for c in cases]
    if all(v == SUCCESS for v in vs):
        return "consistent with the live tap capturing and scoring the intended same-schedule states in every case (bridge and repeat checks included)"
    if any(v.startswith("not consistent with") for v in vs):
        return "not consistent with the live tap capturing and scoring the intended same-schedule states in at least one case"
    return "unresolved bridge in at least one case (see decode_run_same_tokens_as_live_client / reference_repeats_bitwise); not a success"


def run_case(model, tok, fmt, art_path, h, ids, base, pristine, budget, args, device, processor) -> dict:
    tap = ds_tap(art_path, device)
    art = tap.art
    P = len(ids)
    seed = dc.seed_for("diagnostic", h["history_id"], 0)
    last_ids = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
    out = {"history_id": h["history_id"], "budget": budget, "seed": seed}
    # 2. live A1 call through LiveHFClient + LiveProbeTap on a clone of the base cache
    clone = make_clone(base, device)
    out["clone_storage_distinct_from_base"] = storage_distinct(base, clone)
    client = live_client.LiveHFClient(model, tok, fmt, device=device, logits_processor=processor)
    client.reset_episode()
    client._cache, client._processed = clone, list(ids[:-1])
    client.attach_tap(tap)
    try:
        comp = client.complete(h["prompt"], max_tokens=budget, seed=seed, sampling=fmt.sampling)
    finally:
        client.detach_tap()
    gen = list(comp.tokens)
    stopped = comp.stop_type == "eos"
    n = len(gen)
    t = comp.timings
    out["live_call"] = {"n_generated": n, "stopped_naturally": stopped, "n_prefill_tokens": t["n_prefill_tokens"], "n_cached_tokens": t["n_cached_tokens"],
                        "reset_used": t["reset_used"], "prefilled_only_the_held_back_token": t["n_prefill_tokens"] == 1 and not t["reset_used"]}
    dec = tap.decisions[0]
    tap_states = tap.bufs[art.layer][:n].detach().clone()
    # base cache after the decoded clone
    out["base_cache_unchanged_after_decode"] = dc.caches_equal(base, pristine)
    out["mutate_and_restore"] = mutate_and_restore_test(base, pristine, device)
    # the benchmark's own decode loop on another clone: same tokens?
    c2 = make_clone(base, device)
    r2 = dc.decode_run(model, c2, last_ids, stop_ids=fmt.stop_token_ids, sampling=fmt.sampling, seed=seed, max_new_tokens=budget, device=device,
                       tap=None, processor=processor)
    out["decode_run_same_tokens_as_live_client"] = r2["gen_ids"] == gen
    out["base_cache_unchanged_after_second_decode"] = dc.caches_equal(base, pristine)
    # 3. independent same-schedule reference (twice)
    ref1 = reference_states(model, base, last_ids, gen, art.layer, device)
    ref2 = reference_states(model, base, last_ids, gen, art.layer, device)
    out["reference_repeats_bitwise"] = bool(torch.equal(ref1, ref2))
    out["base_cache_unchanged_after_reference"] = dc.caches_equal(base, pristine)
    # 4. comparison
    ref_np = ref1.cpu().numpy()
    off_ref = ct.offline(art, ref_np, stopped)
    tap_np = tap_states.cpu().numpy()
    off_tap_states = ct.offline(art, tap_np, stopped)
    full = ct.full_pass_states(model, ids + gen, art.layer, P - 1, n, device)
    off_full = ct.offline(art, full, stopped)
    same_shape = tuple(tap_states.shape) == tuple(ref1.shape)
    diff = (tap_states.float() - ref1).abs() if same_shape else None
    maxd = float(diff.max()) if same_shape else None
    meand = float(diff.mean()) if same_shape else None

    def sd(a, b):
        return None if a is None or b is None else abs(a - b)
    s_tap, s_ref, s_full = dec["score"], off_ref["score"], off_full["score"]
    out["states"] = {"n_rows": n, "shape_tap": list(tap_states.shape), "shape_reference": list(ref1.shape), "shapes_equal": same_shape,
                     "max_abs_diff": maxd, "mean_abs_diff": meand, "bitwise_equal": bool(same_shape and torch.equal(tap_states.float(), ref1)),
                     "max_abs_state": float(ref1.abs().max()), "pass_vs_0": bool(same_shape and maxd == 0.0),
                     "pass_vs_1e-4": bool(same_shape and maxd <= TOL)}
    out["eligible_tokens"] = {"tap": dec["n_tokens"], "same_schedule_reference": off_ref["eligible_tokens"], "full_forward": off_full["eligible_tokens"],
                              "equal_tap_vs_reference": dec["n_tokens"] == off_ref["eligible_tokens"]}
    out["scores"] = {"tap": s_tap, "tap_states_offline_scorer": off_tap_states["score"], "same_schedule_reference": s_ref, "full_forward": s_full,
                     "abs_diff_tap_vs_reference": sd(s_tap, s_ref), "abs_diff_tap_vs_full_forward": sd(s_tap, s_full),
                     "abs_diff_reference_vs_full_forward": sd(s_ref, s_full),
                     "abs_diff_tap_states_offline_vs_reference": sd(off_tap_states["score"], s_ref),
                     "tap_states_offline_vs_reference_pass_vs_0": sd(off_tap_states["score"], s_ref) == 0.0,
                     "note": "the tap scores in float32 on the device and the offline scorer in float64, so tap vs reference is compared at 1e-4 and "
                             "also reported against 0; the same offline scorer on the tap's states vs on the reference states is expected to be exactly 0",
                     "tap_vs_reference_pass_vs_0": sd(s_tap, s_ref) == 0.0, "tap_vs_reference_pass_vs_1e-4": sd(s_tap, s_ref) is not None and sd(s_tap, s_ref) <= TOL,
                     "tap_vs_full_forward_pass_vs_1e-4": sd(s_tap, s_full) is not None and sd(s_tap, s_full) <= TOL}
    out["alarm"] = {"tap": dec["alarm"], "same_schedule_reference": off_ref["alarm"], "full_forward": off_full["alarm"],
                    "equal_tap_vs_reference": dec["alarm"] == off_ref["alarm"]}
    cache_ok = (out["clone_storage_distinct_from_base"] and out["base_cache_unchanged_after_decode"] and out["base_cache_unchanged_after_second_decode"]
                and out["base_cache_unchanged_after_reference"] and out["mutate_and_restore"]["pass"])
    out["cache_checks_pass"] = bool(cache_ok)
    out["verdict"] = verdict_of(out)
    return out


def ds_tap(art_path, device):
    from evals.tap_live import live_tap
    return live_tap.LiveProbeTap(art_path, device=device, scope="call", time_capture=False, name="A1")


def correctness_statement(path: Path | None) -> dict | None:
    if path is None:
        return None
    c = json.loads(Path(path).read_text())
    hs = c["histories"]
    a_eq = all(x["c_A0_vs_A1"]["identical_token_streams"] for x in hs)
    b_eq = all(x["b_cached_vs_fresh"]["identical_token_streams"] for x in hs)
    return {"source": str(path), "statement": f"A0/A1 token equality {'held' if a_eq else 'did not hold'} in the correctness trial; "
                                              f"cached-vs-fresh generation equality {'held' if b_eq else 'did not hold'}",
            "declared_tolerance_abs_score": c.get("declared_tolerance_abs_score"), "all_declared_checks_pass": c.get("all_declared_checks_pass"),
            "per_history": [{"history_id": x["history_id"], "a1_pass": x["a1_arithmetic_on_tap_states"]["pass"], "a2_pass": x["a2_full_forward_pass"]["pass"],
                             "a2_abs_score_diff": x["a2_full_forward_pass"]["abs_score_diff"],
                             "a2_max_abs_state_diff": x["a2_full_forward_pass"]["max_abs_state_diff_vs_tap_buffer"],
                             "b_identical_token_streams": x["b_cached_vs_fresh"]["identical_token_streams"],
                             "b_first_difference_at": x["b_cached_vs_fresh"]["first_difference_at"],
                             "c_identical_token_streams": x["c_A0_vs_A1"]["identical_token_streams"]} for x in hs],
            "the original a2 result and threshold are unchanged by this diagnostic": True}


def recheck(path: Path) -> Path:
    """No GPU: recompute every case verdict and the overall verdict of a saved diagnostic JSON with the current rule (bridge and
    repeat checks included) and write `<name>.rechecked.json` next to it. The original file is not touched."""
    path = Path(path)
    d = json.loads(path.read_text())
    cases = []
    for c in d["cases"]:
        new = verdict_of(c)
        cases.append({"history_id": c["history_id"], "budget": c["budget"], "original_verdict": c["verdict"], "rechecked_verdict": new,
                      "core_checks_pass": core_ok(c), **bridge_flags(c), "verdict_changed": new != c["verdict"]})
    for c, r in zip(d["cases"], cases):
        c["verdict"] = r["rechecked_verdict"]
    out = {"label": "rechecked with the corrected verdict rule (bridge and repeat checks included); no GPU; the original file is unchanged",
           "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "original_overall_verdict": d["overall_verdict"],
           "rechecked_overall_verdict": overall_of(d["cases"]), "cases": cases, "label_of_original": d.get("label")}
    dest = path.with_name(path.stem + ".rechecked.json")
    dest.write_text(json.dumps(out, indent=1) + "\n")
    return dest


def main(argv=None, *, inject: dict | None = None) -> int:
    argv_list = list(argv if argv is not None else __import__("sys").argv[1:])
    if "--recheck" in argv_list:
        rp = argparse.ArgumentParser()
        rp.add_argument("--recheck", type=Path, required=True)
        dest = recheck(rp.parse_args(argv_list).recheck)
        print(json.dumps({k: v for k, v in json.loads(dest.read_text()).items() if k.endswith("overall_verdict")}, indent=1))
        print(f"wrote {dest}")
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ds.add_common(ap)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--histories-limit", type=int, default=2)
    ap.add_argument("--budgets", default="64,256")
    ap.add_argument("--correctness", type=Path, default=None)
    ap.add_argument("--model-revision", default=None)
    args = ap.parse_args(argv)
    inject = inject or {}
    device, doc, avail, fmt, model, tok, tap = ds.setup(args, inject)
    processor = inject.get("logits_processor")
    budgets = [int(b) for b in args.budgets.split(",")]
    cases, boundaries = [], []
    for h in avail[:args.histories_limit]:
        ids = tok.encode(h["prompt"], add_special_tokens=False)
        base = dc.prefill_base(model, ids[:-1], device, args.prefill_chunk)
        pristine = dc.copy_cache(base, device)[0]
        boundaries.append({"history_id": h["history_id"], "prompt_sha256": h["prompt_sha256"], "prompt_tokens": len(ids),
                           "n_prefilled_tokens": len(ids) - 1, "held_back_last_prompt_token_id": ids[-1],
                           "prefill_chunk": args.prefill_chunk or "single forward of all prefilled tokens",
                           "n_prefill_forwards": -(-(len(ids) - 1) // args.prefill_chunk) if args.prefill_chunk else 1, "cache": cache_report(base)})
        for b in budgets:
            cases.append(run_case(model, tok, fmt, args.artifact, h, ids, base, pristine, b, args, device, processor))
        del base, pristine
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res = {"label": LABEL, "purpose": "separate an indexing/cache/capture error from a difference between cached and full-forward numerical execution; "
                                      "not the overhead benchmark; the correctness trial's a2 result and threshold are unchanged",
           "tolerances": {"same_schedule_states": "exact equality expected; pass flags against 0 and against 1e-4 are separate", "1e-4": TOL},
           "model": {"path": args.model, "revision": db.model_revision(args.model, args.model_revision), "tokenizer": args.tokenizer or args.model,
                     "class": type(model).__name__, "dtype": args.dtype},
           "artifact": tap.describe(), "histories_file": str(args.histories), "histories_sha256": db.sha256_file(args.histories),
           "histories": boundaries, "environment": db.environment(device, model, args), "code_sha256": db.code_hashes(),
           "correctness_trial": correctness_statement(args.correctness), "cases": cases,
           "overall_verdict": overall_of(cases),
           "not_established": ["why the full-forward score differs from the cached-decode score", "equivalence to the offline accuracy implementation",
                               "hardware attestation or certification"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("label", "overall_verdict")}, indent=1))
    for c in cases:
        print(c["history_id"], c["budget"], c["verdict"][:60], "decode_run_same_tokens_as_live_client", c["decode_run_same_tokens_as_live_client"],
              "reference_repeats_bitwise", c["reference_repeats_bitwise"], "states max diff", c["states"]["max_abs_diff"], "scores",
              {k: c["scores"][k] for k in ("tap", "same_schedule_reference", "full_forward")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
