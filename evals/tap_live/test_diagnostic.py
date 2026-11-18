"""CPU tests of the same-schedule diagnostic on the tiny random Qwen3_5 model."""
from __future__ import annotations

import json

import pytest
import torch

from evals.tap_live import diagnostic_same_schedule as diag
from evals.tap_live import live_tap as lt
from evals.tap_live.test_decode_bench import artifact, histories, traj_dir, force  # noqa: F401  (fixtures and helpers)


def args(histories, artifact, out, extra=()):
    return ["--model", "tiny", "--histories", str(histories[0]), "--artifact", str(artifact), "--out", str(out), "--device", "cpu",
            "--dtype", "float32", "--histories-limit", "1", "--budgets", "6,12", *extra]


def run(model, tok, fmt, histories, artifact, tmp_path, extra=(), processor=None):
    out = tmp_path / "diag.json"
    inject = {"model": model, "tokenizer": tok, "fmt": fmt, **({"logits_processor": processor} if processor else {})}
    assert diag.main(args(histories, artifact, out, extra), inject=inject) == 0
    return json.loads(out.read_text())


def test_correct_path_passes_and_the_json_is_complete(model, tok, fmt, histories, artifact, tmp_path):
    corr = tmp_path / "correctness.json"
    corr.write_text(json.dumps({"declared_tolerance_abs_score": 1e-4, "all_declared_checks_pass": False, "histories": [{
        "history_id": "h", "a1_arithmetic_on_tap_states": {"pass": True},
        "a2_full_forward_pass": {"pass": False, "abs_score_diff": 0.005, "max_abs_state_diff_vs_tap_buffer": 1.0},
        "b_cached_vs_fresh": {"identical_token_streams": False, "first_difference_at": 18}, "c_A0_vs_A1": {"identical_token_streams": True}}]}))
    d = run(model, tok, fmt, histories, artifact, tmp_path, ["--correctness", str(corr), "--prefill-chunk", "32"])
    assert d["label"] == "added after observing the a2 failure" and d["overall_verdict"].startswith("consistent with")
    assert "only rounding" not in json.dumps(d) and len(d["cases"]) == 2
    for c in d["cases"]:
        assert c["verdict"].startswith("consistent with") and c["cache_checks_pass"]
        assert c["states"]["bitwise_equal"] and c["states"]["max_abs_diff"] == 0.0 and c["states"]["pass_vs_0"] and c["states"]["pass_vs_1e-4"]
        assert c["eligible_tokens"]["tap"] == c["eligible_tokens"]["same_schedule_reference"] and c["alarm"]["equal_tap_vs_reference"]
        assert c["scores"]["abs_diff_tap_vs_reference"] < 1e-6 and c["scores"]["tap_vs_reference_pass_vs_1e-4"]     # float32 tap vs float64 scorer
        assert c["scores"]["abs_diff_tap_states_offline_vs_reference"] == 0.0 and c["scores"]["tap_states_offline_vs_reference_pass_vs_0"]
        assert set(c["scores"]) >= {"tap", "same_schedule_reference", "full_forward"} and c["scores"]["abs_diff_tap_vs_full_forward"] < 1e-4   # fp32 tiny model
        assert c["reference_repeats_bitwise"] and c["decode_run_same_tokens_as_live_client"]
        assert c["live_call"]["prefilled_only_the_held_back_token"] and c["live_call"]["n_prefill_tokens"] == 1
        assert c["base_cache_unchanged_after_decode"] and c["base_cache_unchanged_after_reference"] and c["clone_storage_distinct_from_base"]
        assert c["mutate_and_restore"]["pass"]
    h = d["histories"][0]
    assert h["prefill_chunk"] == 32 and h["n_prefill_forwards"] == -(-h["n_prefilled_tokens"] // 32) and h["n_prefilled_tokens"] == h["prompt_tokens"] - 1
    assert isinstance(h["held_back_last_prompt_token_id"], int) and h["prompt_sha256"] and h["cache"]["n_recurrent_state_tensors"] > 0
    assert h["cache"]["n_conv_state_tensors"] > 0 and h["cache"]["n_key_value_tensors"] > 0
    assert d["artifact"]["heads"] and d["code_sha256"]["diagnostic_same_schedule.py"] and d["environment"]["torch"] and d["histories_sha256"]
    ct = d["correctness_trial"]
    assert ct["statement"] == "A0/A1 token equality held in the correctness trial; cached-vs-fresh generation equality did not hold"
    assert ct["per_history"][0]["a2_pass"] is False and ct["per_history"][0]["a2_abs_score_diff"] == 0.005


def test_an_off_by_one_in_the_capture_index_is_detected(model, tok, fmt, histories, artifact, tmp_path, monkeypatch):
    orig = lt.LiveProbeTap.end_call

    def shifted(self, **kw):                      # the tap scores the states shifted by one position
        n = kw["n_generated"]
        for L in self.layers:
            self.bufs[L][1:n] = self.bufs[L][0:n - 1].clone()
        return orig(self, **kw)
    monkeypatch.setattr(lt.LiveProbeTap, "end_call", shifted)
    d = run(model, tok, fmt, histories, artifact, tmp_path)
    assert d["overall_verdict"].startswith("not consistent with")
    for c in d["cases"]:
        assert c["verdict"].startswith("not consistent with") and not c["states"]["bitwise_equal"] and c["states"]["max_abs_diff"] > 0
        assert not c["states"]["pass_vs_0"] and not c["scores"]["tap_states_offline_vs_reference_pass_vs_0"]
        assert c["cache_checks_pass"]                                   # the cache is fine; only the capture index is wrong


def test_a_base_cache_mutation_is_detected(model, tok, fmt, histories, artifact, tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "make_clone", lambda base, device: base)       # "clones" that share the base: decoding mutates it
    d = run(model, tok, fmt, histories, artifact, tmp_path)
    for c in d["cases"]:
        assert not c["base_cache_unchanged_after_decode"] and not c["clone_storage_distinct_from_base"] and not c["cache_checks_pass"]
        assert c["verdict"].startswith("not consistent with")


def test_a_stop_token_is_handled(model, tok, fmt, histories, artifact, tmp_path):
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    body = tok.encode(" the user wants me", add_special_tokens=False)
    d = run(model, tok, fmt, histories, artifact, tmp_path, ["--budgets", "40"], processor=force(body, stop))
    c = d["cases"][0]
    assert c["live_call"]["stopped_naturally"] and c["live_call"]["n_generated"] == len(body) + 1
    assert c["eligible_tokens"]["tap"] == c["eligible_tokens"]["same_schedule_reference"] == c["eligible_tokens"]["full_forward"] == len(body)
    assert c["states"]["n_rows"] == len(body) + 1 and c["states"]["bitwise_equal"] and c["verdict"].startswith("consistent with")
    assert c["decode_run_same_tokens_as_live_client"] and c["cache_checks_pass"]


def test_mutate_and_restore_detects_a_shared_clone(model, tok, fmt, histories):
    from evals.tap_live import decode_core as dc
    ids = tok.encode(histories[1]["histories"][0]["prompt"], add_special_tokens=False)
    base = dc.prefill_base(model, ids[:-1], torch.device("cpu"))
    pristine = dc.copy_cache(base, torch.device("cpu"))[0]
    assert diag.mutate_and_restore_test(base, pristine, torch.device("cpu"))["pass"]
    assert diag.storage_distinct(base, dc.copy_cache(base, torch.device("cpu"))[0]) and not diag.storage_distinct(base, base)


def _good_case():
    return {"states": {"bitwise_equal": True}, "eligible_tokens": {"equal_tap_vs_reference": True}, "alarm": {"equal_tap_vs_reference": True},
            "scores": {"tap_vs_reference_pass_vs_1e-4": True, "tap_states_offline_vs_reference_pass_vs_0": True},
            "live_call": {"prefilled_only_the_held_back_token": True}, "cache_checks_pass": True,
            "decode_run_same_tokens_as_live_client": True, "reference_repeats_bitwise": True}


@pytest.mark.parametrize("field", ["decode_run_same_tokens_as_live_client", "reference_repeats_bitwise"])
def test_a_false_bridge_or_repeat_check_is_never_a_success(field):
    c = _good_case()
    assert diag.verdict_of(c) == diag.SUCCESS
    c[field] = False
    v = diag.verdict_of(c)
    assert v != diag.SUCCESS and v.startswith("unresolved bridge") and field in v and not v.startswith("consistent with")
    c["verdict"] = v
    ov = diag.overall_of([{**_good_case(), "verdict": diag.SUCCESS}, c])
    assert ov.startswith("unresolved bridge") and "not a success" in ov
    c[field] = None                                          # a missing field is not a pass either
    assert diag.verdict_of(c).startswith("unresolved bridge")
    bad = _good_case()
    bad["states"]["bitwise_equal"] = False
    bad[field] = False
    assert diag.verdict_of(bad).startswith("not consistent with") and field in diag.verdict_of(bad)


def test_bridge_failure_in_a_real_run_changes_the_overall_verdict(model, tok, fmt, histories, artifact, tmp_path, monkeypatch):
    from evals.tap_live import decode_core as dc
    real = dc.decode_run

    def other_tokens(*a, **k):                              # the benchmark loop disagrees with the client execution
        r = real(*a, **k)
        r["gen_ids"] = r["gen_ids"][:-1] + [(r["gen_ids"][-1] + 1) % 300]
        return r
    monkeypatch.setattr(dc, "decode_run", other_tokens)
    d = run(model, tok, fmt, histories, artifact, tmp_path)
    assert all(not c["decode_run_same_tokens_as_live_client"] and c["verdict"].startswith("unresolved bridge") for c in d["cases"])
    assert d["overall_verdict"].startswith("unresolved bridge") and not d["overall_verdict"].startswith("consistent with")


def test_recheck_recomputes_a_saved_file_without_touching_it(tmp_path):
    ok = {**_good_case(), "history_id": "a@0.25", "budget": 64, "verdict": "consistent with the old wording"}
    bad = {**_good_case(), "history_id": "a@0.5", "budget": 64, "verdict": "consistent with the old wording", "reference_repeats_bitwise": False}
    src = tmp_path / "diagnostic_same_schedule.json"
    src.write_text(json.dumps({"label": "x", "overall_verdict": "consistent with the old wording", "cases": [ok, bad]}))
    before = src.read_bytes()
    assert diag.main(["--recheck", str(src)]) == 0
    assert src.read_bytes() == before
    r = json.loads((tmp_path / "diagnostic_same_schedule.rechecked.json").read_text())
    assert r["rechecked_overall_verdict"].startswith("unresolved bridge") and r["original_overall_verdict"].startswith("consistent with")
    assert r["cases"][0]["rechecked_verdict"] == diag.SUCCESS and r["cases"][1]["rechecked_verdict"].startswith("unresolved bridge")
    assert r["cases"][1]["reference_repeats_bitwise"] is False and r["source_sha256"]
