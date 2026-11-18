"""CPU tests of reconstruction, alignment, layer convention, cache reuse and I/O (tiny Qwen3.5-MoE)."""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest
import torch

from evals.reward_hacking import activations as act
from evals.reward_hacking import tiny_qwen as tq

LAYERS = [1, 2, 3]


@pytest.fixture(scope="module")
def tok():
    return tq.tiny_tokenizer()


@pytest.fixture(scope="module")
def model(tok):
    return tq.tiny_model(len(tok))


@pytest.fixture(scope="module")
def traj(tok):
    return tq.synthetic_trajectory(tok)


def test_transformers_supports_architecture(model):
    # native support, no custom code: the real hybrid (linear + full attention, sparse MoE) classes
    assert type(model).__name__ == "Qwen3_5MoeForCausalLM"
    assert model.config.model_type == "qwen3_5_moe_text"   # text part of the qwen3_5_moe config
    assert {"linear_attention", "full_attention"} <= set(model.config.get_text_config().layer_types)


def test_reconstruct_uses_prompt_tokens_and_recorded_gen_ids(tok, traj):
    calls = act.reconstruct(traj, tok)
    mcs = traj.of("model_call")
    assert [c.call_index for c in calls] == [0, 1, 2, 3]
    for c, e in zip(calls, mcs):
        assert c.prompt_ids == tok.encode(e.data["prompt"], add_special_tokens=False)
        assert c.gen_ids == e.data["tokens"]
        assert len(c.channel) == len(c.gen_ids)
    # the non-canonical call's recorded ids are not the canonical tokenization of its text
    assert calls[2].gen_ids[:-1] != tok.encode(mcs[2].data["generation"], add_special_tokens=False)


def test_channel_labels(tok, traj):
    c = act.reconstruct(traj, tok)[0]
    ids = np.asarray(c.gen_ids)
    close, copen, cclose = (tok.convert_tokens_to_ids(x) for x in ("</think>", "<tool_call>", "</tool_call>"))
    i_close = int(np.nonzero(ids == close)[0][0])
    i_open, i_cclose = int(np.nonzero(ids == copen)[0][0]), int(np.nonzero(ids == cclose)[0][0])
    ch = c.channel
    assert (ch[:i_close + 1] == act.REASONING).all()                    # includes </think> itself
    assert (ch[i_close + 1:i_open] == act.VISIBLE).all() and i_open > i_close + 1
    assert (ch[i_open:i_cclose + 1] == act.TOOL_CALL).all()
    assert ch[-1] == act.END and (ch[i_cclose + 1:-1] == act.VISIBLE).all()
    # a prompt without an open <think> starts the generation in the visible channel
    marks = act._marker_ids(tok)
    p = tok.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
    assert act.label_channels(p, ids[:3], marks).tolist() == [act.VISIBLE] * 3


def test_prefix_and_decode_report(tok, traj):
    r = act.consistency_report(traj, tok)
    assert [x["ok"] for x in r["prefix"]] == [True, True, False]
    assert r["prefix"][2]["diverges_in_gen"] is True
    assert all(x["ok"] for x in r["decode"])


def test_layer_convention_matches_hidden_states_tuple(model):
    ids = torch.randint(0, 300, (1, 40))
    n = model.config.get_text_config().num_hidden_layers
    ex = act.CallExtractor(model, list(range(1, n)))
    try:
        states, _ = ex.process(ids[0, :10].tolist(), ids[0, 10:].tolist())
    finally:
        ex.close()
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    for L in range(1, n):
        assert torch.allclose(states[L], hs[L][0, 9:39], atol=1e-5), L
    for bad in (0, n, n + 1):
        with pytest.raises(ValueError):
            act.check_layers(model, [bad])


def test_predictive_alignment_vs_manual_forward(tok, model, traj):
    out = act.extract(model, tok, traj, LAYERS)
    calls = act.reconstruct(traj, tok)
    assert len(out) == sum(len(c.gen_ids) for c in calls)
    off = 0
    for c in calls:
        full = torch.tensor([c.full_ids])
        with torch.no_grad():
            hs = model(full, output_hidden_states=True).hidden_states
        P, n = len(c.prompt_ids), len(c.gen_ids)
        for L in LAYERS:
            got = out.hidden[L][off:off + n]
            want = hs[L][0, P - 1:P + n - 1].numpy()           # state that produced token j
            np.testing.assert_allclose(got, want, atol=2e-5)
            at = hs[L][0, P:P + n].numpy()                     # the wrong ("at") alignment differs a lot
            assert np.abs(got - at).max() > 100 * np.abs(got - want).max()
        np.testing.assert_array_equal(out.token_id[off:off + n], c.gen_ids)
        np.testing.assert_array_equal(out.position[off:off + n], np.arange(n))
        np.testing.assert_array_equal(out.channel[off:off + n], c.channel)
        off += n


def test_kv_reuse_equals_full_recompute(tok, model, traj):
    calls = act.reconstruct(traj, tok)
    ex = act.CallExtractor(model, LAYERS, chunk_tokens=64, pad_to_chunk=False)   # the reuse path
    reused, a = [], []
    ex.reset()
    for c in calls:
        s, info = ex.process(c.prompt_ids, c.gen_ids)
        a.append({L: v.clone() for L, v in s.items()})
        reused.append(info["reused"])
    # call 1 reuses call 0; call 2 reuses call 1; call 3 follows the non-canonical call 2 -> recompute
    assert reused == [False, True, True, False]
    ex2 = act.CallExtractor(model, LAYERS, chunk_tokens=10_000, pad_to_chunk=False)
    for c, ref in zip(calls, a):
        ex2.reset()                                            # force a full recompute every call
        s, info = ex2.process(c.prompt_ids, c.gen_ids)
        assert not info["reused"] and info["computed"] == len(c.full_ids)
        for L in LAYERS:
            torch.testing.assert_close(s[L], ref[L], atol=5e-5, rtol=1e-4)
    ex.close(), ex2.close()


def test_padded_fixed_shape_chunks_equal_full_forward(tok, model, traj):
    """Padding a short final chunk AFTER the real tokens leaves the real positions' states unchanged."""
    calls = act.reconstruct(traj, tok)
    padded = act.CallExtractor(model, LAYERS, chunk_tokens=64)                     # default: pad, no reuse
    full = act.CallExtractor(model, LAYERS, chunk_tokens=10_000, pad_to_chunk=False)
    for c in calls:
        a, ia = padded.process(c.prompt_ids, c.gen_ids)
        full.reset()
        b, _ = full.process(c.prompt_ids, c.gen_ids)
        assert not ia["reused"]
        for L in LAYERS:
            torch.testing.assert_close(a[L], b[L], atol=5e-5, rtol=1e-4)
    padded.close(), full.close()


def test_single_sequence_extraction_equals_per_call(tok, model, traj):
    """One pass over the final sequence gives each prefix call's states; a non-prefix call gets its own pass."""
    calls = act.reconstruct(traj, tok)
    ex = act.CallExtractor(model, LAYERS, chunk_tokens=64)
    one = list(act.extract_calls(ex, calls, single_sequence=True))
    per = list(act.extract_calls(ex, calls, single_sequence=False))
    assert [c.call_index for c, _, _ in one] == [c.call_index for c, _, _ in per]
    assert any(i["single_sequence"] for _, _, i in one)
    assert sum(i["computed"] for _, _, i in one if i["single_sequence"]) == len(calls[-1].full_ids)
    for (c, sa, _), (_, sb, _) in zip(one, per):
        for L in LAYERS:
            assert sa[L].shape[0] == len(c.gen_ids)
            torch.testing.assert_close(sa[L], sb[L], atol=5e-5, rtol=1e-4)
    ex.close()


def test_roundtrip_float16_and_overflow(tok, model, traj, tmp_path):
    out = act.extract(model, tok, traj, [2], store_dtype="float16")
    assert out.hidden[2].dtype == np.float16
    p = tmp_path / "r.npz"
    act.save_extracted(p, out)
    assert not list(tmp_path.glob("*.tmp"))
    back = act.load_extracted(p, layers=[2])
    np.testing.assert_array_equal(back.hidden[2], out.hidden[2])
    np.testing.assert_array_equal(back.channel, out.channel)
    assert back.meta["layers"] == [2] and "predictive" in back.meta["alignment"]
    with pytest.raises(OverflowError):
        act._to_store(torch.full((2, 2), 1e6), "float16")


def test_cli_extracts_and_resumes(tok, model, traj, tmp_path):
    from evals.reward_hacking.scripts import extract_activations as cli
    mdir, tdir, out = tmp_path / "model", tmp_path / "trajs", tmp_path / "out"
    model.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    tdir.mkdir()
    (tdir / "syn__r0.json").write_text(traj.to_json())
    argv = [str(tdir), "--model", str(mdir), "--layers", "2", "3", "--out", str(out), "--dtype", "float32",
            "--store-dtype", "float32", "--device", "cpu", "--model-class", "auto"]
    cli.main(argv)
    f = out / "syn__r0.npz"
    assert f.exists() and (out / "manifest.0of1.json").exists()
    ref = act.extract(model, tok, traj, [2, 3])
    got = act.load_extracted(f)
    np.testing.assert_allclose(got.hidden[3], ref.hidden[3], atol=1e-5)
    mtime = f.stat().st_mtime_ns
    cli.main(argv)                                              # resume: nothing recomputed
    assert f.stat().st_mtime_ns == mtime


def test_conditional_generation_wrapper_gives_same_states(tok, model, traj):
    """The checkpoint's native class (with vision tower) and the text-only class agree."""
    from transformers import Qwen3_5MoeForConditionalGeneration
    cfg = tq.tiny_config(len(tok))
    cfg.vision_config.depth = 1
    cg = Qwen3_5MoeForConditionalGeneration(cfg).eval()
    sd = {("model.language_model." + k[len("model."):] if k.startswith("model.") else k): v
          for k, v in model.state_dict().items()}
    missing = cg.load_state_dict(sd, strict=False)
    assert not [k for k in missing.missing_keys if "visual" not in k]
    a = act.extract(model, tok, traj, [2])
    b = act.extract(cg, tok, traj, [2])
    np.testing.assert_allclose(a.hidden[2], b.hidden[2], atol=2e-5)


def test_cli_conditional_class(tok, model, traj, tmp_path):
    from transformers import Qwen3_5MoeForConditionalGeneration
    from evals.reward_hacking.scripts import extract_activations as cli
    cfg = tq.tiny_config(len(tok))
    cfg.vision_config.depth = 1
    mdir, tdir = tmp_path / "m", tmp_path / "t"
    Qwen3_5MoeForConditionalGeneration(cfg).eval().save_pretrained(mdir)
    tok.save_pretrained(mdir)
    tdir.mkdir()
    (tdir / "r.json").write_text(traj.to_json())
    cli.main([str(tdir), "--model", str(mdir), "--layers", "2", "--out", str(tmp_path / "o"), "--dtype", "float32",
              "--store-dtype", "float32", "--device", "cpu"])
    assert (tmp_path / "o" / "r.npz").exists()
