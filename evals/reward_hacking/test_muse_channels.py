"""Channel labelling for Muse Glimmer 30B (ATEM wire format) and a guard that the Qwen labelling is unchanged.

The real-tokenizer tests read the tokenizer from $MUSE_TOKENIZER_DIR (no default: the tests skip unless it is set)
and use the real generated token ids of data/muse_raw_fixtures.json; they skip when the tokenizer is not present.
The synthetic tests use made-up ids and a stub decoder and always run.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

from evals.reward_hacking import activations as act
from evals.reward_hacking import tiny_qwen as tq
from evals.reward_hacking import trajectory as tj

R, T, V, E = act.REASONING, act.TOOL_CALL, act.VISIBLE, act.END
FIXTURES = json.loads((pathlib.Path(__file__).parent / "data" / "muse_raw_fixtures.json").read_text())
TOK_DIR = pathlib.Path(os.environ.get("MUSE_TOKENIZER_DIR", "/nonexistent"))
START, EOM, EOT, MSG = 200022, 200007, 200008, 200023


@pytest.fixture(scope="module")
def mtok():
    if not (TOK_DIR / "tokenizer.json").exists():
        pytest.skip(f"Muse tokenizer not found in {TOK_DIR} (set MUSE_TOKENIZER_DIR)")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOK_DIR))


@pytest.fixture(scope="module")
def marks(mtok):
    return act.muse_marks(mtok)


def _prompt(mtok):
    return mtok.encode("<|begin_of_text|><|start|>user<|message|>hi<|eot|><|start|>assistant", add_special_tokens=False)


def _label(mtok, marks, ids, prompt=None):
    return act.label_channels(_prompt(mtok) if prompt is None else prompt, ids, marks)


def _segments(ids):
    """(start, end_inclusive) of each segment of a completion: it ends at `<|eom|>`; the last at `<|eot|>`."""
    out, s = [], 0
    for i, t in enumerate(ids):
        if t in (EOM, EOT):
            out.append((s, i))
            s = i + 1
    return out


def _samples(case):
    return [s["token_ids"] for s in FIXTURES["cases"][case]["samples"]]


def _all_samples():
    return [(k, i, s["token_ids"]) for k, c in FIXTURES["cases"].items() for i, s in enumerate(c["samples"])]


# ---------------------------------------------------------------------------
# real tokenizer, real generations

def test_special_ids_and_marks(mtok, marks):
    assert (marks["start"], marks["eom"], marks["eot"], marks["message"]) == (START, EOM, EOT, MSG)
    assert marks["end"] == frozenset({EOT}) and marks["format"] == "atem"
    assert act.channel_marks(mtok, "muse-glimmer-30b")["format"] == "atem"
    assert act.channel_marks(mtok, "qwen3.6") is None and act.channel_marks(mtok) is None


def test_every_token_of_every_sample_gets_exactly_one_label(mtok, marks):
    for case, i, ids in _all_samples():
        ch = _label(mtok, marks, ids)
        assert ch.dtype == np.int8 and len(ch) == len(ids), (case, i)
        assert set(ch.tolist()) <= {R, T, V, E}, (case, i)
        assert ch[-1] == E and ids[-1] == EOT and (ch[:-1] != E).all(), (case, i)       # END only on the stop token


def test_single_call_reasoning_then_atem_block_as_tool_call(mtok, marks):
    for ids in _samples("single_call"):
        ch = _label(mtok, marks, ids)
        (a0, a1), (b0, b1) = _segments(ids)
        assert ids[a1] == EOM and ids[b1] == EOT
        assert (ch[a0:a1 + 1] == R).all()                        # ` to=self<|message|>` + reasoning body + `<|eom|>`
        assert (ch[b0:b1] == T).all() and ch[b1] == E            # `<|start|>assistant to=get_weather<|message|>` + block
        # the atem block itself is tool_call
        text = mtok.decode(ids[b0:b1])
        assert "<atem:function_calls>" in text and text.rstrip().endswith("</atem:function_calls>")
        assert ids[a0 + 2] == MSG and ch[a0] == R and ch[a0 + 2] == R          # header tokens belong to the reasoning
        assert ids[b0] == START and ch[b0] == T                                   # `<|start|>` of the tool segment
        assert mtok.decode(ids[a0:a0 + 2]) == " to=self"


def test_header_of_later_segment_is_labelled_with_that_segments_channel(mtok, marks):
    ids = _samples("single_call")[0]
    ch = _label(mtok, marks, ids)
    (_, a1), (b0, _) = _segments(ids)
    j_msg = b0 + ids[b0:].index(MSG)
    assert mtok.decode(ids[b0:j_msg]) == "<|start|>assistant to=get_weather"
    assert (ch[a1 + 1:j_msg + 1] == T).all() and ch[a1] == R


def test_parallel_calls_two_tool_segments(mtok, marks):
    for ids in _samples("parallel_calls"):
        ch = _label(mtok, marks, ids)
        segs = _segments(ids)
        assert len(segs) == 3
        assert (ch[segs[0][0]:segs[0][1] + 1] == R).all()
        assert (ch[segs[1][0]:segs[1][1] + 1] == T).all()       # first call ends with <|eom|>, still tool_call
        assert ids[segs[1][1]] == EOM
        assert (ch[segs[2][0]:segs[2][1]] == T).all() and ch[-1] == E


def test_plain_answer_visible_final_segment(mtok, marks):
    for ids in _samples("plain_answer"):
        ch = _label(mtok, marks, ids)
        (a0, a1), (b0, b1) = _segments(ids)
        assert (ch[a0:a1 + 1] == R).all()
        assert (ch[b0:b1] == V).all() and ch[b1] == E
        assert mtok.decode(ids[b0:b1]).startswith("<|start|>assistant to=user<|message|>")
        assert not (ch == T).any()


def test_code_call_multiline_labels(mtok, marks):
    for ids in _samples("code_call_multiline"):
        ch = _label(mtok, marks, ids)
        (a0, a1), (b0, b1) = _segments(ids)
        assert (ch[a0:a1 + 1] == R).all() and (ch[b0:b1] == T).all() and ch[b1] == E


def test_channel_counts_partition_the_sequence(mtok, marks):
    for _, _, ids in _all_samples():
        ch = _label(mtok, marks, ids)
        assert sum(int((ch == c).sum()) for c in (R, T, V, E)) == len(ids)


def test_truncated_mid_reasoning_is_all_reasoning(mtok, marks):
    ids = _samples("single_call")[0]
    (_, a1), _ = _segments(ids)
    cut = ids[:a1 - 20]                                          # no `<|eom|>`, no `<|eot|>`
    ch = _label(mtok, marks, cut)
    assert len(ch) == len(cut) and (ch == R).all()
    assert (_label(mtok, marks, ids[:a1]) == R).all()            # cut right before `<|eom|>`
    assert (_label(mtok, marks, ids[:a1 + 1]) == R).all()        # `<|eom|>` itself is still reasoning


def test_truncated_inside_header_and_inside_tool_body(mtok, marks):
    ids = _samples("single_call")[0]
    (_, a1), (b0, b1) = _segments(ids)
    j_msg = b0 + ids[b0:].index(MSG)
    ch = _label(mtok, marks, ids[:j_msg])                        # cut before `<|message|>`: header read from its text
    assert (ch[:a1 + 1] == R).all() and (ch[a1 + 1:] == T).all() and len(ch) == j_msg
    ch = _label(mtok, marks, ids[:b0 + 3 + 6])                   # a few tool-body tokens, nothing else
    assert (ch[b0:] == T).all()
    ch = _label(mtok, marks, ids[:a1 + 1] + [START])             # cut right after the `<|start|>` of the next segment
    assert (ch[:a1 + 1] == R).all() and len(ch) == a1 + 2 and ch[-1] in (R, T, V)
    assert len(_label(mtok, marks, [])) == 0


def test_prompt_side_state_prefill_continuation(mtok, marks):
    """A prompt ending inside an assistant segment continues that body in its channel."""
    p = mtok.encode("<|start|>user<|message|>hi<|eot|><|start|>assistant to=self<|message|>Let me think", add_special_tokens=False)
    body = mtok.encode(" more.", add_special_tokens=False)
    ch = act.label_channels(p, body + [EOM], marks)
    assert (ch == R).all()
    p2 = mtok.encode("<|start|>assistant to=user<|message|>Hello", add_special_tokens=False)
    assert (act.label_channels(p2, body + [EOT], marks)[:-1] == V).all()
    p3 = mtok.encode("<|start|>assistant to=self<|message|>x<|eom|>", add_special_tokens=False)   # after eom: new header
    seg = mtok.encode("<|start|>assistant to=user<|message|>ok", add_special_tokens=False) + [EOT]
    assert act.label_channels(p3, seg, marks).tolist() == [V] * (len(seg) - 1) + [E]


def test_history_render_samples_label_cleanly(mtok, marks):
    h = FIXTURES["history_render"]
    p = mtok.encode(h["prompt"], add_special_tokens=False)
    assert p[-2:] == [START, mtok.encode("assistant", add_special_tokens=False)[0]]
    for s in h["samples"]:
        ch = act.label_channels(p, s["token_ids"], marks)
        assert len(ch) == len(s["token_ids"]) and ch[-1] == E and (ch[:-1] != E).all()


def test_reconstruct_with_muse_marks_and_prompt_from_text(mtok, marks):
    ids = _samples("single_call")[0]
    prompt_text = "<|begin_of_text|><|start|>user<|message|>hi<|eot|><|start|>assistant"
    traj = tj.Trajectory(meta={"task_id": "t", "rollout_id": "r"})
    traj.add(tj.MODEL_CALL, {"prompt": prompt_text, "generation": "", "tokens": ids, "stop_type": "eos"}, call_index=0)
    c = act.reconstruct(traj, mtok, marks=act.channel_marks(mtok, "muse-glimmer-30b"))[0]
    np.testing.assert_array_equal(c.channel, _label(mtok, marks, ids))
    with pytest.raises(ValueError, match="not a single token"):                  # the Qwen markers do not exist here
        act.reconstruct(traj, mtok)


# ---------------------------------------------------------------------------
# synthetic ids (no tokenizer files needed)

# ids: 1 = ' to', 2 = '=self', 3 = '=user', 4 = '=tool', 5 = 'assistant', 10.. = body words
_WORDS = {1: " to", 2: "=self", 3: "=user", 4: "=get_weather", 5: "assistant", START: "<|start|>"}


def _stub_marks():
    m = {"format": "atem", "start": START, "eom": EOM, "eot": EOT, "message": MSG, "end": frozenset({EOT}),
         "decode": lambda ids: "".join(_WORDS.get(int(t), f"w{int(t)}") for t in ids)}
    return m


def test_synthetic_state_machine_without_tokenizer():
    m = _stub_marks()
    prompt = [99, START, 5]
    gen = [1, 2, MSG, 10, 11, EOM, START, 5, 1, 4, MSG, 12, 13, EOM, START, 5, 1, 3, MSG, 14, EOT]
    ch = act.label_channels(prompt, gen, m)
    assert ch.tolist() == [R] * 6 + [T] * 8 + [V] * 6 + [E]
    # no marks key "format": the Qwen path is untouched by the stub (it would need the Qwen keys)
    with pytest.raises(KeyError):
        act.label_channels(prompt, gen, {k: v for k, v in m.items() if k != "format"})


def test_synthetic_missing_eom_and_unknown_recipient():
    m = _stub_marks()
    # `<|start|>` in a body without `<|eom|>` opens a new segment; header without to= is visible; stop in a header
    gen = [1, 2, MSG, 10, START, 5, 1, 4, MSG, 11, 12, START, 5, MSG, 13, EOT]
    ch = act.label_channels([START, 5], gen, m)
    assert ch.tolist() == [R] * 4 + [T] * 7 + [V] * 4 + [E]
    assert act.label_channels([START, 5], [1, 2, EOT], m).tolist() == [R, R, E]
    assert act.label_channels([START, 5], [1, 2, MSG], m).tolist() == [R, R, R]


# ---------------------------------------------------------------------------
# Qwen unchanged

def _reference_qwen_labels(prompt_ids, gen_ids, marks):
    """Verbatim copy of the Qwen labelling before the Muse change (the regression reference)."""
    t_open, t_close = marks[act.THINK_OPEN], marks[act.THINK_CLOSE]
    state = act.VISIBLE
    for tid in reversed(prompt_ids):
        if tid == t_open:
            state = act.REASONING
            break
        if tid == t_close:
            break
    out = np.empty(len(gen_ids), dtype=np.int8)
    for j, tid in enumerate(gen_ids):
        if tid in marks["end"]:
            out[j] = act.END
        elif state == act.REASONING:
            out[j] = act.REASONING
            if tid == t_close:
                state = act.VISIBLE
        elif state == act.TOOL_CALL:
            out[j] = act.TOOL_CALL
            if tid == marks[act.CALL_CLOSE]:
                state = act.VISIBLE
        else:
            if tid == marks[act.CALL_OPEN]:
                state = act.TOOL_CALL
                out[j] = act.TOOL_CALL
            else:
                out[j] = act.VISIBLE
    return out


def test_qwen_labelling_unchanged_on_the_existing_fixture():
    tok = tq.tiny_tokenizer()
    traj = tq.synthetic_trajectory(tok)
    marks = act._marker_ids(tok)
    assert "format" not in marks
    calls = act.reconstruct(traj, tok)
    assert len(calls) == 4
    seen = set()
    for c in calls:
        ref = _reference_qwen_labels(c.prompt_ids, c.gen_ids, marks)
        np.testing.assert_array_equal(c.channel, ref)
        np.testing.assert_array_equal(act.label_channels(c.prompt_ids, c.gen_ids, marks), ref)
        seen |= set(c.channel.tolist())
    assert seen == {R, T, V, E}                                  # the fixture exercises every Qwen channel
    # explicit channel_marks / extract plumbing leaves Qwen on the default path
    assert act.channel_marks(tok, "qwen3.6") is None
    for c, c2 in zip(calls, act.reconstruct(traj, tok, marks=act.channel_marks(tok, "qwen3.6"))):
        np.testing.assert_array_equal(c.channel, c2.channel)


# ---------------------------------------------------------------------------
# plumbing through the extraction script and the parity gate

def test_extract_dir_passes_muse_marks_only_for_the_muse_target(mtok, tmp_path, monkeypatch):
    import argparse
    from evals.reward_hacking.scripts import extract_activations as cli
    seen = []

    def fake_extract(model, tok, traj, layers, **kw):
        seen.append(kw.get("marks"))
        return act.Extracted(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32),
                             np.zeros(0, np.int8), {}, {"calls": []})

    class Ex:
        chunk_tokens = 8
        def close(self): pass

    monkeypatch.setattr(act, "extract", fake_extract)
    monkeypatch.setattr(act, "save_extracted", lambda path, ex: pathlib.Path(path).write_bytes(b""))
    traj = tj.Trajectory(meta={"task_id": "t", "rollout_id": "r"})
    traj.add(tj.MODEL_CALL, {"prompt": "x", "generation": "", "tokens": [EOT]}, call_index=0)
    src = tmp_path / "in"
    src.mkdir()
    (src / "r.json").write_text(traj.to_json())
    for target, expect_muse in (("muse-glimmer-30b", True), ("qwen3.6", False)):
        a = argparse.Namespace(shard="0/1", limit=None, device="cpu", store_dtype="float32", chunk_tokens=8,
                               target_model=target)
        man = {"layers": [1], "model": "m"}
        cli.extract_dir([str(src / "r.json")], tmp_path / target, None, mtok, man, a, ex=Ex())
        assert (seen[-1] is not None and seen[-1]["format"] == "atem") == expect_muse
