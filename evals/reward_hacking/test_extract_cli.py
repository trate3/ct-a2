"""Production extraction CLI: output layout, throughput logging, resume safety, interruption, OOM back-off."""
from __future__ import annotations

import json
import os
import signal

import pytest
import torch

from evals.reward_hacking import activations as act
from evals.reward_hacking import tiny_qwen as tq
from evals.reward_hacking.scripts import extract_activations as cli


@pytest.fixture(scope="module")
def tok():
    return tq.tiny_tokenizer()


@pytest.fixture(scope="module")
def model(tok):
    return tq.tiny_model(len(tok))


def test_dir_name_uses_the_run_name_for_trajectories_dirs():
    assert cli.dir_name("/x/s1_pilot/trajectories") == "s1_pilot"
    assert cli.dir_name("/x/c1_trajs") == "c1_trajs"


def _setup(tok, model, tmp_path):
    mdir = tmp_path / "model"
    model.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    dirs = []
    for run, rids in (("runA", ["a0", "a1"]), ("runB", ["b0"])):
        t = tmp_path / run / "trajectories"
        t.mkdir(parents=True)
        for rid in rids:
            (t / f"{rid}.json").write_text(tq.synthetic_trajectory(tok, rollout_id=rid).to_json())
        dirs.append(str(t))
    base = ["--model", str(mdir), "--model-class", "auto", "--dtype", "float32", "--device", "cpu"]
    return dirs, base


def test_out_root_layout_throughput_and_defaults(tok, model, tmp_path):
    dirs, base = _setup(tok, model, tmp_path)
    root = tmp_path / "acts"
    assert cli.main([*dirs, *base, "--layers", "2", "3", "--out-root", str(root), "--chunk-tokens", "64"]) == 0
    assert sorted(p.name for p in root.iterdir()) == ["runA", "runB"]
    a = root / "runA"
    assert sorted(p.name for p in a.glob("*.npz")) == ["a0.npz", "a1.npz"]
    e = act.load_extracted(a / "a0.npz")
    assert str(e.hidden[2].dtype) == "float32"                        # production default storage (float16 overflows on Qwen3.6)
    tp = json.loads((a / "throughput.0of1.json").read_text())
    assert tp["rollouts"] == 2 and tp["tokens_prefilled"] > 0 and tp["prefill_tok_per_s"] > 0
    rec = [json.loads(line) for line in (a / "progress.0of1.jsonl").read_text().splitlines()]
    assert len(rec) == 2 and {"seconds", "prefill_tok_per_s", "chunk_tokens", "peak_gpu_gb"} <= set(rec[0])
    assert json.loads((a / "manifest.0of1.json").read_text())["layers"] == [2, 3]
    # resume: nothing is recomputed, and nothing is loaded (no model needed to answer "nothing to do")
    m = (a / "a0.npz").stat().st_mtime_ns
    assert cli.main([*dirs, "--model", "/does/not/exist", "--layers", "2", "3", "--out-root", str(root)]) == 0
    assert (a / "a0.npz").stat().st_mtime_ns == m


def test_default_layers_are_the_registered_candidates():
    assert cli.DEFAULT_LAYERS == (10, 20, 30)
    assert cli.DEFAULT_OUT_ROOT == "~/rh/activations"


def test_stale_tmp_removed_and_changed_configuration_refused(tok, model, tmp_path):
    dirs, base = _setup(tok, model, tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    (out / "a0.npz.tmp").write_bytes(b"half written")
    argv = [dirs[0], *base, "--layers", "2", "--out", str(out), "--store-dtype", "float32"]
    assert cli.main(argv) == 0
    assert not list(out.glob("*.tmp")) and (out / "a0.npz").exists()
    (out / "a1.npz").unlink()
    with pytest.raises(SystemExit, match="different configuration"):
        cli.main([dirs[0], *base, "--layers", "2", "3", "--out", str(out), "--store-dtype", "float32"])
    with pytest.raises(SystemExit):                                  # --out with several input dirs
        cli.main([*dirs, *base, "--out", str(out)])


def test_interrupt_flag_stops_between_rollouts_and_returns_75(tok, model, tmp_path, monkeypatch):
    dirs, base = _setup(tok, model, tmp_path)
    out = tmp_path / "o"
    real = cli.extract_dir

    def stop_after_first(paths, out_, model_, tok_, man, a, *, ex, stop):
        orig = act.save_extracted

        def save_and_signal(*args, **kw):
            orig(*args, **kw)
            os.kill(os.getpid(), signal.SIGUSR1)                  # the handler installed by Interrupt
        monkeypatch.setattr(act, "save_extracted", save_and_signal)
        return real(paths, out_, model_, tok_, man, a, ex=ex, stop=stop)
    monkeypatch.setattr(cli, "extract_dir", stop_after_first)
    rc = cli.main([dirs[0], *base, "--layers", "2", "--out", str(out)])
    assert rc == cli.EXIT_INTERRUPTED and [p.name for p in out.glob("*.npz")] == ["a0.npz"]
    monkeypatch.undo()
    assert cli.main([dirs[0], *base, "--layers", "2", "--out", str(out)]) == 0      # resumes with the second
    assert sorted(p.name for p in out.glob("*.npz")) == ["a0.npz", "a1.npz"]


def test_oom_backoff_halves_chunk_and_retries_down_to_the_floor():
    class Ex:
        chunk_tokens = 16384
        resets = 0

        def reset(self):
            self.resets += 1
    ex, calls = Ex(), []

    def fn():
        calls.append(ex.chunk_tokens)
        if len(calls) < 3:
            raise torch.cuda.OutOfMemoryError("oom")
        return "ok"
    assert cli.with_oom_backoff(ex, fn) == "ok" and calls == [16384, 8192, 4096] and ex.resets == 2

    ex.chunk_tokens = cli.MIN_CHUNK

    def always():
        raise torch.cuda.OutOfMemoryError("oom")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        cli.with_oom_backoff(ex, always)
