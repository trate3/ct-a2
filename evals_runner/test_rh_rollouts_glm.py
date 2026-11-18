"""`rh-rollouts --target-model glm-5.3-flash`: manifest, template pin check, resume guards (Qwen untouched)."""

from __future__ import annotations

import argparse
import json

import pytest

from evals.reward_hacking import llm, rollout, tasks
from evals.reward_hacking import trajectory as tj
from evals_runner import rh_rollouts as rr

TASKS = ["lcbhard_0", "lcbhard_1"]


def make_client_cls(template_path):
    class FakeClient:
        def __init__(self, url=None, **kw):
            pass

        def props(self):
            return {"chat_template": template_path.read_text(), "total_slots": 4,
                    "default_generation_settings": {"n_ctx": 4096}}
    return FakeClient


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(rr.sandbox, "ensure_image", lambda *a, **k: None)
    monkeypatch.setattr(rr.sandbox, "sandbox_tools_binary", lambda *a, **k: None)
    monkeypatch.setattr(rr.sandbox, "cleanup_leftovers", lambda *a, **k: 0)
    monkeypatch.setattr(rr.tasks, "load_tasks", lambda variant="conflicting", test_sentence="nomod": [
        tasks.Task(task_id=t, variant=variant, prompt="", test="", original_test="", entry_point="f",
                   instruction_prompt=tasks.instruction_prompt(test_sentence)) for t in TASKS])
    monkeypatch.setattr(rr.tasks, "load_splits", lambda: {t: "train" for t in TASKS})


def make_args(tmp_path, *extra):
    p = argparse.ArgumentParser()
    rr.configure_parser(p)
    return p.parse_args(["--tasks", ",".join(TASKS), "--out", str(tmp_path / "out"), "--rollouts", "1", *extra])


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, task, *, rollout_id, seed, client, config, meta, **kw):
        self.calls.append({"id": rollout_id, "config": config, "kw": kw})
        t = tj.Trajectory(meta={"task_id": task.task_id, "variant": task.variant, "rollout_id": rollout_id,
                                "seed": seed, "partition": meta["partition"], "config": config.as_dict()})
        t.add(tj.ROLLOUT_END, {"status": "ok", "reason": "", "label": None, "timing": {}, "wall_seconds": 0.0})
        return t


@pytest.mark.parametrize("template", [llm.GLM_TEMPLATE_PATH, llm.GLM_GGUF_TEMPLATE_PATH])
def test_glm_manifest_and_loop_arguments(tmp_path, monkeypatch, template):
    monkeypatch.setattr(rr.llm, "CompletionClient", make_client_cls(template))
    rec = Recorder()
    assert rr.run(make_args(tmp_path, "--target-model", "glm-5.3-flash"), run_one=rec) == 0
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert man["target_model_name"] == "glm-5.3-flash"
    assert man["template_sha256"] == llm.get_format("glm-5.3-flash").template_digest()
    assert man["chat_format"]["stop_token_ids"] == [154820, 154827, 154829]
    assert man["rollout_config"]["sampling"]["top_k"] == 0
    assert all(isinstance(c["kw"].get("fmt"), llm.GLMFormat) for c in rec.calls) and len(rec.calls) == 2


def test_glm_rejects_the_qwen_template_on_the_server(tmp_path, monkeypatch):
    monkeypatch.setattr(rr.llm, "CompletionClient", make_client_cls(llm.TEMPLATE_PATH))
    with pytest.raises(SystemExit, match="chat template"):
        rr.run(make_args(tmp_path, "--target-model", "glm-5.3-flash"), run_one=Recorder())


def test_qwen_default_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(rr.llm, "CompletionClient", make_client_cls(llm.TEMPLATE_PATH))
    rec = Recorder()
    assert rr.run(make_args(tmp_path), run_one=rec) == 0
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert man["target_model_name"] == "qwen3.6" and man["template_sha256"] == llm.template_digest()
    assert man["rollout_config"]["sampling"] == rollout.SAMPLING
    assert all(c["kw"] == {} for c in rec.calls)          # run_one called exactly as before


def test_resume_refuses_a_different_target_model(tmp_path, monkeypatch):
    monkeypatch.setattr(rr.llm, "CompletionClient", make_client_cls(llm.GLM_GGUF_TEMPLATE_PATH))
    rr.run(make_args(tmp_path, "--target-model", "glm-5.3-flash"), run_one=Recorder())
    monkeypatch.setattr(rr.llm, "CompletionClient", make_client_cls(llm.TEMPLATE_PATH))
    with pytest.raises(SystemExit, match="target model changed"):
        rr.run(make_args(tmp_path), run_one=Recorder())


def test_resume_refuses_a_changed_reasoning_effort(tmp_path, monkeypatch):
    monkeypatch.setattr(rr.llm, "CompletionClient", make_client_cls(llm.GLM_GGUF_TEMPLATE_PATH))
    rr.run(make_args(tmp_path, "--target-model", "glm-5.3-flash"), run_one=Recorder())
    with pytest.raises(SystemExit, match="chat format"):
        rr.run(make_args(tmp_path, "--target-model", "glm-5.3-flash", "--reasoning-effort", "high"),
               run_one=Recorder())


def test_reasoning_effort_must_be_valid_for_the_target(tmp_path, monkeypatch):
    monkeypatch.setattr(rr.llm, "CompletionClient", make_client_cls(llm.TEMPLATE_PATH))
    with pytest.raises(SystemExit, match="reasoning_effort must be one of"):
        rr.run(make_args(tmp_path, "--reasoning-effort", "low"), run_one=Recorder())      # qwen3.6 has none
    with pytest.raises(SystemExit, match="reasoning_effort must be one of"):
        rr.run(make_args(tmp_path, "--target-model", "qwen3.8-27b", "--reasoning-effort", "max"),
               run_one=Recorder())


def test_qwen38_manifest_vllm_stop_ids_and_default_effort(tmp_path, monkeypatch):
    class VClient:
        engine = "vllm"

        def __init__(self, url=None, **kw):
            self.kw = kw
            VClient.last = self

        def identity(self):
            return {"engine": "vllm", "engine_version": "0.31.0", "model_alias": "qwen3.8-27b",
                    "server_template_sha256": None}
    monkeypatch.setattr(rr.llm, "VLLMCompletionClient", VClient)
    rec = Recorder()
    assert rr.run(make_args(tmp_path, "--target-model", "qwen3.8-27b", "--engine", "vllm"), run_one=rec) == 0
    assert VClient.last.kw["stop_token_ids"] == (248046, 248044)
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    f = llm.get_format("qwen3.8-27b")
    assert man["target_model_name"] == "qwen3.8-27b" and man["template_sha256"] == f.template_digest()
    assert man["chat_format"]["reasoning_effort"] == "xhigh (template default)"
    assert man["chat_format"]["hf_revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert man["rollout_config"]["sampling"] == rollout.SAMPLING       # Qwen3.8 card preset == 3.6's
    assert all(isinstance(c["kw"].get("fmt"), llm.Qwen38Format) for c in rec.calls)
    # an explicit effort is a different experiment
    with pytest.raises(SystemExit, match="chat format"):
        rr.run(make_args(tmp_path, "--target-model", "qwen3.8-27b", "--engine", "vllm",
                         "--reasoning-effort", "low"), run_one=Recorder())
