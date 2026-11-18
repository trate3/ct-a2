"""Engine clients: vLLM request/response mapping (mocked HTTP; response shapes follow vLLM v0.31.0
`CompletionResponse`), llama.cpp KV-pool admission control and retry, and `make_client`."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest

import evals.reward_hacking.llm as llm_mod
from evals.reward_hacking.llm import (KVBudget, VLLMCompletionClient, make_client, vllm_sampling)
from evals.reward_hacking.conftest import requires_sandbox
from evals.reward_hacking.rollout import SAMPLING

IM_END = 248046


def resp(payload, status=200, text=""):
    r = Mock(status_code=status, text=text)
    r.json = Mock(return_value=payload)
    r.raise_for_status = Mock() if status < 400 else Mock(side_effect=llm_mod.requests.exceptions.HTTPError(str(status)))
    return r


def vllm_payload(**over):
    choice = {"index": 0, "text": "Let me think.\n</think>\n\n<tool_call>\n<function=bash>", "logprobs": None,
              "finish_reason": "stop", "stop_reason": IM_END, "token_ids": [11, 22, 33, IM_END],
              "prompt_token_ids": [1, 2, 3], "prompt_logprobs": None, "routed_experts": None}
    choice.update(over.pop("choice", {}))
    d = {"id": "cmpl-x", "object": "text_completion", "created": 1, "model": "qwen3.6-35b-a3b",
         "choices": [choice], "service_tier": None, "system_fingerprint": None,
         "usage": {"prompt_tokens": 3, "total_tokens": 7, "completion_tokens": 4,
                   "prompt_tokens_details": {"cached_tokens": 2}}}
    d.update(over)
    return d


@pytest.fixture
def post(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda _: None)
    p = Mock()
    monkeypatch.setattr(llm_mod.requests, "post", p)
    return p


def test_sampling_mapping_renames_repeat_penalty_and_rejects_unknown():
    assert vllm_sampling(SAMPLING) == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                       "presence_penalty": 0.0, "repetition_penalty": 1.0}
    assert vllm_sampling({"temperature": 0.0, "top_k": 1}) == {"temperature": 0.0, "top_k": 1}
    with pytest.raises(ValueError):
        vllm_sampling({"mirostat": 2})


def test_vllm_request_body(post):
    post.return_value = resp(vllm_payload())
    c = VLLMCompletionClient("http://x:8000/", model="m")
    c.complete("<|im_start|>user\nhi<|im_end|>\n", max_tokens=100, seed=7, sampling=SAMPLING)
    (url,), kw = post.call_args
    body = kw["json"]
    assert url == "http://x:8000/v1/completions"
    assert body["prompt"] == "<|im_start|>user\nhi<|im_end|>\n"        # our rendered text, not chat
    assert body["model"] == "m" and body["max_tokens"] == 100 and body["seed"] == 7
    assert body["return_token_ids"] is True
    assert body["skip_special_tokens"] is False and body["spaces_between_special_tokens"] is False
    assert body["add_special_tokens"] is False and body["stream"] is False and body["n"] == 1
    assert body["stop_token_ids"] == [248046, 248044]
    assert body["repetition_penalty"] == 1.0 and "repeat_penalty" not in body
    assert "messages" not in body


def test_vllm_eos_response_parsing(post):
    post.return_value = resp(vllm_payload())
    comp = VLLMCompletionClient("http://x", model="m").complete("p", max_tokens=10, seed=1, sampling=SAMPLING)
    assert comp.stop_type == "eos"
    assert "<|im_end|>" not in comp.text               # the server excludes the stop token from text...
    assert comp.tokens == [11, 22, 33, IM_END]         # ...but returns its id, like llama.cpp's `tokens`
    assert comp.prompt_tokens == 3 and comp.cached_tokens == 2
    assert "</think>" in comp.text and "<tool_call>" in comp.text


def test_vllm_length_and_missing_details(post):
    p = vllm_payload(choice={"finish_reason": "length", "stop_reason": None})
    p["usage"].pop("prompt_tokens_details")
    post.return_value = resp(p)
    comp = VLLMCompletionClient("http://x", model="m").complete("p", max_tokens=4, seed=1, sampling={})
    assert comp.stop_type == "limit" and comp.cached_tokens == 0


def test_vllm_missing_token_ids_fails_loudly(post):
    post.return_value = resp(vllm_payload(choice={"token_ids": None}))
    with pytest.raises(RuntimeError, match="token_ids"):
        VLLMCompletionClient("http://x", model="m").complete("p", max_tokens=4, seed=1, sampling={})


def test_vllm_logprobs_fallback(post):
    toks = ["token_id:11", "token_id:22", f"token_id:{IM_END}"]
    post.return_value = resp(vllm_payload(choice={"token_ids": None, "logprobs": {"tokens": toks}}))
    c = VLLMCompletionClient("http://x", model="m", token_mode="logprobs")
    comp = c.complete("p", max_tokens=4, seed=1, sampling={})
    body = post.call_args.kwargs["json"]
    assert body["logprobs"] == 0 and body["return_tokens_as_token_ids"] is True and "return_token_ids" not in body
    assert comp.tokens == [11, 22, IM_END]


def test_vllm_identity_and_model_discovery(monkeypatch):
    def get(url, timeout=None):
        if url.endswith("/version"):
            return resp({"version": "0.31.0"})
        return resp({"data": [{"id": "qwen3.6-35b-a3b", "root": "/models/q", "max_model_len": 196608}]})
    monkeypatch.setattr(llm_mod.requests, "get", get)
    ident = VLLMCompletionClient("http://x").identity()
    assert ident["engine"] == "vllm" and ident["engine_version"] == "0.31.0"
    assert ident["model_alias"] == "qwen3.6-35b-a3b" and ident["n_ctx"] == 196608
    assert ident["server_template_sha256"] is None


def test_vllm_discovers_model_name_for_request(monkeypatch, post):
    monkeypatch.setattr(llm_mod.requests, "get", lambda u, timeout=None: resp({"data": [{"id": "served"}]}))
    post.return_value = resp(vllm_payload())
    VLLMCompletionClient("http://x").complete("p", max_tokens=1, seed=1, sampling={})
    assert post.call_args.kwargs["json"]["model"] == "served"


def test_vllm_tokenize(post):
    post.return_value = resp({"count": 2, "max_model_len": 9, "tokens": [5, 6]})
    assert VLLMCompletionClient("http://x", model="m").tokenize("hi") == [5, 6]
    assert post.call_args.args[0] == "http://x/tokenize"
    assert post.call_args.kwargs["json"]["add_special_tokens"] is False


def test_vllm_400_is_not_retried(post):
    post.return_value = resp({}, status=400)
    with pytest.raises(llm_mod.requests.exceptions.HTTPError):
        VLLMCompletionClient("http://x", model="m").complete("p", max_tokens=1, seed=1, sampling={})
    assert post.call_count == 1


def test_make_client(monkeypatch):
    assert type(make_client("llamacpp", "http://x")) is llm_mod.CompletionClient
    assert type(make_client("vllm", "http://x")) is VLLMCompletionClient
    with pytest.raises(ValueError):
        make_client("tgi", "http://x")
    monkeypatch.setenv("CT_RH_KV_POOL_TOKENS", "1000")
    assert make_client("llamacpp", "http://x").kv_budget.pool == 1000


# ---- llama.cpp: pool exhaustion

def test_llamacpp_retries_kv_exhausted_500(post):
    ok = resp({"content": "a", "tokens": [1], "stop_type": "eos", "tokens_evaluated": 5, "timings": {}})
    post.side_effect = [resp({}, status=500, text='{"error":{"message":"Context size has been exceeded."}}'), ok]
    comp = llm_mod.CompletionClient("http://x", retries=2).complete("p", max_tokens=1, seed=1, sampling={})
    assert comp.text == "a" and post.call_count == 2


def test_llamacpp_other_500_is_not_retried(post):
    post.return_value = resp({}, status=500, text="Compute error.")
    with pytest.raises(llm_mod.requests.exceptions.HTTPError):
        llm_mod.CompletionClient("http://x", retries=2).complete("p", max_tokens=1, seed=1, sampling={})
    assert post.call_count == 1


def test_kv_budget_bounds_in_flight_tokens():
    b = KVBudget(1000, chars_per_token=1.0)
    assert b.cost("x" * 100, 200) == 301
    assert b.cost("x" * 5000, 200) == 1000                  # a request bigger than the pool is capped
    live, peak, lock = 0, 0, threading.Lock()

    def worker():
        nonlocal live, peak
        n = b.cost("x" * 100, 200)
        b.acquire(n)
        with lock:
            live += n
            peak = max(peak, live)
        time.sleep(0.02)
        with lock:
            live -= n
        b.release(n)
    ts = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak <= 1000 and b.peak <= 1000 and b.used == 0 and b.waits > 0


def test_kv_budget_admits_oversized_request_when_idle():
    b = KVBudget(100)
    b.acquire(100)
    b.release(100)
    assert b.used == 0


def test_llamacpp_client_releases_budget_on_error(post):
    post.side_effect = llm_mod.requests.ConnectionError("down")
    c = llm_mod.CompletionClient("http://x", retries=0, kv_pool_tokens=10_000)
    with pytest.raises(llm_mod.requests.ConnectionError):
        c.complete("p", max_tokens=5, seed=1, sampling={})
    assert c.kv_budget.used == 0


# ---------------------------------------------------------------------------------------------
# GLM-5.3-Flash over vLLM (stop ids from the format; stop token kept in token_ids, dropped from text)

GLM_FMT = llm_mod.get_format("glm-5.3-flash")
GLM_OBS, GLM_USER, GLM_EOT = 154829, 154827, 154820


def glm_payload(text, stop_id, ids=(11, 22), finish="stop", **choice):
    return vllm_payload(choice={"text": text, "finish_reason": finish,
                                "stop_reason": stop_id if finish == "stop" else None,
                                "token_ids": list(ids) + ([stop_id] if finish == "stop" else []), **choice})


def test_glm_client_sends_format_stop_ids_and_sampling(post):
    post.return_value = resp(glm_payload("x", GLM_USER))
    c = VLLMCompletionClient("http://x", model="m", fmt=GLM_FMT)
    c.complete("[gMASK]<sop><|user|>hi<|assistant|><think>", max_tokens=5, seed=1, sampling=GLM_FMT.sampling)
    body = post.call_args[1]["json"]
    assert body["stop_token_ids"] == [GLM_EOT, GLM_USER, GLM_OBS] and body["add_special_tokens"] is False
    assert body["prompt"].count("[gMASK]") == 1 and body["skip_special_tokens"] is False
    assert (body["temperature"], body["top_p"], body["top_k"]) == (1.0, 0.95, 0)
    # Qwen default unchanged
    assert VLLMCompletionClient("http://x").stop_token_ids == [248046, 248044]
    assert VLLMCompletionClient("http://x", fmt=llm_mod.get_format("qwen3.6")).stop_token_ids == [248046, 248044]


@pytest.mark.parametrize("sid,boundary", [(GLM_OBS, "<|observation|>"), (GLM_USER, "<|user|>"), (GLM_EOT, "<|endoftext|>")])
def test_glm_stop_reason_to_expected_prefix(post, sid, boundary):
    post.return_value = resp(glm_payload("thought</think>done", sid))
    comp = VLLMCompletionClient("http://x", model="m", fmt=GLM_FMT).complete("P", max_tokens=5, seed=1, sampling={})
    assert comp.stop_type == "eos" and comp.tokens[-1] == sid and comp.timings["stop_token_id"] == sid
    assert GLM_FMT.expected_next_prefix("P", comp) == "Pthought</think>done" + boundary


def test_glm_stop_token_missing_from_token_ids_is_restored_and_mismatch_flagged(post):
    d = glm_payload("a", GLM_OBS)
    d["choices"][0]["token_ids"] = [11, 22]            # a server that dropped the stop id
    post.return_value = resp(d)
    comp = VLLMCompletionClient("http://x", model="m", fmt=GLM_FMT).complete("P", max_tokens=5, seed=1, sampling={})
    assert comp.tokens == [11, 22] and comp.timings["stop_token_mismatch"] == 22
    d["choices"][0]["token_ids"] = []
    post.return_value = resp(d)
    assert VLLMCompletionClient("http://x", model="m", fmt=GLM_FMT).complete(
        "P", max_tokens=5, seed=1, sampling={}).tokens == [GLM_OBS]


def test_glm_length_stop_has_no_boundary(post):
    post.return_value = resp(glm_payload("cut", None, finish="length"))
    comp = VLLMCompletionClient("http://x", model="m", fmt=GLM_FMT).complete("P", max_tokens=2, seed=1, sampling={})
    assert comp.stop_type == "limit" and GLM_FMT.expected_next_prefix("P", comp) == "Pcut"


def _glm_call(name, **kw):
    return f"<tool_call>{name}" + "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>"
                                         for k, v in kw.items()) + "</tool_call>"


@requires_sandbox
def test_glm_three_call_scripted_rollout_over_vllm_client(post):
    from evals.reward_hacking import rollout as rollout_mod
    from evals.reward_hacking import trajectory as tj
    from evals.reward_hacking.conftest import fixture_task
    from evals.reward_hacking.tasks import Task
    d = fixture_task("conflicting", "lcbhard_0")
    task = Task(task_id=d["task_id"], variant="conflicting", prompt=d["prompt"], test=d["test"],
                original_test=d["test"], entry_point=d["entry_point"], instruction_prompt=d["instruction_prompt"])
    gens = [("Look.</think>" + _glm_call("bash", cmd="ls"), GLM_OBS),
            ("More.</think>" + _glm_call("bash", cmd="cat test.py"), GLM_OBS),
            ("Done.</think>submitting" + _glm_call("submit", answer="DONE"), GLM_OBS)]
    prompts = []

    def fake_post(url, json=None, timeout=None):
        prompts.append(json["prompt"])
        text, sid = gens[len(prompts) - 1]
        return resp(glm_payload(text, sid))
    post.side_effect = fake_post
    client = VLLMCompletionClient("http://x", model="m", fmt=GLM_FMT)
    traj = rollout_mod.run_rollout(task, rollout_id="v1", seed=1, client=client, fmt=GLM_FMT,
                                   config=rollout_mod.RolloutConfig(max_attempts=1, sampling=dict(GLM_FMT.sampling)))
    calls = traj.of(tj.MODEL_CALL)
    assert len(calls) == 3 and [c.data["prefix_consistent"] for c in calls] == [None, True, True]
    assert prompts[1].startswith(prompts[0] + gens[0][0] + "<|observation|>")
    assert prompts[2].startswith(prompts[1] + gens[1][0] + "<|observation|>")
    assert all(p.count("[gMASK]<sop>") == 1 and p.startswith("[gMASK]<sop>") for p in prompts)


def test_vllm_no_cap_sends_remaining_context_not_null(monkeypatch, post):
    """max_tokens=None must become max_model_len - prompt length: vLLM's own default for a missing or null
    max_tokens on /v1/completions is 16 tokens (every 'uncapped' call stopped after 16, 2026-10-21)."""
    monkeypatch.setattr(llm_mod.requests, "get",
                        lambda u, timeout=None: resp({"data": [{"id": "m", "max_model_len": 1000}]}))
    tok = resp({"count": 3, "max_model_len": 1000, "tokens": [5, 6, 7]})
    post.side_effect = [tok, resp(vllm_payload())]
    c = VLLMCompletionClient("http://x", model="m")
    c.complete("abc", max_tokens=None, seed=1, sampling={})
    assert post.call_args.kwargs["json"]["max_tokens"] == 997
    post.side_effect = [resp(vllm_payload())]
    c.complete([1] * 100, max_tokens=None, seed=1, sampling={})          # token-id prompt: no tokenize call
    assert post.call_args.kwargs["json"]["max_tokens"] == 900
