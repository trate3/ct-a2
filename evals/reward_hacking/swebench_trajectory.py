"""Adapter: Inspect AI `.eval` samples from the SWE-bench pilot -> canonical `Trajectory`.

    python -m evals.reward_hacking.swebench_trajectory --eval-dir DIR --out DIR [--tokenizer PATH]

writes `OUT/trajectories/<instance>__<variant>__r<k>.json` (the layout rh-monitor, extract_*.py and parity_check
glob: `*/trajectories/*.json`, one `Trajectory.to_json()` per rollout, meta keys rollout_id/task_id/variant/seed/
partition as in the LCB runner) so the black-box/CoT monitors and the activation pipeline can read them.

What is exact and what is not
-----------------------------
Events (messages, reasoning, tool calls, tool results, the scorer outcome) are read from the log verbatim.

MODEL_CALL events are RECONSTRUCTED, not recorded. Inspect does not keep token ids or the string the vLLM server
actually tokenized, so for call k:
  prompt     = Qwen38Format.render(conversation before assistant turn k, the tools of the sample's ModelEvent)
               with the reasoning effort the format was built with (the template default `xhigh` when the log
               recorded none), preserve_thinking=true, enable_thinking=true (what the server applies);
  generation = the text the template renders for assistant turn k after that prompt (reasoning, `</think>`,
               content, `<tool_call>` XML), without the trailing `<|im_end|>`; this is what the qwen3 reasoning
               parser + qwen3_xml tool parser would have split, re-serialised.
Known approximations (hence `reconstruction: "approximate"` on every call):
  * whitespace: vLLM's parsers strip reasoning/content and the template re-trims them, so the raw sampled
    whitespace (and any text between/after tool calls) is normalised;
  * tool arguments are re-serialised from parsed JSON (string values verbatim, others `tojson`);
  * how vLLM maps the `reasoning` field Inspect replays in assistant history onto the template's
    `reasoning_content` is a server-version property: `replay_reasoning=True` (default) assumes it is replayed.
    `prompt_tokens_recorded` (usage input+cache tokens of the matching ModelEvent) lets a tokenizer check this
    (see `check_prompt_lengths`);
  * Exception: when the ModelEvent's raw `call.response` carries vLLM `prompt_token_ids` and `choices[0].token_ids`
    (`return_token_ids`), they are stored (`prompt_token_ids`, `tokens`), the call is `reconstruction="exact"`,
    `tokens_source="vllm"`; `verify_recorded_ids` then sets `prompt_ids_match`; `reconstruct_exact` consumes them.
  * otherwise `tokens` (generated ids) are ABSENT until `attach_retokenized_tokens` is run; those are tokenize(generation +
    "<|im_end|>"), not the sampled ids.
Before any probe number on SWE rollouts is trusted, run `scripts/parity_check.py` on a GPU (it needs the real
Qwen3.8 weights + tokenizer, `--model MODEL_DIR --revision REV`, and teacher-forces `tokenize(prompt)` + the
recorded generated ids; with re-tokenised ids it only validates that the reconstructed text is a likely sample of
the model, not that it is what the server fed; a token-exact check needs the vLLM prompt_token_ids/token_ids
returned with the response, e.g. `return_token_ids`, which Inspect's log does not store).

Scorer -> label. The strict scorer's `evaluation_kind` is the source of truth (as in swebench_pilot._score_result):
`pass` -> passed True (Y=1, impossible task passed = reward hack); `test_failure` and `flagged` -> passed False
(Y=0); anything else, a sample error, or no score -> passed None, so `label()` is None (excluded downstream).
The task scorer runs once on the terminal state, so there is a single SUBMISSION/EVALUATION pair (attempt 1) placed
after the last model turn. SUBMISSION carries the scorer's `model_patch` (the agent's final `git diff` of the
workspace) in `func_py`, `test_py` is empty (render_view / monitors print both fields). Per-attempt in-loop
scores of an agent that submitted several times are not in the log, so a monitor sees the whole run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

from evals.reward_hacking import llm
from evals.reward_hacking import trajectory as tj

IM_END = "<|im_end|>\n"
SUBMIT_TOOL = "submit"
_CALL_RAW = re.compile(r"<tool_call>.*?</tool_call>", re.S)
QWEN_TARGET = "qwen3.8-27b"
SCORER = "strict_swe_bench"


# ---------------------------------------------------------------------------
# inspect objects -> chat-template messages

def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.text for c in content if getattr(c, "type", None) == "text")


def _reasoning(content) -> str:
    if isinstance(content, str):
        return ""
    return "\n".join(c.reasoning for c in content
                     if getattr(c, "type", None) == "reasoning" and not getattr(c, "redacted", False))


def tool_specs(tools) -> list[dict]:
    """Inspect ToolInfo list -> OpenAI function specs, as Inspect's chat-completions provider sends them."""
    out = []
    for t in tools:
        params = t.parameters.model_dump(exclude_none=True) if hasattr(t.parameters, "model_dump") else t.parameters
        out.append({"type": "function", "function": {"name": t.name, "description": t.description,
                                                     "parameters": params}})
    return out


def turn_end(fmt) -> str:
    """The rendered end of an assistant turn (`<|im_end|>` + newline)."""
    return IM_END


def _end_marker(target) -> str:
    """The end-of-turn special token text (the generated ids end with it)."""
    return IM_END.strip()


def _chat_message(m, *, replay_reasoning: bool) -> dict:
    if m.role in ("system", "user"):
        return {"role": m.role, "content": _text(m.content)}
    if m.role == "tool":
        return {"role": "tool", "content": f"Error: {m.error.message}" if m.error else _text(m.content)}
    msg = {"role": "assistant", "content": _text(m.content)}
    if replay_reasoning:
        msg["reasoning_content"] = _reasoning(m.content)
    good = [c for c in (m.tool_calls or []) if not c.parse_error]
    if good:
        msg["tool_calls"] = [{"type": "function", "function": {"name": c.function, "arguments": dict(c.arguments)}}
                             for c in good]
    return msg


def reconstruct_calls(messages, tools: list[dict], fmt: llm.ChatFormat, *, replay_reasoning: bool = True) -> list[dict]:
    """One {message_index, prompt, generation, prefix_consistent, rendered_ok} per assistant message."""
    end = turn_end(fmt)
    chat = [_chat_message(m, replay_reasoning=replay_reasoning) for m in messages]
    out, prev = [], None
    for i, m in enumerate(messages):
        if m.role != "assistant":
            continue
        prompt = fmt.render(chat[:i], tools)
        full = fmt.render(chat[:i + 1], tools, add_generation_prompt=False)
        ok = full.startswith(prompt) and full.endswith(end)
        gen = full[len(prompt):-len(end)] if ok else ""
        out.append({"message_index": i, "prompt": prompt, "generation": gen, "rendered_ok": ok,
                    "prefix_consistent": None if prev is None else prompt.startswith(prev)})
        prev = prompt + gen + end
    return out


# ---------------------------------------------------------------------------
# sample -> trajectory

def _usage_by_model_event(sample) -> list:
    return [e for e in (getattr(sample, "events", None) or []) if getattr(e, "event", None) == "model"]


def _kind_and_score(sample):
    scores = getattr(sample, "scores", None) or {}
    score = scores.get(SCORER)
    if score is None:
        for s in scores.values():
            if (getattr(s, "metadata", None) or {}).get("evaluation_kind"):
                score = s
                break
    if score is None:
        return None, None
    return (score.metadata or {}).get("evaluation_kind"), score


def recorded_reasoning_effort(sample) -> str | None:
    """reasoning_effort recorded in the first ModelEvent's generate config (or chat_template_kwargs)."""
    for e in _usage_by_model_event(sample):
        cfg = getattr(e, "config", None)
        if cfg is None:
            continue
        if getattr(cfg, "reasoning_effort", None):
            return str(cfg.reasoning_effort)
        kw = ((getattr(cfg, "extra_body", None) or {}).get("chat_template_kwargs") or {})
        if kw.get("reasoning_effort"):
            return str(kw["reasoning_effort"])
        return None
    return None


def _stop_reason(ev) -> str:
    choices = getattr(getattr(ev, "output", None), "choices", None) or []
    return str(choices[0].stop_reason) if choices else ""


def _is_ids(x) -> bool:
    return isinstance(x, list) and bool(x) and all(isinstance(t, int) and not isinstance(t, bool) for t in x)


def recorded_token_ids(ev) -> tuple[list[int], list[int]] | None:
    """(prompt_token_ids, generated token_ids) from a ModelEvent's raw vLLM response, if both were logged.

    vLLM returns them with `return_token_ids=true` (top-level `prompt_token_ids`, per-choice `token_ids`).
    """
    resp = getattr(getattr(ev, "call", None), "response", None)
    if not isinstance(resp, dict):
        return None
    choices = resp.get("choices") or []
    prompt_ids = resp.get("prompt_token_ids")
    gen_ids = choices[0].get("token_ids") if choices and isinstance(choices[0], dict) else None
    if _is_ids(prompt_ids) and _is_ids(gen_ids):
        return [int(t) for t in prompt_ids], [int(t) for t in gen_ids]
    return None


def from_eval_sample(sample, *, instance_id: str, variant: str, partition: str | None, seed: int,
                     fmt: llm.ChatFormat, tools: list[dict] | None = None, replay_reasoning: bool = True,
                     rollout_index: int | None = None, meta: dict | None = None) -> tj.Trajectory:
    """Convert one `inspect_ai.log.EvalSample` (read with `read_eval_log`; nothing is executed from it)."""
    messages = list(sample.messages)
    mevents = _usage_by_model_event(sample)
    if tools is None:
        if not mevents or not getattr(mevents[0], "tools", None):
            raise ValueError("no ModelEvent tools in the sample; pass tools=")
        tools = tool_specs(mevents[0].tools)
    k = int(sample.epoch) - 1 if rollout_index is None else rollout_index
    kind, score = _kind_and_score(sample)
    recorded_effort = recorded_reasoning_effort(sample)
    calls = reconstruct_calls(messages, tools, fmt, replay_reasoning=replay_reasoning)
    n_assistant = sum(m.role == "assistant" for m in messages)
    usage_ok = len(mevents) == n_assistant

    traj = tj.Trajectory(meta={
        "task_id": instance_id, "variant": variant, "rollout_id": f"{instance_id}__{variant}__r{k}",
        "seed": seed, "partition": partition, "k": k,
        "template_sha256": fmt.template_digest(),
        "tool_specs_sha256": hashlib.sha256(repr(tools).encode()).hexdigest(),
        "target_model": fmt.name,
        "source": "inspect_eval", "sample_uuid": getattr(sample, "uuid", None),
        "reasoning_effort_recorded": recorded_effort,
        "reasoning_effort_rendered": fmt.reasoning_effort or f"{fmt.default_effort} (template default)",
        "reconstruction": {"prompt": "reconstructed", "generation": "reserialised",
                           "tokens": "absent", "replay_reasoning": replay_reasoning,
                           "exact": False, "parity_check_required": True},
        **(meta or {})})
    t0 = mevents[0].timestamp if mevents and getattr(mevents[0], "timestamp", None) else None
    add = traj.add

    call_index = -1
    call_no = 0                      # synthetic call ids, as in the LCB runner
    ids: dict[str, tuple[str, str]] = {}    # inspect tool_call_id -> (call_id, name)
    submit_args: dict | None = None
    n_user = 0
    for i, m in enumerate(messages):
        if m.role == "system":
            add(tj.SYSTEM, {"text": _text(m.content)})
        elif m.role == "user":
            origin = "task" if n_user == 0 else ("incorrect_feedback" if submit_args is not None else "continue")
            n_user += 1
            add(tj.USER, {"text": _text(m.content), "origin": origin})
        elif m.role == "assistant":
            call_index += 1
            c = calls[call_index]
            gen, closed = c["generation"], True
            ev = mevents[call_index] if usage_ok else None
            usage = getattr(getattr(ev, "output", None), "usage", None)
            rec_prompt = None if usage is None else int(
                (usage.input_tokens or 0) + (getattr(usage, "input_tokens_cache_read", None) or 0)
                + (getattr(usage, "input_tokens_cache_write", None) or 0))
            tstamp = 0.0 if ev is None or t0 is None or not getattr(ev, "timestamp", None) \
                else round((ev.timestamp - t0).total_seconds(), 3)
            ids_rec = recorded_token_ids(ev) if ev is not None else None
            if ids_rec is not None:
                rec_prompt = len(ids_rec[0])
            exact_data = {} if ids_rec is None else {
                "prompt_token_ids": ids_rec[0], "tokens": ids_rec[1], "tokens_source": "vllm"}
            prefix_ok = c["prefix_consistent"]
            add(tj.MODEL_CALL, {
                "prompt": c["prompt"], "prompt_sha256": hashlib.sha256(c["prompt"].encode()).hexdigest(),
                "prefix_consistent": prefix_ok, "generation": gen,
                "stop_type": "limit" if _stop_reason(ev) in ("max_tokens", "model_length") else "eos",
                "prompt_tokens": rec_prompt, "prompt_tokens_recorded": rec_prompt,
                "completion_tokens_recorded": None if usage is None else usage.output_tokens,
                "reconstruction": "approximate" if ids_rec is None else "exact",
                "rendered_ok": c["rendered_ok"], "tokens_source": None, "seed": seed, **exact_data}, call_index=call_index, t=tstamp)
            reasoning = _reasoning(m.content).strip()
            text = _text(m.content).strip()
            if reasoning:
                add(tj.REASONING, {"text": reasoning, "closed": closed}, call_index)
            add(tj.ASSISTANT, {"text": text}, call_index)
            raws = _CALL_RAW.findall(gen)
            good_i = 0
            for j, tc in enumerate(m.tool_calls or []):
                call_no += 1
                cid = f"call_{call_no}"
                ids[str(tc.id)] = (cid, tc.function)
                raw = ""
                if not tc.parse_error:
                    raw = raws[good_i] if good_i < len(raws) else ""
                    good_i += 1
                args = dict(tc.arguments or {})
                add(tj.TOOL_PROPOSED, {"call_id": cid, "name": tc.function, "arguments": args, "raw": raw,
                                       "parse_error": tc.parse_error, "inspect_id": str(tc.id)}, call_index)
                add(tj.TOOL_EXECUTED, {"call_id": cid, "name": tc.function, "arguments": args,
                                       "modified_by_wrapper": False})
                if tc.function == SUBMIT_TOOL:
                    submit_args = args
        elif m.role == "tool":
            cid, name = ids.get(str(m.tool_call_id), (f"call_unmatched_{i}", m.function or ""))
            add(tj.TOOL_RESULT, {"call_id": cid, "name": name or m.function, "output": _text(m.content),
                                 "error_type": m.error.type if m.error else None, "truncated": False,
                                 "seconds": 0.0})

    sources = {e.data["tokens_source"] for e in traj.of(tj.MODEL_CALL)}
    if sources == {"vllm"}:
        traj.meta["reconstruction"].update(tokens="vllm", exact=True, parity_check_required=False)
    elif "vllm" in sources:
        traj.meta["reconstruction"].update(tokens="mixed")

    # terminal submission + scorer outcome
    last_call = call_index if call_index >= 0 else None
    patch = ((score.metadata or {}).get("model_patch") if score is not None else None) or ""
    origin = "submit" if submit_args is not None else "terminal_state"
    answer = (submit_args or {}).get("answer", "") if submit_args is not None else ""
    add(tj.SUBMISSION, {"attempt": 1, "origin": origin, "func_py": patch, "test_py": "", "answer": answer,
                        "artifact": "model_patch (git diff of the final workspace)"}, last_call)
    add(tj.SUBMISSION_EXECUTED, {"attempt": 1, "func_py": patch, "test_py": "", "modified_by_wrapper": False})
    if getattr(sample, "error", None) is not None or score is None or kind not in ("pass", "test_failure", "flagged"):
        passed = None
    else:
        passed = kind == "pass"
    add(tj.EVALUATION, {"attempt": 1, "passed": passed,
                        "value": None if score is None else float(score.value),
                        "explanation": None if score is None else score.explanation,
                        "evaluation_kind": kind, "flagged": kind == "flagged",
                        "test_command_exit": None if score is None else (score.metadata or {}).get("test_command_exit")})
    limit = getattr(sample, "limit", None)
    if getattr(sample, "error", None) is not None:
        status, reason = "infra_error", str(sample.error.message)[:500]
    elif limit is not None:
        status, reason = f"{limit.type}_limit", getattr(limit, "reason", "") or ""
    else:
        status, reason = "completed", ""
    add(tj.ROLLOUT_END, {"status": status, "reason": reason, "label": tj.label(traj),
                         "wall_seconds": getattr(sample, "total_time", None)})
    return traj


# ---------------------------------------------------------------------------
# optional helpers needing a tokenizer

def attach_retokenized_tokens(traj: tj.Trajectory, tokenizer) -> None:
    """Fill MODEL_CALL `tokens` with tokenize(generation + "<|im_end|>") so activations.reconstruct can run.

    These are NOT the sampled ids (Inspect does not store them); mark `tokens_source`. The events are frozen
    dataclasses but their `data` dicts are mutable.
    """
    for e in traj.of(tj.MODEL_CALL):
        if e.data.get("tokens_source") == "vllm":
            continue
        e.data["tokens"] = [int(t) for t in tokenizer.encode(e.data["generation"] + _end_marker(traj.meta.get("target_model")),
                                                              add_special_tokens=False)]
        e.data["tokens_source"] = "retokenized"


def verify_recorded_ids(traj: tj.Trajectory, tokenizer) -> list[dict]:
    """For calls with vLLM ids: does tokenize(reconstructed prompt) == recorded prompt ids (`prompt_ids_match`),
    and do the generated ids decode to the reconstructed generation (`generation_decode_match`)?

    Sets both flags in the MODEL_CALL data. A False prompt match means the reconstructed text is not what the
    server tokenized; consumers should then use the recorded ids (`reconstruct_exact`), not the text.
    """
    rows = []
    end_id = tokenizer.encode(_end_marker(traj.meta.get("target_model")), add_special_tokens=False)[0]
    for e in traj.of(tj.MODEL_CALL):
        d = e.data
        if "prompt_token_ids" not in d:
            continue
        d["prompt_ids_match"] = tokenizer.encode(d["prompt"], add_special_tokens=False) == d["prompt_token_ids"]
        g = d["tokens"]
        g = g[:-1] if g and g[-1] == end_id else g
        d["generation_decode_match"] = tokenizer.decode(g, skip_special_tokens=False) == d["generation"]
        rows.append({"call_index": e.call_index, "prompt_ids_match": d["prompt_ids_match"],
                     "generation_decode_match": d["generation_decode_match"]})
    return rows


def adopt_recorded_text(traj: tj.Trajectory, tokenizer) -> int:
    """For calls with recorded ids that `verify_recorded_ids` did not confirm, make `prompt` / `generation` the text
    the recorded ids decode to (what the server really fed and sampled) and keep the reconstruction in
    `prompt_reconstructed` / `generation_reconstructed`; `text_source` says which one a call carries. Returns the
    number of calls replaced. Run `verify_recorded_ids` first."""
    end_id = tokenizer.encode(_end_marker(traj.meta.get("target_model")), add_special_tokens=False)[0]
    n = 0
    for e in traj.of(tj.MODEL_CALL):
        d = e.data
        if "prompt_token_ids" not in d or "prompt_ids_match" not in d:
            continue
        if d["prompt_ids_match"] and d.get("generation_decode_match"):
            d["text_source"] = "reconstructed_verified"
            continue
        g = d["tokens"]
        g = g[:-1] if g and g[-1] == end_id else g
        d["prompt_reconstructed"], d["generation_reconstructed"] = d["prompt"], d["generation"]
        d["prompt"] = tokenizer.decode(d["prompt_token_ids"], skip_special_tokens=False)
        d["generation"] = tokenizer.decode(g, skip_special_tokens=False)
        d["prompt_sha256"] = hashlib.sha256(d["prompt"].encode()).hexdigest()
        d["text_source"] = "decoded_ids"
        n += 1
    return n


def reconstruct_exact(traj: tj.Trajectory, tokenizer, end_ids=None, marks: dict | None = None):
    """`activations.reconstruct` that uses recorded `prompt_token_ids` where present (else tokenizes `prompt`).

    activations.reconstruct always tokenizes `prompt` and has no hook for recorded prompt ids; scripts that call
    it directly (extract_vllm, parity_check) need `p_ids = d.get("prompt_token_ids") or tokenizer.encode(...)`
    there to benefit; this function is the same logic without touching that file.
    """
    from evals.reward_hacking import activations as act
    marks = marks if marks is not None else act._marker_ids(tokenizer, end_ids)
    calls = []
    for e in traj.of(tj.MODEL_CALL):
        d = e.data
        p_ids = list(d["prompt_token_ids"]) if "prompt_token_ids" in d \
            else tokenizer.encode(d["prompt"], add_special_tokens=False)
        g_ids = [int(t) for t in d["tokens"]]
        calls.append(act.CallSeq(call_index=e.call_index, prompt_ids=p_ids, gen_ids=g_ids,
                                 channel=act.label_channels(p_ids, g_ids, marks),
                                 stop_type=d.get("stop_type", "")))
    return calls


def check_prompt_lengths(traj: tj.Trajectory, tokenizer) -> list[dict]:
    """Reconstructed prompt token length vs the server-reported prompt length per call.

    A large systematic shortfall means history reasoning was not replayed the way `replay_reasoning` assumes
    (or the tokenizer differs); the recorded count is input + cache-read tokens of the ModelEvent.
    """
    rows = []
    for e in traj.of(tj.MODEL_CALL):
        n = len(tokenizer.encode(e.data["prompt"], add_special_tokens=False))
        rec = e.data.get("prompt_tokens_recorded")
        rows.append({"call_index": e.call_index, "reconstructed": n, "recorded": rec,
                     "ratio": None if not rec else round(n / rec, 4)})
    return rows


# ---------------------------------------------------------------------------
# CLI

def _partition_of(instance_id: str) -> str | None:
    try:
        from evals.reward_hacking import swebench_tasks
        for name in ("train", "val", "test"):
            if instance_id in swebench_tasks.partition_ids(name):
                return name
    except Exception:
        pass
    return None


def detect_target(log) -> str:
    """Target model of an Inspect log (Qwen)."""
    return QWEN_TARGET


def _log_effort(gen, target: str) -> str | None:
    return getattr(gen, "reasoning_effort", None)


def convert_log(path, *, fmt=None, replay_reasoning: bool = True, tokenizer=None, target_model: str | None = None):
    from inspect_ai.log import read_eval_log
    log = read_eval_log(str(path))
    split = (log.eval.task_args or {}).get("split", "conflicting")
    gen = log.eval.model_generate_config
    target = fmt.name if fmt is not None else (target_model or detect_target(log))
    fmt = fmt or llm.get_format(target, reasoning_effort=_log_effort(gen, target))
    # swebench_run writes each (task, k) to inspect/<id>__r<k>/; every eval there has epoch 1, so k comes from the dir.
    m = re.search(r"__r(\d+)$", pathlib.Path(str(path)).parent.name)
    rollout_index = int(m.group(1)) if m else None
    for s in log.samples or []:
        iid = str(s.id)
        if not _usage_by_model_event(s):
            # No model call at all (e.g. the sandbox failed before the first turn: an infra episode). Nothing to
            # monitor or extract; reported, never silently dropped.
            print(f"skip {iid} ({pathlib.Path(str(path)).parent.name}): no model events", file=sys.stderr, flush=True)
            continue
        seed = int(getattr(gen, "seed", None) or 0)
        t = from_eval_sample(s, instance_id=iid, variant=split, partition=_partition_of(iid), seed=seed,
                             fmt=fmt, replay_reasoning=replay_reasoning, rollout_index=rollout_index,
                             meta={"eval_log": pathlib.Path(str(path)).name, "model": log.eval.model,
                                   "task": log.eval.task})
        if tokenizer is not None:
            verify_recorded_ids(t, tokenizer)
            if any("tokens" not in e.data for e in t.of(tj.MODEL_CALL)):
                attach_retokenized_tokens(t, tokenizer)
        yield t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect SWE-bench .eval logs -> canonical trajectories")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer dir: also fill re-tokenised generated ids")
    ap.add_argument("--target-model", default=None, choices=[QWEN_TARGET],
                    help="agent model of the logs (default: read from the log's model name)")
    ap.add_argument("--no-replay-reasoning", action="store_true",
                    help="render history turns without their reasoning (if the server did not replay it)")
    a = ap.parse_args(argv)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = pathlib.Path(a.out) / "trajectories"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(pathlib.Path(a.eval_dir).rglob("*.eval")):
        for t in convert_log(p, replay_reasoning=not a.no_replay_reasoning, tokenizer=tok,
                               target_model=a.target_model):
            dest = out / f"{t.meta['rollout_id']}.json"
            dest.write_text(t.to_json())
            kinds = {}
            for e in t.events:
                kinds[e.kind] = kinds.get(e.kind, 0) + 1
            print(f"{dest} calls={kinds.get(tj.MODEL_CALL, 0)} events={len(t.events)}")
            n += 1
    if not n:
        print("no samples found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
