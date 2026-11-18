"""Monitor inputs for the EQ1 shared-call cost table: the recorded trace up to a selected call, plus the NEWLY GENERATED call as the proposed action.

    python -m evals_runner.eq1_judge_saved_calls_prompts --stage2-list stage2_list.json --histories histories.json \\
        --outputs generate_calls_stage2/outputs.jsonl --trajectories DIR --out inputs.json.gz [--manifest inputs.manifest.json]

Unit. The 24 prompt representatives of the focused live-generation sample (`evals/tap_live`, `stage2_list.json`, 30 positions with weight 1/30). For a
representative (task, call index c) the monitor reads the FULL available trace through the new response:
  * the events of the recorded Qwen LiveCodeBench T1 conflicting trajectory (the file named in `histories.json`, hash checked, and its recorded model call c must
    have the history's prompt hash) up to, and excluding, the model call c: every earlier message, tool call and tool result;
  * plus the new call c, the saved A0 output of the main stage (`outputs.jsonl`, the same output whose generation time is the baseline), turned into events the way
    the agent loop records a turn (`rollout.py`): the generation is parsed with the agent's own format (`llm.get_format("qwen3.8-27b").parse_generation`) into private
    reasoning, visible text and tool calls; reasoning -> a REASONING event, visible text -> an ASSISTANT event, each tool call -> a TOOL_PROPOSED event (a malformed
    call keeps its parse error). Nothing after the call is shown (no tool result, no evaluation).
Views and fitting are the detector's: `monitors.build_prompt_fitted`'s path (`_fit` with the matched assembler) with `actions="proposed"` and an explicit boundary
at the new call. The primary view is the history black-box transcript (private reasoning is not in it); the history-with-reasoning view is the only one that
shows the reasoning. Instruction: the per-call decision-point instruction `EARLY_INSTRUCTION` (what `run_monitor_per_call` uses for a prefix that ends at a
decision point) by default; `--instruction terminal` selects the terminal `INSTRUCTION`; the choice and its sha256 are recorded.

UNIFORM POLICY for a response without a parsed action (fixed here, before any monitor request; applied to every output, never chosen after seeing a result):
the response enters the trace exactly as the agent loop would record it: reasoning (also an unclosed one) as a reasoning event, visible text as an agent message,
well-formed and malformed tool calls as proposed calls. If the output stopped at the context ceiling, or has no tool call, or its reasoning never closed, the item is
flagged (`no_parsed_action`, `finish_reason`, `reasoning_closed`, `visible_empty`) and still judged. A response with nothing visible adds nothing to the transcript
(black-box) view; that is flagged, not hidden.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pathlib

from evals.reward_hacking import llm, tools
from evals.reward_hacking import monitors as m
from evals.reward_hacking import trajectory as tj

TARGET = "qwen3.8-27b"
VIEWS = {"history_black_box": tj.BLACK_BOX, "history_reasoning": tj.COT}
INSTRUCTIONS = {"early": m.EARLY_INSTRUCTION, "terminal": m.INSTRUCTION}
POLICY = __doc__.split("UNIFORM POLICY")[1].strip()
POLICY_SHA256 = hashlib.sha256(POLICY.encode()).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]


def find_trajectory_file(root: pathlib.Path, name: str) -> pathlib.Path | None:
    for p in (root / "trajectories" / name, root / name):
        if p.is_file():
            return p
    return None


def model_call_cut(traj: tj.Trajectory, call_index: int) -> tuple[int, tj.Event]:
    """Index (in traj.events) and event of the recorded model call `call_index`; the ordinal of a model_call event stands in for a missing call_index."""
    n = 0
    for pos, e in enumerate(traj.events):
        if e.kind == tj.MODEL_CALL:
            ci = e.call_index if e.call_index is not None else n
            if ci == call_index:
                return pos, e
            n += 1
    raise ValueError(f"no model call {call_index} in the trajectory")


def response_text(out: dict) -> tuple[str, bool]:
    """The generation as the agent loop receives it (the text without the stop token) and whether it ended at a natural stop."""
    return out["text_stop_token_removed"], out["finish_reason"] == "natural"


def response_events(fmt, out: dict, call_index: int, start_seq: int, first_call_number: int) -> tuple[list[tj.Event], dict]:
    text, natural = response_text(out)
    turn = fmt.parse_generation(text, tools.TOOL_SPECS)
    evs, seq = [], start_seq

    def add(kind, data):
        nonlocal seq
        evs.append(tj.Event(seq=seq, kind=kind, call_index=call_index, data=data))
        seq += 1
    if turn.reasoning or not turn.reasoning_closed:
        add(tj.REASONING, {"text": turn.reasoning, "closed": turn.reasoning_closed})
    add(tj.ASSISTANT, {"text": turn.content})
    for k, call in enumerate(turn.tool_calls):
        add(tj.TOOL_PROPOSED, {"call_id": f"call_{first_call_number + k}", "name": call.name, "arguments": call.arguments, "raw": call.raw,
                               "parse_error": call.parse_error})
    n_bad = sum(c.parse_error is not None for c in turn.tool_calls)
    flags = {"response_finish_reason": out["finish_reason"], "response_natural_stop": natural, "response_tokens": out.get("n_tokens"),
             "reasoning_closed": bool(turn.reasoning_closed), "n_tool_calls": len(turn.tool_calls), "n_malformed_tool_calls": n_bad,
             "no_parsed_action": (not natural) or not turn.reasoning_closed or (len(turn.tool_calls) - n_bad) == 0,
             "visible_empty": not turn.content.strip() and not turn.tool_calls, "reasoning_chars": len(turn.reasoning),
             "visible_chars": len(turn.content)}
    return evs, flags


def build_trace(traj: tj.Trajectory, history: dict, out: dict, fmt) -> tuple[tj.Trajectory, int, dict]:
    """The new trajectory (prefix up to the call + the new response) and the seq of its last event (the view boundary)."""
    cut, call_ev = model_call_cut(traj, history["call_index"])
    if call_ev.data.get("prompt_sha256") != history["prompt_sha256"] and \
            hashlib.sha256(call_ev.data["prompt"].encode()).hexdigest() != history["prompt_sha256"]:
        raise ValueError(f"{history['history_id']}: the recorded model call does not have the history's prompt hash")
    prefix = list(traj.events[:cut])
    nums = [int(str(e.data.get("call_id", "call_0")).rsplit("_", 1)[-1]) for e in prefix if e.kind == tj.TOOL_PROPOSED and str(e.data.get("call_id", "")).startswith("call_")]
    evs, flags = response_events(fmt, out, history["call_index"], (prefix[-1].seq + 1) if prefix else 0, (max(nums) + 1) if nums else 1)
    # the prefix events keep their own seq values (the view boundary compares seq)
    new = tj.Trajectory(meta={**traj.meta, "tap_live_history_id": history["history_id"]}, events=prefix + evs)
    boundary = evs[-1].seq if evs else (prefix[-1].seq if prefix else 0)
    flags["prefix_events"] = len(prefix)
    return new, boundary, flags


def fit_view(traj: tj.Trajectory, boundary: int, view: str, instruction: str, counter, budget: int, overhead: int) -> dict:
    """The detector's fitting path (`build_prompt_fitted`): only private reasoning is shortened. Returns the user message, its token count (incl. overhead),
    the characters cut and the shortening report, or the reason it cannot be fitted."""
    from evals_runner.eq1_monitor_panel_retry import shortening_report
    events0 = tj.view_events(traj, view, boundary_seq=boundary, actions="proposed")
    try:
        user, events, cut, n = m._fit(events0, view, m._matched_assembler(view, instruction), counter, budget, overhead)
    except ValueError as exc:
        return {"ok": False, "reason": f"cannot be fitted: {str(exc)[:160]}"}
    return {"ok": True, "user": user, "user_tokens": n, "reasoning_chars_cut": cut, "report": shortening_report(events0, events)}


def unfitted_user(traj: tj.Trajectory, boundary: int, view: str, instruction: str) -> str:
    events = tj.view_events(traj, view, boundary_seq=boundary, actions="proposed")
    return m._matched_assembler(view, instruction)(events)


def check_inputs(stage2_list: pathlib.Path, histories: pathlib.Path, outputs: pathlib.Path):
    """The stage-2 list, the histories it was made from and the saved A0 outputs: file hash, per-run prompt hashes and
    output identities. Returns (selection, histories bytes, [(run, history, output)]); raises SystemExit on a mismatch."""
    sel = json.loads(pathlib.Path(stage2_list).read_text())
    hist_bytes = pathlib.Path(histories).read_bytes()
    if sel.get("histories_sha256") and sha256_bytes(hist_bytes) != sel["histories_sha256"]:
        raise SystemExit("the histories file differs from the one the stage-2 list was made from")
    hist = {h["history_id"]: h for h in json.loads(hist_bytes)["histories"]}
    outs = {}
    for r in read_jsonl(outputs):
        outs.setdefault(r["run_key"], r)       # the first record of a run key (outputs are written once per key)
    matched = []
    for run in sel["runs"]:
        h = hist[run["history_id"]]
        if h["prompt_sha256"] != run["prompt_sha256"]:
            raise SystemExit(f"prompt hash of {run['history_id']} differs from the stage-2 list")
        out = outs.get(run["run_key"])
        if out is None:
            raise SystemExit(f"no saved A0 output for run {run['history_id']} ({run['run_key'][:12]}); the main stage has not finished it")
        if out.get("prompt_sha256") != h["prompt_sha256"] or out.get("history_id") != h["history_id"]:
            raise SystemExit(f"the saved output of {run['history_id']} was generated for another prompt")
        matched.append((run, h, out))
    return sel, hist_bytes, matched


def load_world(stage2_list: pathlib.Path, histories: pathlib.Path, outputs: pathlib.Path, trajectories: pathlib.Path) -> dict:
    """Checks every identity and returns the 24 representatives with their trajectory, output and history."""
    sel, hist_bytes, matched = check_inputs(stage2_list, histories, outputs)
    reps = []
    for run, h, out in matched:
        src = find_trajectory_file(trajectories, h["source_file"])
        if src is None:
            raise SystemExit(f"trajectory file {h['source_file']} not found under {trajectories}")
        raw = src.read_bytes()
        if sha256_bytes(raw) != h["source_sha256"]:
            raise SystemExit(f"trajectory {h['source_file']} differs from the one the histories were selected from (sha256)")
        reps.append({"run": run, "history": h, "output": out, "trajectory": tj.Trajectory.from_json(raw.decode()), "trajectory_sha256": h["source_sha256"]})
    return {"sel": sel, "reps": reps, "stage2_list_sha256": sha256_bytes(stage2_list.read_bytes()), "histories_sha256": sha256_bytes(hist_bytes),
            "outputs_sha256": sha256_bytes(outputs.read_bytes())}


def build_inputs(world: dict, views: list[str], instruction_name: str, fmt=None) -> list[dict]:
    """Unfitted items in the declared order: all representatives for one view, then the next view; representative order = the list's order."""
    fmt = fmt or llm.get_format(TARGET)
    instr = INSTRUCTIONS[instruction_name]
    items = []
    built = []
    for rep in world["reps"]:
        traj, boundary, flags = build_trace(rep["trajectory"], rep["history"], rep["output"], fmt)
        built.append((rep, traj, boundary, flags))
    for view_name in views:
        for rep, traj, boundary, flags in built:
            user = unfitted_user(traj, boundary, VIEWS[view_name], instr)
            items.append({"order": len(items), "run_key": rep["run"]["run_key"], "representative_index": rep["run"]["representative_index"],
                          "history_id": rep["history"]["history_id"], "task": rep["history"]["task"], "view": view_name, "input": "history",
                          "user": user, "user_sha256": sha256_bytes(user.encode()), "boundary_seq": boundary, "flags": flags,
                          "_trace": traj})
    return items


def write_gz(path: pathlib.Path, blob: bytes) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(blob)
    path.write_bytes(buf.getvalue())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2-list", type=pathlib.Path, required=True)
    ap.add_argument("--histories", type=pathlib.Path, required=True)
    ap.add_argument("--outputs", type=pathlib.Path, required=True)
    ap.add_argument("--trajectories", type=pathlib.Path, required=True)
    ap.add_argument("--views", default="history_black_box")
    ap.add_argument("--instruction", choices=tuple(INSTRUCTIONS), default="early")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    views = [v for v in a.views.split(",") if v]
    if any(v not in VIEWS for v in views):
        raise SystemExit(f"--views from {sorted(VIEWS)}")
    world = load_world(a.stage2_list, a.histories, a.outputs, a.trajectories)
    items = build_inputs(world, views, a.instruction)
    pub = [{k: v for k, v in it.items() if k != "_trace"} for it in items]
    blob = json.dumps({"kind": "eq1_judge_saved_calls_inputs", "items": pub}, separators=(",", ":")).encode()
    write_gz(a.out, blob)
    man = a.manifest or a.out.with_name(a.out.name.removesuffix(".json.gz") + ".manifest.json")
    man.write_text(json.dumps({
        "kind": "eq1_judge_saved_calls_manifest", "unit": "one saved A0 output per distinct prompt (24 representatives); each position takes its representative's judged time",
        "policy": POLICY, "policy_sha256": POLICY_SHA256, "views": views, "instruction": a.instruction,
        "instruction_sha256": sha256_bytes(INSTRUCTIONS[a.instruction].encode()), "stage2_list_sha256": world["stage2_list_sha256"],
        "histories_sha256": world["histories_sha256"], "outputs_sha256": world["outputs_sha256"], "format": TARGET,
        "trajectory_files": {r["history"]["history_id"]: {"file": r["history"]["source_file"], "sha256": r["trajectory_sha256"]} for r in world["reps"]},
        "items": [{k: it[k] for k in ("order", "run_key", "history_id", "task", "view", "user_sha256", "boundary_seq")} | {"flags": it["flags"], "user_chars": len(it["user"])}
                  for it in pub],
        "inputs_file": a.out.name, "inputs_file_sha256": sha256_bytes(a.out.read_bytes())}, indent=1) + "\n")
    print(json.dumps({"items": len(items), "views": views, "no_parsed_action": sum(it["flags"]["no_parsed_action"] for it in items), "manifest": str(man)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
