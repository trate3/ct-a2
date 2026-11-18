"""Fixed selection of recorded Qwen LiveCodeBench T1 call histories for the focused live-decoding measurement.

  python -m evals.tap_live.select_histories --trajectories DIR --tokenizer DIR --out histories.json

Rule (fixed before any measurement; independent of timings, statuses and labels, none of which is read):
  tasks   the 10 positions round(i * 38 / 9), i = 0..9, of the sorted intersection of the TRAIN assignments of
          evals/reward_hacking/data/splits.json and data/a32/splits_oneoff_v2.json, without lcbhard_35, 77 and 87
          (`select_tasks.select`);
  episode for each task the recorded trajectory with the smallest k (the `__r0` file, meta.k = 0 when present; the lowest
          `__r<k>` otherwise), taken from DIR/trajectories/ or DIR itself;
  depths  the model calls at 25 %, 50 % and 75 % of the episode: call index floor(f * (n_calls - 1)), n_calls = the number of
          model_call events;
  cap     48,000 prompt tokens (the model tokenizer on the recorded prompt text, special tokens not added): if the call at a
          depth is longer, the nearest EARLIER call that fits is used and the substitution is recorded; if no call fits the
          history is recorded as 'unavailable' (and counted as missing).
Only `model_call` events are used: data.prompt (the exact rendered text of the call) and, as a cross-check, data.prompt_tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from evals.tap_live import select_tasks

DEPTHS = (0.25, 0.5, 0.75)
PROMPT_CAP = 48000
_RK = re.compile(r"__r(\d+)\.json$")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def episode_file(traj_dir: Path, task: str) -> Path | None:
    """The file of the task's smallest k (from the file name; `*.snapshots.pkl`, `infra_attempts/` are not matched)."""
    d = traj_dir / "trajectories" if (traj_dir / "trajectories").is_dir() else traj_dir
    cands = []
    for p in d.glob(f"{task}__*.json"):
        m = _RK.search(p.name)
        if m and p.name.split("__")[0] == task:
            cands.append((int(m.group(1)), p.name, p))
    return min(cands)[2] if cands else None


def model_call_prompts(path: Path) -> list[tuple[int, str, int | None]]:
    """[(call_index, prompt text, recorded prompt_tokens)] from the model_call events only, in order."""
    events = json.loads(path.read_text())["events"]
    out = []
    for e in events:
        if e["kind"] == "model_call":
            out.append((e["call_index"] if e.get("call_index") is not None else len(out), e["data"]["prompt"],
                        e["data"].get("prompt_tokens")))
    return out


def select(traj_dir: Path, tokenizer, *, cap: int = PROMPT_CAP, task_ids: list[str] | None = None, k: int = 10,
           depths=DEPTHS) -> dict:
    sel = select_tasks.select(k)
    ids = task_ids or sel["task_ids"]
    hist = []
    for task in ids:
        path = episode_file(Path(traj_dir), task)
        calls = model_call_prompts(path) if path else []
        src = {"source_file": path.name if path else None, "source_sha256": sha256_bytes(path.read_bytes()) if path else None}
        tok_counts: dict[int, int] = {}

        def n_tok(i):
            if i not in tok_counts:
                tok_counts[i] = len(tokenizer.encode(calls[i][1], add_special_tokens=False))
            return tok_counts[i]
        for f in depths:
            rec = {"task": task, "depth_fraction": f, "n_calls": len(calls), **src}
            if not calls:
                rec.update(status="unavailable", reason="no trajectory file" if path is None else "no model calls")
                hist.append(rec)
                continue
            nominal = int(f * (len(calls) - 1))
            rec["nominal_call_index"] = nominal
            use = next((i for i in range(nominal, -1, -1) if n_tok(i) <= cap), None)
            if use is None:
                rec.update(status="unavailable", reason=f"no call up to index {nominal} fits {cap} tokens")
            else:
                rec.update(status="ok", call_index=use, fallback_to_earlier_call=use != nominal, prompt=calls[use][1],
                           prompt_tokens=n_tok(use), recorded_prompt_tokens=calls[use][2],
                           prompt_sha256=hashlib.sha256(calls[use][1].encode()).hexdigest())
            hist.append(rec)
    for h in hist:
        h["history_id"] = f"{h['task']}@{h['depth_fraction']}"
    return {"rule": __doc__.split("Rule (")[1].split("Only `model_call`")[0].strip(), "prompt_cap_tokens": cap, "depths": list(depths),
            "tasks": ids, "splits_json": sel["splits_json"], "splits_oneoff_v2_json": sel["splits_oneoff_v2_json"],
            "task_selection_sha256": sel["selection_sha256"], "histories": hist,
            "n_ok": sum(h["status"] == "ok" for h in hist), "n_unavailable": sum(h["status"] != "ok" for h in hist)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=PROMPT_CAP)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    res = select(a.trajectories, tok, cap=a.cap)
    res["tokenizer"] = a.tokenizer
    for h in res["histories"]:
        print(f"{h['history_id']:24s} {h['status']:12s} call {h.get('call_index', '-')}/{h['n_calls']} "
              f"{h.get('prompt_tokens', '-')} tokens{' (earlier call)' if h.get('fallback_to_earlier_call') else ''} "
              f"{h.get('reason', '')}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {a.out}: {res['n_ok']} histories, {res['n_unavailable']} unavailable")


if __name__ == "__main__":
    main()
