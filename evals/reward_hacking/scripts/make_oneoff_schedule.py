"""LCB-oneoff T1 schedules (data/a32): k=0..5 on every usable task, trainval sharded for parallel jobs.

Same 100 tasks as the conflicting T1 population (the runner drops `rollout.MALFORMED_TASKS` for every
variant; on oneoff only lcbhard_77 is actually malformed). Seeds come from (task:prompt_id, k), so each
oneoff rollout pairs with the conflicting rollout of the same task and k. Run with `--variant oneoff`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from evals.reward_hacking import rollout

D = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT = D / "a32"
PROMPT = "T1"
KS = tuple(range(6))
TRAINVAL_SHARDS = 3


def build() -> dict[str, list[dict]]:
    assign = json.loads((D / "splits.json").read_text())["assignment"]
    usable = sorted(t for t in assign if t not in rollout.MALFORMED_TASKS)
    files = {}
    for name, splits in (("trainval", ("train", "val")), ("test", ("test",))):
        rows = [{"task_id": t, "prompt_id": PROMPT, "k": k} for k in KS for t in usable if assign[t] in splits]
        if name == "test":
            files["oneoff_T1_test.jsonl"] = rows
        else:
            # Shard by task so each task's six rollouts land in one job.
            tids = sorted({r["task_id"] for r in rows})
            for i in range(TRAINVAL_SHARDS):
                mine = set(tids[i::TRAINVAL_SHARDS])
                files[f"oneoff_T1_trainval_s{i}.jsonl"] = [r for r in rows if r["task_id"] in mine]
    return files


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = build()
    man = {"variant": "oneoff", "prompt_id": PROMPT, "k": list(KS), "rows": {}, "sha256": {},
           "excluded_malformed": sorted(rollout.MALFORMED_TASKS)}
    for name, rows in files.items():
        data = "".join(json.dumps(r) + "\n" for r in rows)
        (OUT / name).write_text(data)
        man["rows"][name] = len(rows)
        man["sha256"][name] = hashlib.sha256(data.encode()).hexdigest()
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")


if __name__ == "__main__":
    main()
