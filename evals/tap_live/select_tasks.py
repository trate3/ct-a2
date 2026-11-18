"""Declared task set for the live tap-overhead run on LiveCodeBench, chosen by a rule fixed before any measurement.

Rule: the sorted intersection of the TRAIN assignments of `evals/reward_hacking/data/splits.json` (conflicting split)
and `evals/reward_hacking/data/lcb_oneoff_schedules/splits_oneoff_v2.json` (oneoff split), without the three malformed tasks
(lcbhard_35, lcbhard_77, lcbhard_87), has 39 tasks; take the positions round(i * 38 / 9), i = 0..9. Nothing measured
(time, tokens, outcomes) enters the choice. The output records the ids and the sha256 of both split files.

  python -m evals.tap_live.select_tasks --out evals_results/tap_live/tasks.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evals.reward_hacking import rollout, tasks

SPLITS_A = tasks.SPLITS_PATH
SPLITS_B = Path(tasks.__file__).parent / "data" / "lcb_oneoff_schedules" / "splits_oneoff_v2.json"
EXCLUDED = frozenset(rollout.MALFORMED_TASKS)


def positions(n: int, k: int) -> list[int]:
    if not 1 <= k <= n:
        raise ValueError(f"cannot take {k} positions from {n} tasks")
    return [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def select(k: int = 10, splits_a: Path = SPLITS_A, splits_b: Path = SPLITS_B, excluded=EXCLUDED) -> dict:
    a = json.loads(Path(splits_a).read_text())["assignment"]
    b = json.loads(Path(splits_b).read_text())["assignment"]
    pool = sorted(t for t, part in a.items() if part == "train" and b.get(t) == "train" and t not in excluded)
    chosen = [pool[p] for p in positions(len(pool), k)]
    return {"rule": __doc__.split("Rule:")[1].split("\n\n")[0].strip(), "n_intersection": len(pool), "k": k,
            "positions": positions(len(pool), k), "task_ids": chosen, "excluded": sorted(excluded),
            "splits_json": {"path": str(splits_a), "sha256": sha256_file(splits_a)},
            "splits_oneoff_v2_json": {"path": str(splits_b), "sha256": sha256_file(splits_b)},
            "selection_sha256": hashlib.sha256(json.dumps(chosen).encode()).hexdigest()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sel = select(a.n)
    for pos, t in zip(sel["positions"], sel["task_ids"]):
        print(f"{pos:3d}  {t}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(sel, indent=1) + "\n")
    print(f"{sel['n_intersection']} tasks in the intersection; wrote {a.out}  selection_sha256={sel['selection_sha256'][:16]}")


if __name__ == "__main__":
    main()
