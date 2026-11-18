"""80/20 task-grouped train/test splits for the new EQ2 datasets; no separate validation split.

Hyperparameters and thresholds are chosen by 5-fold task-grouped cross-validation inside train. Test = the 20% of
usable tasks with the smallest salted SHA-256, excluding tasks whose individual outcomes were inspected before this
split (SWE: data/swebench/inspected_before_split_v2.json), which are forced into train.

  python -m evals.reward_hacking.scripts.make_splits_v2
"""
from __future__ import annotations

import glob
import hashlib
import json
import pathlib

from evals.reward_hacking import rollout

D = pathlib.Path(__file__).resolve().parents[1] / "data"
SALT = "certified-taps/eq2/splits-v2"
TEST_FRACTION = 0.20


def _h(task: str) -> str:
    return hashlib.sha256(f"{SALT}:{task}".encode()).hexdigest()


def split(usable: list[str], forced_train: set[str]) -> dict[str, str]:
    n_test = round(TEST_FRACTION * len(usable))
    eligible = sorted((t for t in usable if t not in forced_train), key=_h)
    test = set(eligible[:n_test])
    return {t: ("test" if t in test else "train") for t in sorted(usable)}


def swe_usable() -> list[str]:
    rows = {}
    for f in sorted(glob.glob(str(D / "swebench" / "preflight_v1" / "results-*.jsonl"))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return sorted(i for i, r in rows.items() if r["status"] == "ok" and (r.get("original_oracle") or {}).get("pass"))


def main() -> None:
    seen = set(json.loads((D / "swebench" / "inspected_before_split_v2.json").read_text()))
    usable = swe_usable()
    out = {"salt": SALT, "test_fraction": TEST_FRACTION, "usable_rule": "preflight ok and original oracle passes",
           "forced_train": sorted(seen & set(usable)), "assignment": split(usable, seen)}
    (D / "swebench" / "splits_v2.json").write_text(json.dumps(out, indent=1) + "\n")
    lcb = sorted(t for t in json.loads((D / "splits.json").read_text())["assignment"] if t not in rollout.MALFORMED_TASKS)
    out2 = {"salt": SALT, "test_fraction": TEST_FRACTION, "applies_to": "LCB-oneoff",
            "note": "task ids are shared with LCB-conflicting, whose split differs; do not evaluate conflicting-trained "
                    "probes on oneoff tasks that were in conflicting train",
            "assignment": split(lcb, set())}
    (D / "lcb_oneoff_schedules" / "splits_oneoff_v2.json").write_text(json.dumps(out2, indent=1) + "\n")
    for name, o in (("swe", out), ("oneoff", out2)):
        vals = list(o["assignment"].values())
        print(name, len(vals), "train", vals.count("train"), "test", vals.count("test"))


if __name__ == "__main__":
    main()
