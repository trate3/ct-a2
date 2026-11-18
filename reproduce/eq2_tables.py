"""Regenerate the EQ2 coding tables (AUROC for reward hacking detection) on CPU.

Prints the tables: AUROC, valid scores / episodes.

Where each cell comes from:
  SWE-bench conflicting, SWE-bench one-off: every cell is recomputed from shipped
      per-episode monitor scores and cached probe scores with the repository's report
      script (`python -m evals_runner rh-eq2-cv-report`), which also refits the monitor
      thresholds on the training partition.
  LiveCodeBench, all monitors and both probes: recomputed from shipped per-episode scores with
      the repository's AUROC and task-clustered bootstrap (evals/reward_hacking/metrics.py).
  LiveCodeBench conflicting, Gemma E4B given only the final code (2 cells of the EQ3
      table): read from the aggregate report in data/eq2/reports.

Usage: python reproduce/eq2_tables.py [--out DIR] [--skip-swe]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.reward_hacking import metrics as M  # noqa: E402

DATA = ROOT / "data" / "eq2"
REP = DATA / "reports"
COLS = ["lcb_conflicting", "lcb_oneoff", "swe_conflicting", "swe_oneoff"]
LABELS = {"gemma4-e4b": "E4B, no monitor reasoning", "gemma4-e4b-think": "E4B, monitor reasons",
          "qwen3.8-27b-think": "Qwen, monitor reasons", "glm5.3-flash-nvfp4-think": "GLM, monitor reasons"}
EQ2_ROWS = ([(f"{m}:black_box", LABELS[m]) for m in LABELS] + [(f"{m}:cot", LABELS[m]) for m in LABELS]
            + [("probe:token_mean", "Probe: mean"), ("probe:ema_max", "Probe: smoothed maximum")])
EQ3_ROWS = [(f"{m}.final_state:black_box", LABELS[m]) for m in LABELS]
QG = ["gemma4-e4b", "gemma4-e4b-think", "qwen3.8-27b-think", "glm5.3-flash-nvfp4-think"]


def jl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def valid(row):
    s = row.get("score")
    return (s is not None and not isinstance(s, bool) and math.isfinite(float(s)) and not row.get("error"))


def cell(rows):
    """Valid-score AUROC with a task-clustered bootstrap (2,000 draws, seed 0)."""
    def stat(rs):
        v = [r for r in rs if r["valid"]]
        return M.auroc([r["label"] for r in v], [r["score"] for r in v]) if v else None
    b = M.cluster_bootstrap(rows, stat, n_boot=2000, seed=0)
    return {"auroc": b["point"], "ci": [b["ci_low"], b["ci_high"]],
            "valid": sum(r["valid"] for r in rows), "n": len(rows)}


def lcb_rows(path, keep):
    """Rows of one monitor file as {view: {rollout_id: row}}; error rows yield to clean ones."""
    out = {}
    for r in jl(path):
        if r.get("label") not in (0, 1) or not keep(r):
            continue
        row = {"rollout_id": r["rollout_id"], "task_id": r["task_id"], "label": int(r["label"]),
               "score": float(r["score"]) if valid(r) else float("inf"), "valid": valid(r)}
        bucket = out.setdefault(r["view"], {})
        old = bucket.get(r["rollout_id"])
        if old is None or (not old["valid"] and row["valid"]):
            bucket[r["rollout_id"]] = row
    return out


def lcb_population():
    """Print the LiveCodeBench test populations, counted on the Qwen reasoning-on transcript rows."""
    for name, rows in (("lcb_conflicting", POP["lcb_conflicting"]), ("lcb_oneoff", POP["lcb_oneoff"])):
        print(f"{name}: {len({r['task_id'] for r in rows})} tasks, {len(rows)} episodes, "
              f"{sum(r['label'] for r in rows)} positive")


POP = {}


def lcb_recompute(suffix=""):
    """Monitor cells for both LiveCodeBench populations (suffix '.final_state' for final code).
    A monitor without a per-episode file is skipped (its cell then comes from a report)."""
    res = {}
    status = json.loads((DATA / "lcb_conflicting" / "t1_val_test_labels_status.json").read_text())["test"]
    # primary T1 population: positives, plus negatives that used all attempts or were flagged
    keep_c = lambda r: (status[r["rollout_id"]]["label"] == 1
                        or status[r["rollout_id"]]["status"] in ("attempts_exhausted", "flagged"))
    split = json.loads((DATA / "lcb_oneoff" / "splits_oneoff_v2.json").read_text())["assignment"]
    for m in QG:
        fc = DATA / "lcb_conflicting/monitors/test" / f"monitor_scores.{m}{suffix}.jsonl"
        for view, rows in (lcb_rows(fc, keep_c).items() if fc.exists() else []):
            res.setdefault(f"{m}{suffix}:{view}", {})["lcb_conflicting"] = cell(list(rows.values()))
            if m == "qwen3.8-27b-think" and not suffix and view == "black_box":
                POP["lcb_conflicting"] = list(rows.values())
        merged = {}
        for sub in ("test", "trainval_s0", "trainval_s1", "trainval_s2"):
            f = DATA / "lcb_oneoff/monitors" / sub / f"monitor_scores.{m}{suffix}.jsonl"
            if not f.exists():
                continue
            for view, rows in lcb_rows(f, lambda r: split.get(r["task_id"]) == "test").items():
                for k, v in rows.items():
                    old = merged.setdefault(view, {}).get(k)
                    if old is None or (not old["valid"] and v["valid"]):
                        merged[view][k] = v
        for view, rows in merged.items():
            res.setdefault(f"{m}{suffix}:{view}", {})["lcb_oneoff"] = cell(list(rows.values()))
            if m == "qwen3.8-27b-think" and not suffix and view == "black_box":
                POP["lcb_oneoff"] = list(rows.values())
    return res


def lcb_probes():
    """Probe cells for both LiveCodeBench populations from the cached per-episode probe scores (field S)."""
    res = {}
    status = json.loads((DATA / "lcb_conflicting" / "t1_val_test_labels_status.json").read_text())["test"]
    split = json.loads((DATA / "lcb_oneoff" / "splits_oneoff_v2.json").read_text())["assignment"]
    for pool, art in (("token_mean", "tap_p2_logistic_mean.json"), ("ema_max", "tap_p2_logistic.json")):
        digest = json.loads((DATA / "lcb_conflicting/probes" / pool / art).read_text())["digest"]
        rows = []
        for r in jl(DATA / "lcb_conflicting/probes" / pool / "test_scores.jsonl"):
            st = status[r["rollout_id"]]
            if not (st["label"] == 1 or st["status"] in ("attempts_exhausted", "flagged")):
                continue
            assert r.get("artifact_digest") in (None, digest)
            ok = r.get("S") is not None and not r.get("reason")
            rows.append({"rollout_id": r["rollout_id"], "task_id": r["task_id"], "label": int(st["label"]),
                         "score": float(r["S"]) if ok else float("inf"), "valid": ok})
        res.setdefault(f"probe:{pool}", {})["lcb_conflicting"] = cell(rows)
    for pool, art in (("token_mean", "tap_p2_cv_logistic_mean.json"), ("ema_max", "tap_p2_cv_logistic.json")):
        digest = json.loads((DATA / "lcb_oneoff/probes" / pool / art).read_text())["digest"]
        bucket = {}
        for r in jl(DATA / "lcb_oneoff/probes" / pool / "scores_all.jsonl"):
            if split.get(r["task_id"]) != "test" or r.get("label") not in (0, 1):
                continue
            assert r.get("artifact_digest") in (None, digest)
            ok = r.get("S") is not None and not r.get("reason")
            bucket[r["rollout_id"]] = {"rollout_id": r["rollout_id"], "task_id": r["task_id"], "label": int(r["label"]),
                                       "score": float(r["S"]) if ok else float("inf"), "valid": ok}
        res.setdefault(f"probe:{pool}", {})["lcb_oneoff"] = cell(list(bucket.values()))
    return res


def from_reports():
    """LiveCodeBench cells read from the aggregate reports."""
    res = {}
    t1 = json.loads((REP / "lcb_conflicting_monitors_report.json").read_text())["terminal"]
    main = json.loads((REP / "lcb_conflicting_main_report.json").read_text())["probe_terminal"]
    one = json.loads((REP / "lcb_oneoff_report.json").read_text())["test"]["detectors"]
    for m in ("gemma4-e4b", "gemma4-e4b-think"):
        for v in ("black_box", "cot"):
            e = t1[f"{m}/{v}"]["metrics"]
            res.setdefault(f"{m}:{v}", {})["lcb_conflicting"] = {
                "auroc": e["fprs"]["0.10"]["auroc"]["point"], "valid": e["n_eval_primary_scored"] - e["n_eval_invalid_scores"],
                "n": e["n_eval_primary_scored"], "from_report": True}
            o = one[f"{m}:{v}"]
            res[f"{m}:{v}"]["lcb_oneoff"] = {"auroc": o["auroc"]["point"], "valid": o["n_valid"], "n": o["n"], "from_report": True}
    for key, pool in (("probe:token_mean", "mean"), ("probe:ema_max", "ema_max")):
        p = main[pool]["fprs"]["0.10"]["auroc"]["point"]
        res.setdefault(key, {})["lcb_conflicting"] = {"auroc": p, "valid": 206, "n": 206, "from_report": True}
        o = one[key]
        res[key]["lcb_oneoff"] = {"auroc": o["auroc"]["point"], "valid": o["n_valid"], "n": o["n"], "from_report": True}
    fs = json.loads((REP / "lcb_conflicting_final_code_e4b_report.json").read_text())["test"]["models"]
    fo = json.loads((REP / "lcb_oneoff_final_code_e4b_report.json").read_text())["test"]["detectors"]
    for m in ("gemma4-e4b", "gemma4-e4b-think"):
        r = fs[m]["final_state"]
        res.setdefault(f"{m}.final_state:black_box", {})["lcb_conflicting"] = {
            "auroc": r["valid_auroc"]["point"], "valid": r["score_present"] - r["invalid_present"], "n": r["expected"],
            "from_report": True}
        o = fo[f"{m}.final_state:black_box"]
        res[f"{m}.final_state:black_box"]["lcb_oneoff"] = {"auroc": o["auroc"]["point"], "valid": o["n_valid"],
                                                           "n": o["n"], "from_report": True}
    return res


def swe(name, out_dir):
    d = DATA / name
    out = out_dir / f"{name}_report.json"
    probes = [f"{d}/probes/ema_max/tap_p2_cv_logistic.json={d}/probes/ema_max/scores_all.jsonl",
              f"{d}/probes/token_mean/tap_p2_cv_logistic_mean.json={d}/probes/token_mean/scores_all.jsonl"]
    cmd = [sys.executable, "-m", "evals_runner", "rh-eq2-cv-report", "--splits", str(d / "splits_v2.json"),
           "--monitor-dirs", str(d / "monitors"), "--reference", "gemma4-e4b:black_box",
           "--include-variants", "final_state", "--probe", *probes, "--out", str(out)]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    rep = json.loads(out.read_text())
    pop = rep["test"]["population"]
    print(f"  test population: {pop['n_tasks']} tasks, {pop['n']} episodes, {pop['n_pos']} positive")
    for c in rep["test"]["probe_vs_reference"]["contrasts"]:
        print(f"  paired AUROC difference {c['a']} - {c['b']}: {c['point']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
              f" on {c['n_matched']} common episodes")
    res = {}
    for k, v in rep["test"]["detectors"].items():
        res[k] = {name: {"auroc": v["auroc"]["point"], "ci": [v["auroc"]["ci_low"], v["auroc"]["ci_high"]],
                         "valid": v["n_valid"], "n": v["n"]}}
    return res


def merge(*parts):
    out = {}
    for p in parts:
        for k, v in p.items():
            out.setdefault(k, {}).update(v)
    return out


def show(title, rows, got):
    print(f"\n{title}\nAUROC valid/episodes; * = read from a shipped aggregate report")
    print(f"{'detector':62s}" + "".join(f"{c:>20s}" for c in COLS))
    for key, label in rows:
        line = f"{label + ' ' + key:62s}"
        for c in COLS:
            g = got.get(key, {}).get(c)
            if g is None or g.get("auroc") is None:
                line += f"{'--':>20s}"
                continue
            line += f"{g['auroc']:.3f} {g['valid']}/{g['n']}{'*' if g.get('from_report') else ' '}".rjust(20)
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results_regenerated" / "eq2")
    ap.add_argument("--skip-swe", action="store_true", help="skip the two slower SWE-bench recomputations")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    # report values first, then every recomputed cell overrides them
    parts = [from_reports(), lcb_recompute(), lcb_recompute(".final_state"), lcb_probes()]
    lcb_population()
    if not args.skip_swe:
        parts += [swe("swe_conflicting", args.out), swe("swe_oneoff", args.out)]
    got = merge(*parts)
    (args.out / "eq2_cells.json").write_text(json.dumps(got, indent=1) + "\n")
    show("EQ2 table: AUROC, transcript and private-reasoning monitors and probes", EQ2_ROWS, got)
    show("EQ3 table: AUROC, outside monitors given only the task and submitted code", EQ3_ROWS, got)
    print(f"\nwrote {args.out / 'eq2_cells.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
