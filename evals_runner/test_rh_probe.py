"""rh-probe on synthetic activation files and trajectories (no model, no GPU)."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from evals.reward_hacking import activations as act
from evals.reward_hacking import metrics
from evals.reward_hacking import probe as P
from evals.reward_hacking import tap as T
from evals.reward_hacking import trajectory as tj
from evals_runner import rh_probe as RP

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = 12
U = np.zeros(D)
U[:3] = 1 / np.sqrt(3)
V = np.zeros(D)
V[5:8] = 1 / np.sqrt(3)
LAYERS = (2, 3)


def make_traj(rid, task, label, calls=4):
    t = tj.Trajectory(meta={"rollout_id": rid, "task_id": task})
    for c in range(calls):
        t.add(tj.MODEL_CALL, {"tokens": [0] * 10, "prompt_tokens": 100}, call_index=c)
        t.add(tj.REASONING, {"text": "x"}, call_index=c)
        if c < calls - 1:
            t.add(tj.TOOL_PROPOSED, {"call_id": f"c{c}", "name": "bash", "arguments": {}, "raw": "", "parse_error": None},
                  call_index=c)
            t.add(tj.TOOL_EXECUTED, {"call_id": f"c{c}", "name": "bash", "arguments": {}, "modified_by_wrapper": False})
            t.add(tj.TOOL_RESULT, {"call_id": f"c{c}", "name": "bash", "output": "ok", "error_type": None})
        else:
            t.add(tj.SUBMISSION, {"attempt": 1, "func_py": "x", "test_py": "", "answer": "a"}, call_index=c)
    t.add(tj.EVALUATION, {"attempt": 1, "passed": bool(label)})
    return t


def test_primary_population_excludes_budget_truncated_nonhacks():
    statuses = [
        RP.Rollout("hack", "t1", "train", 1, "passed", "/a"),
        RP.Rollout("honest", "t2", "train", 0, "attempts_exhausted", "/b"),
        RP.Rollout("flagged", "t3", "train", 0, "flagged", "/c"),
        RP.Rollout("cutoff", "t4", "train", 0, "message_limit", "/d"),
        RP.Rollout("infra_hack", "t5", "train", 1, "infra_error", "/e"),
    ]
    rows = statuses + [RP.Rollout(r.rollout_id + "v", r.task_id + "v", "val", r.label, r.status, r.acts_path)
                       for r in statuses]
    for partition in ("train", "val"):
        assert {s.rollout_id for s in RP._samples(rows, partition, "primary")} == {
            "hack" + ("v" if partition == "val" else ""),
            "honest" + ("v" if partition == "val" else ""),
            "flagged" + ("v" if partition == "val" else ""),
        }
        assert len(RP._samples(rows, partition, "all_labelled")) == 5


def test_scan_and_score_use_the_labelled_submission_cutoff_for_both_classes(tmp_path):
    acts, trajs = tmp_path / "acts", tmp_path / "trajs"
    acts.mkdir(); trajs.mkdir()
    specs = [("positive", "tp", 1), ("negative", "tn", 0)]
    assignments = {"tp": "train", "tn": "train"}
    for rid, task, label in specs:
        t = tj.Trajectory(meta={"rollout_id": rid, "task_id": task})
        t.add(tj.MODEL_CALL, {"tokens": [0]}, call_index=0)
        t.add(tj.TOOL_PROPOSED, {"name": "submit"}, call_index=0)
        # The submit tool proposal is not the observation boundary: the generated call is
        # processed in full before SUBMISSION is recorded.
        t.add(tj.TOOL_PROPOSED, {"name": "bash"}, call_index=0)
        t.add(tj.SUBMISSION, {"attempt": 1}, call_index=None)
        t.add(tj.EVALUATION, {"attempt": 1, "passed": bool(label)})
        if not label:
            t.add(tj.MODEL_CALL, {"tokens": [1]}, call_index=1)
            t.add(tj.SUBMISSION, {"attempt": 2}, call_index=None)
            t.add(tj.EVALUATION, {"attempt": 2, "passed": False})
        t.add(tj.MODEL_CALL, {"tokens": [2]}, call_index=2)  # must be excluded
        (trajs / f"{rid}.json").write_text(t.to_json())
        ci = np.array([0, 0, 1, 1, 2, 2], np.int32)
        h = np.zeros((6, D), np.float32)
        h[:, 0] = [2, 2, -1, -1, 100, 100]
        ex = act.Extracted(ci, np.arange(6, dtype=np.int32), np.zeros(6, np.int32),
                           np.full(6, act.REASONING, np.int8), {LAYERS[0]: h}, {})
        act.save_extracted(acts / f"{rid}.npz", ex)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"assignment": assignments}))

    rows = RP.scan_rollouts(acts, trajs, assignments)
    by_id = {r.rollout_id: r for r in rows}
    assert by_id["positive"].max_call == 1
    assert by_id["negative"].max_call == 2
    samples = RP._samples(rows, "train")
    assert {s.rollout_id: s.max_call for s in samples} == {"positive": 1, "negative": 2}

    art = T.TapArtifact(layer=LAYERS[0], w=np.r_[1.0, np.zeros(D - 1)], b=0.0, mu=None, sigma=None,
                        alpha=1.0, threshold=0.0)
    assert RP.score_rollout(art, by_id["positive"])["n_scored"] == 2
    assert RP.score_rollout(art, by_id["negative"])["n_scored"] == 4


def test_dom_validation_grid_uses_the_same_terminal_cutoff(tmp_path):
    ex = act.Extracted(np.array([0, 0, 1, 1], np.int32), np.arange(4, dtype=np.int32),
                       np.zeros(4, np.int32), np.full(4, act.REASONING, np.int8),
                       {LAYERS[0]: np.array([[2] + [0] * (D - 1), [2] + [0] * (D - 1),
                                             [100] + [0] * (D - 1), [100] + [0] * (D - 1)], np.float32)}, {})
    path = tmp_path / "val.npz"
    act.save_extracted(path, ex)
    sample = P.RolloutSample("val", "task", 1, "val", 1, str(path))
    art = T.TapArtifact(layer=LAYERS[0], w=np.r_[1.0, np.zeros(D - 1)], b=0.0, mu=None, sigma=None,
                        alpha=1.0, threshold=0.0)
    scores = RP._val_S_grid([sample], {(LAYERS[0], 1.0): art}, workers=1)
    assert scores[(LAYERS[0], 1.0)] == pytest.approx([2.0])


def make_ex(rng, label, signal_dirs, n=120, calls=4, strength=4.0):
    h = {}
    for L in LAYERS:
        x = rng.normal(size=(n, D))
        if label:
            x[n // 3:] += strength * signal_dirs[L]
        h[L] = x.astype(np.float32)
    ci = np.sort(rng.integers(0, calls, n)).astype(np.int32)
    ch = np.full(n, act.REASONING, np.int8)
    ch[::11] = act.END
    return act.Extracted(ci, np.arange(n, dtype=np.int32), np.zeros(n, np.int32), ch, h, {"layers": list(LAYERS)})


class World:
    """tmp dirs: acts/, traj/, splits.json, dom fit dir."""

    def __init__(self, tmp, specs, signal_dirs, seed=0):
        """specs: list of (rollout_id, task, partition, label, has_acts)."""
        self.acts, self.traj = tmp / "acts", tmp / "traj"
        self.acts.mkdir(), self.traj.mkdir()
        rng = np.random.default_rng(seed)
        assign = {}
        for rid, task, part, label, has in specs:
            assign[task] = part
            (self.traj / f"{rid}.json").write_text(make_traj(rid, task, label).to_json())
            if has:
                act.save_extracted(self.acts / f"{rid}.npz", make_ex(rng, label, signal_dirs))
        self.splits = tmp / "splits.json"
        self.splits.write_text(json.dumps({"assignment": assign}))
        self.fit = tmp / "fit"
        self.fit.mkdir()

    def write_dom(self, dirs, holdout_auroc):
        rep = {"layers": {}}
        for L in LAYERS:
            w = dirs[L]
            art = T.TapArtifact(layer=L, w=w, b=0.0, mu=np.zeros(D), sigma=np.ones(D), alpha=0.05, threshold=0.0,
                                channels=("tool_call", "visible"), method="dom")
            art.save(self.fit / f"tap_dom_L{L}.json")
            np.savez(self.fit / f"dom_L{L}.npz", w=w, b=np.float64(0.0), mu=np.zeros(D), sigma=np.ones(D))
            rep["layers"][str(L)] = {"holdout": {"auroc_S_alpha0.05": holdout_auroc[L], "auroc_mean_score": holdout_auroc[L]}}
        (self.fit / "dom_fit.json").write_text(json.dumps(rep))

    def args(self, *extra):
        return ["--acts", str(self.acts), "--rollouts", str(self.traj), "--splits", str(self.splits), *extra]


def specs_for(n_val_pos_tasks=6, with_test=False, n_train=6):
    s = []
    for i in range(n_train):
        s.append((f"tr{i}", f"ttr{i}", "train", i % 2, True))
    for i in range(8):                                           # val: tasks vp* positive, vn* negative
        if i < n_val_pos_tasks:
            s += [(f"vp{i}", f"tvp{i}", "val", 1, True)]
        s += [(f"vn{i}", f"tvn{i}", "val", 0, True), (f"vm{i}", f"tvn{i}", "val", 0, True)]
    if with_test:
        s += [("te0", "tte0", "test", 1, True), ("te1", "tte1", "test", 0, True)]
    return s


def run(argv):
    return RP.main(argv)


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path, specs_for(), {2: U, 3: U})
    w.write_dom({2: U, 3: U}, {2: 0.6, 3: 0.9})
    return w


# ---- P1 -------------------------------------------------------------------------------------------------------

def test_select_dom_picks_best_val_auroc_and_calibrates_on_val_negatives(tmp_path):
    w = World(tmp_path, specs_for(), {2: V, 3: U})        # only layer 3 carries the signal at U
    w.write_dom({2: U, 3: U}, {2: 0.9, 3: 0.5})           # held-out pair AUROC would prefer layer 2 (must not be used)
    out = tmp_path / "p1"
    assert run(["select-dom", "--fit", str(w.fit), "--layers", "2", "3", *w.args("--out", str(out))]) == 0
    rep = json.loads((out / "select_p1_report.json").read_text())
    art = T.TapArtifact.load(out / "tap_p1_dom.json")
    assert rep["rule"] == "val_auroc_of_S" and rep["chosen"]["layer"] == 3 and art.layer == 3
    assert len(rep["grid"]) == 2 * 4
    aucs = {(g["layer"], g["alpha"]): g["val_auroc"] for g in rep["grid"]}
    assert max(aucs.values()) == aucs[(3, rep["chosen"]["alpha"])] and aucs[(3, rep["chosen"]["alpha"])] > aucs[(2, 0.02)]
    assert art.channels == ("reasoning", "tool_call", "visible") and art.ema_init == "first"
    assert art.digest == rep["digest"]
    # threshold = threshold_at_fpr on val NEGATIVES' S only (recomputed independently)
    neg = []
    for rid in [r for r in (p.stem for p in w.traj.glob("*.json")) if r.startswith(("vn", "vm"))]:
        neg.append(T.score_extracted(art, act.load_extracted(w.acts / f"{rid}.npz")).S)
    assert art.threshold == metrics.threshold_at_fpr(neg, 0.10) == rep["threshold_fpr10"]
    assert rep["threshold_fpr5"] == metrics.threshold_at_fpr(neg, 0.05)
    assert art.extra["target_fpr"] == 0.10 and art.extra["dom_fit_sha256"]


def test_select_dom_ties_keep_earlier_grid_entry(world, tmp_path):
    out = tmp_path / "p1"
    # both layers use the same perfectly separating direction and alpha barely matters: AUROC ties at 1.0
    run(["select-dom", "--fit", str(world.fit), "--layers", "2", "3", "--alphas", "1.0", "0.2",
         *world.args("--out", str(out))])
    rep = json.loads((out / "select_p1_report.json").read_text())
    top = max(g["val_auroc"] for g in rep["grid"])
    first = next(g for g in rep["grid"] if g["val_auroc"] == top)
    assert rep["chosen"] == {"layer": first["layer"], "alpha": first["alpha"]}


def test_select_dom_fallback_when_few_positive_tasks(tmp_path):
    w = World(tmp_path, specs_for(n_val_pos_tasks=3), {2: U, 3: V})   # val AUROC would favour layer 2
    w.write_dom({2: U, 3: U}, {2: 0.55, 3: 0.95})
    out = tmp_path / "p1"
    run(["select-dom", "--fit", str(w.fit), "--layers", "2", "3", *w.args("--out", str(out))])
    rep = json.loads((out / "select_p1_report.json").read_text())
    assert rep["rule"].startswith("fallback") and rep["n_val_pos_tasks"] == 3
    assert rep["chosen"] == {"layer": 3, "alpha": 0.05}
    art = T.TapArtifact.load(out / "tap_p1_dom.json")
    assert (art.layer, art.alpha) == (3, 0.05)
    # the threshold is still calibrated on val negatives
    assert art.threshold == rep["threshold_fpr10"]


def test_select_and_train_refuse_test_rollouts_before_reading_activations(tmp_path):
    w = World(tmp_path, specs_for(with_test=True), {2: U, 3: U})
    w.write_dom({2: U, 3: U}, {2: 0.6, 3: 0.9})
    (w.acts / "tr0.npz").write_bytes(b"corrupt")          # would fail if read before the guard
    for argv in (["select-dom", "--fit", str(w.fit), "--layers", "2", "3"], ["train-logistic", "--layers", "2"]):
        out = tmp_path / ("o_" + argv[0])
        with pytest.raises(SystemExit) as e:
            run([*argv, *w.args("--out", str(out))])
        assert e.value.code == 2 and "test" in str(e.value)
        assert not out.exists()


# ---- P2 -------------------------------------------------------------------------------------------------------

def test_train_logistic_freezes_artifact(tmp_path):
    w = World(tmp_path, specs_for(n_train=10), {2: U, 3: U})
    out = tmp_path / "p2"
    assert run(["train-logistic", "--layers", "2", "3", "--alphas", "0.05", "0.2", *w.args("--out", str(out))]) == 0
    art = T.TapArtifact.load(out / "tap_p2_logistic.json")
    rep = json.loads((out / "select_p2_report.json").read_text())
    assert art.method == "logistic_token" and art.digest == rep["digest"]
    assert len(rep["grid"]) == 2 * 3 * 2
    assert art.extra["fit_kw"] == {"max_tokens_per_rollout": 4096, "class_balance": False} and art.extra["seed"] == 0
    assert art.extra["target_fpr"] == 0.10 and rep["threshold_fpr5"] >= rep["threshold_fpr10"]
    assert rep["val_auroc"] > 0.9


# ---- score ----------------------------------------------------------------------------------------------------

def frozen(w, tmp_path, **kw):
    art = T.TapArtifact(layer=3, w=U, b=-0.5, mu=np.zeros(D), sigma=np.ones(D), alpha=0.2, threshold=1.0,
                        method="dom", **kw)
    p = tmp_path / "art.json"
    art.save(p)
    return art, p


def test_score_schema_call_scores_and_decision_points(tmp_path):
    w = World(tmp_path, specs_for(), {2: U, 3: U})
    art, ap = frozen(w, tmp_path)
    out = tmp_path / "scores.jsonl"
    assert run(["score", "--artifact", str(ap), *w.args("--out", str(out))]) == 0
    rows = RP.read_rows(out)
    assert set(rows) == {p.stem for p in w.traj.glob("*.json")}
    r = rows["vp0"]
    assert set(r) >= {"rollout_id", "task_id", "partition", "label", "S", "call_scores", "n_scored", "artifact_digest"}
    assert r["artifact_digest"] == art.digest and r["label"] == 1 and r["partition"] == "val"
    ex = act.load_extracted(w.acts / "vp0.npz")
    ref = T.score_extracted(art, ex)
    assert r["S"] == pytest.approx(ref.S) and r["n_scored"] == ref.n_scored
    # call_scores: keyed by call_index, max-so-far (non-decreasing), final == S, equals tap.score_extracted
    cs = {int(k): v for k, v in r["call_scores"].items()}
    assert cs == {c: pytest.approx(v) for c, v in ref.call_scores.items()}
    vals = [v for _, v in sorted(cs.items()) if v is not None]
    assert vals == sorted(vals) and vals[-1] == pytest.approx(r["S"])
    # consistent with an independent per-call recomputation of the max-so-far
    s = art.token_scores(ex.hidden[3])
    keep = np.isin(ex.channel, art.channel_codes)
    e = T.ema_scan(s[keep], art.alpha, float(s[keep][0]))
    ci = ex.call_index[keep]
    for c, v in cs.items():
        m = e[ci <= c]
        assert v == pytest.approx(float(m.max()) if len(m) else None) if len(m) else v is None
    # and decision_point_scores can consume the row directly
    traj = tj.Trajectory.from_json((w.traj / "vp0.json").read_text())
    dps = T.decision_point_scores(traj, {int(k): v for k, v in r["call_scores"].items()})
    assert [d["call_index"] for d in dps] == [0, 1, 2, 3] and all(d["score"] == cs[d["call_index"]] for d in dps)
    man = json.loads((tmp_path / "scores.jsonl.manifest.json").read_text())
    assert man["include_test"] is False and man["artifact_digest"] == art.digest


def test_score_fail_closed_and_resumable(tmp_path):
    specs = specs_for()[:4] + [("nomiss", "tnm", "train", 1, False)]
    w = World(tmp_path, specs, {2: U, 3: U})
    art, ap = frozen(w, tmp_path)
    out = tmp_path / "scores.jsonl"
    run(["score", "--artifact", str(ap), *w.args("--out", str(out))])
    rows = RP.read_rows(out)
    assert rows["nomiss"]["S"] is None and rows["nomiss"]["reason"] == "missing_activations"
    assert all(r["S"] is not None for k, r in rows.items() if k != "nomiss")
    # activations appear later; torn trailing line; finished rows are not recomputed, the failed one is retried
    before = out.read_text()
    act.save_extracted(w.acts / "nomiss.npz", make_ex(np.random.default_rng(9), 1, {2: U, 3: U}))
    out.write_text(before + '{"rollout_id": "torn", "S"')
    run(["score", "--artifact", str(ap), *w.args("--out", str(out))])
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip().endswith("}")]
    n = {}
    for line in lines:
        n[line["rollout_id"]] = n.get(line["rollout_id"], 0) + 1
    assert n["nomiss"] == 2 and all(v == 1 for k, v in n.items() if k != "nomiss")
    assert RP.read_rows(out)["nomiss"]["S"] is not None
    # unreadable file -> load_error row, S null
    (w.acts / "tr0.npz").write_bytes(b"junk")
    out2 = tmp_path / "s2.jsonl"
    run(["score", "--artifact", str(ap), *w.args("--out", str(out2))])
    assert RP.read_rows(out2)["tr0"]["reason"] == "load_error" and RP.read_rows(out2)["tr0"]["S"] is None


def test_score_no_eligible_tokens_is_null_with_reason(tmp_path):
    w = World(tmp_path, [("a", "ta", "train", 1, True)], {2: U, 3: U})
    ex = act.load_extracted(w.acts / "a.npz")
    ex.channel[:] = act.END
    act.save_extracted(w.acts / "a.npz", ex)
    _, ap = frozen(w, tmp_path)
    out = tmp_path / "s.jsonl"
    run(["score", "--artifact", str(ap), *w.args("--out", str(out))])
    r = RP.read_rows(out)["a"]
    assert r["S"] is None and r["reason"] == "no_eligible_tokens" and set(r["call_scores"].values()) == {None}


def test_score_test_guard_and_include_test_manifest(tmp_path):
    w = World(tmp_path, specs_for(with_test=True), {2: U, 3: U})
    art, ap = frozen(w, tmp_path)
    out = tmp_path / "s.jsonl"
    with pytest.raises(SystemExit) as e:
        run(["score", "--artifact", str(ap), *w.args("--out", str(out))])
    assert e.value.code == 2 and not out.exists()
    run(["score", "--artifact", str(ap), "--include-test", *w.args("--out", str(out))])
    assert RP.read_rows(out)["te0"]["partition"] == "test"
    man = json.loads((tmp_path / "s.jsonl.manifest.json").read_text())
    assert man["include_test"] is True and man["n_test_rows"] == 2


def test_score_refuses_file_from_another_artifact(tmp_path):
    w = World(tmp_path, specs_for(), {2: U, 3: U})
    a1, p1 = frozen(w, tmp_path)
    out = tmp_path / "s.jsonl"
    run(["score", "--artifact", str(p1), *w.args("--out", str(out))])
    a2 = T.TapArtifact(layer=3, w=U, b=0.0, mu=np.zeros(D), sigma=np.ones(D), alpha=0.2, threshold=1.0, method="dom")
    p2 = tmp_path / "a2.json"
    a2.save(p2)
    with pytest.raises(SystemExit):
        run(["score", "--artifact", str(p2), *w.args("--out", str(out))])


def test_artifact_digest_stable_across_runs_and_roundtrip(world, tmp_path):
    d = []
    for i in range(2):
        out = tmp_path / f"p1_{i}"
        run(["select-dom", "--fit", str(world.fit), "--layers", "2", "3", *world.args("--out", str(out))])
        d.append(T.TapArtifact.load(out / "tap_p1_dom.json").digest)
    assert d[0] == d[1]
    a = T.TapArtifact.load(tmp_path / "p1_0" / "tap_p1_dom.json")
    assert T.TapArtifact.from_json(a.to_json()).digest == a.digest


def test_registered_in_cli():
    p = subprocess.run([sys.executable, "-m", "evals_runner", "rh-probe", "score", "--help"],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0 and "--include-test" in p.stdout


# ---- train-cv ----------------------------------------------------------------------------------

def cv_specs(n_pos=10, n_neg=14, per_task=2, with_test=False, with_val=False):
    s = []
    for i in range(n_pos):
        s += [(f"p{i}_{j}", f"tp{i}", "train", 1 if j == 0 else 0, True) for j in range(per_task)]
    for i in range(n_neg):
        s += [(f"n{i}_{j}", f"tn{i}", "train", 0, True) for j in range(per_task)]
    if with_val:
        s += [("v0", "tv0", "val", 1, True), ("v1", "tv1", "val", 0, True)]
    if with_test:
        s += [("te0", "tte0", "test", 1, True), ("te1", "tte1", "test", 0, True)]
    return s


def _cv_samples(specs):
    return [P.RolloutSample(rid, task, label, "train") for rid, task, _p, label, _h in specs]


def test_assign_folds_groups_tasks_is_deterministic_and_spreads_positives():
    sm = _cv_samples(cv_specs())
    f1, f2 = RP.assign_folds(sm, 5, 0), RP.assign_folds(sm, 5, 0)
    assert f1 == f2 and set(f1.values()) == set(range(5)) and len(f1) == 24
    assert RP.assign_folds(sm, 5, 1) != f1
    pos = [f1[t] for t in {s.task_id for s in sm if s.label == 1}]
    counts = [pos.count(f) for f in range(5)]
    assert max(counts) - min(counts) <= 1 and sum(counts) == 10
    tot = [sum(1 for s in sm if f1[s.task_id] == f) for f in range(5)]
    assert max(tot) - min(tot) <= 2


def test_train_cv_selects_informative_layer_and_thresholds_from_oof_negatives(tmp_path):
    w = World(tmp_path, cv_specs(with_val=True), {2: np.zeros(D), 3: U})     # layer 2 carries no signal
    out = tmp_path / "cv"
    assert run(["train-cv", "--layers", "2", "3", "--alphas", "0.05", "0.2", *w.args("--out", str(out))]) == 0
    art = T.TapArtifact.load(out / "tap_p2_cv_logistic.json")
    rep = json.loads((out / "select_p2_cv_report.json").read_text())
    x = art.extra
    assert art.layer == 3 and art.digest == rep["digest"] and x["selection"] == "5-fold task-grouped CV"
    assert len(x["grid"]) == 2 * 3 * 2 and all("oof_auroc" in g for g in x["grid"])
    assert x["oof_auroc"] > 0.9 and rep["oof_auroc"] == x["oof_auroc"]
    assert x["n_train"] == 50 and x["n_train_tasks"] == 26 and x["n_train_pos"] == 11     # val counted as train
    assert x["n_val_entries_treated_as_train"] == 2 and len(x["splits_sha256"]) == 64
    # same-task rollouts share a fold; all tasks present
    assert x["folds"]["tp0"] in range(5) and set(x["folds"].values()) == set(range(5))
    neg = [v for rid, v in x["oof_scores"].items() if rid.startswith("n") or rid.endswith("_1") or rid == "v1"]
    assert art.threshold == metrics.threshold_at_fpr(neg, 0.10)
    assert x["threshold_fpr5"] == metrics.threshold_at_fpr(neg, 0.05) and x["threshold_fpr5"] >= art.threshold
    out2 = tmp_path / "cv2"
    run(["train-cv", "--layers", "2", "3", "--alphas", "0.05", "0.2", *w.args("--out", str(out2))])
    assert T.TapArtifact.load(out2 / "tap_p2_cv_logistic.json").digest == art.digest


def test_train_cv_never_reads_test_rollouts_and_token_mean_runs(tmp_path):
    w = World(tmp_path, cv_specs(with_test=True), {2: U, 3: U})
    (w.acts / "te0.npz").write_bytes(b"corrupt")
    out = tmp_path / "o"
    with pytest.raises(SystemExit) as e:
        run(["train-cv", "--layers", "2", *w.args("--out", str(out))])
    assert e.value.code == 2 and not out.exists()
    w2 = World(tmp_path / "w2" if (tmp_path / "w2").mkdir() is None else tmp_path, cv_specs(), {2: U, 3: U})
    assert run(["train-cv", "--layers", "3", "--pooling", "token_mean", *w2.args("--out", str(tmp_path / "m"))]) == 0
    assert T.TapArtifact.load(tmp_path / "m" / "tap_p2_cv_logistic_mean.json").pooling == "token_mean"


def test_train_cv_skip_test_drops_test_rollouts_unread(tmp_path):
    w = World(tmp_path, cv_specs(with_test=True), {2: U, 3: U})
    (w.acts / "te0.npz").write_bytes(b"corrupt")          # would fail if the test rollout were read
    out = tmp_path / "o"
    assert run(["train-cv", "--layers", "3", "--skip-test", *w.args("--out", str(out))]) == 0
    art = T.TapArtifact.load(out / "tap_p2_cv_logistic.json")
    assert not any(k.startswith("te") for k in art.extra["oof_scores"])


def test_mean_pooled_score_of_mean_vector_equals_mean_of_token_scores():
    # train-cv's streaming mean path scores the mean vector; val_token_scores scores tokens. Linearity makes them equal.
    import numpy as np
    rng = np.random.default_rng(0)
    H = rng.normal(size=(37, 8))
    p = P.fit_logistic_mean_X(rng.normal(size=(20, 8)), np.array([0, 1] * 10, dtype=np.float32), l2=0.1)
    wv = p["w"] / p["sigma"]
    b0 = p["b"] - p["mu"] @ wv
    assert np.isclose((H @ wv + b0).mean(), ((H.mean(0) - p["mu"]) / p["sigma"]) @ p["w"] + p["b"])


def _cv_run(argv, full_cache):
    ap = RP.argparse.ArgumentParser(prog="rh-probe")
    RP.configure_parser(ap)
    a = ap.parse_args(argv)
    a._full_cache = full_cache
    return RP.run(a)


@pytest.mark.parametrize("extra", [[], ["--ema-scope", "call"]])
def test_train_cv_streaming_equals_full_cache_reference(tmp_path, extra):
    w = World(tmp_path, cv_specs(with_val=True), {2: np.zeros(D), 3: U})
    outs = {}
    for name, full in (("new", False), ("old", True)):
        out = tmp_path / name
        assert _cv_run(["train-cv", "--layers", "2", "3", "--alphas", "0.05", "0.2", *extra,
                        *w.args("--out", str(out))], full) == 0
        outs[name] = (T.TapArtifact.load(out / "tap_p2_cv_logistic.json"),
                      json.loads((out / "select_p2_cv_report.json").read_text()))
    (an, rn), (ao, ro) = outs["new"], outs["old"]
    assert an.digest == ao.digest and rn["digest"] == ro["digest"]
    assert rn["chosen"] == ro["chosen"] and rn["grid"] == ro["grid"]
    xn, xo = an.extra["oof_scores"], ao.extra["oof_scores"]
    assert xn.keys() == xo.keys()
    np.testing.assert_allclose([np.nan if v is None else v for v in xn.values()],
                               [np.nan if v is None else v for v in xo.values()], rtol=0, atol=1e-10)
    assert np.array_equal(an.w, ao.w) and an.b == ao.b and an.threshold == ao.threshold


def test_compact_train_pass_is_bit_identical_to_train_pass(tmp_path):
    w = World(tmp_path, cv_specs(n_pos=4, n_neg=4), {2: U, 3: U})
    sm = P.build_samples(w.acts, w.traj, P.load_splits(w.splits))
    sm = [s for s in sm if s.label in (0, 1)]
    ch = act.DEFAULT_CHANNELS
    for mtok in (16, 4096):                      # subsampling active / inactive
        n0, (r0, s0, l0) = P.train_pass(sm, 3, ch, mtok, 7)
        items = P.compact_pass(sm, 3, ch, mtok, 7)
        n1, (r1, s1, l1) = P.train_pass_from_compact(items)
        assert np.array_equal(n0.mu, n1.mu) and np.array_equal(n0.sigma, n1.sigma) and s0 == s1
        assert all(np.array_equal(a, b) for a, b in zip(r0, r1)) and np.array_equal(l0, l1)
        assert max(s1) <= mtok
        p0 = P.fit_logistic_token(sm, 3, norm=n0, seed=7, max_tokens_per_rollout=mtok, pre=(r0, s0, l0))
        p1 = P.fit_logistic_token(None, 3, norm=n1, seed=7, max_tokens_per_rollout=mtok, pre=(r1, s1, l1))
        assert np.array_equal(p0["w"], p1["w"]) and p0["b"] == p1["b"]
