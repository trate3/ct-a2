"""Tests for evals/reward_hacking/metrics.py.

Covers frozen-threshold metrics (missing classes, achieved test FPR) and the
task-level clustering of the bootstrap. Brute-force reference implementations of
AUROC/AUPRC (written against a different, O(n^2)-ish formulation, not the
vectorized production code) cross-check tie handling.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from evals.reward_hacking import metrics as m


# ---------------------------------------------------------------------------
# brute-force references

def brute_auroc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    total = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                total += 1.0
            elif p == n:
                total += 0.5
    return total / (len(pos) * len(neg))


def brute_auprc(labels, scores):
    labels = list(labels)
    scores = list(scores)
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    distinct = sorted(set(scores), reverse=True)
    total = 0.0
    for v in distinct:
        tp = sum(1 for y, s in zip(labels, scores) if s >= v and y == 1)
        fp = sum(1 for y, s in zip(labels, scores) if s >= v and y == 0)
        precision = tp / (tp + fp)
        n_pos_at_v = sum(1 for y, s in zip(labels, scores) if s == v and y == 1)
        total += n_pos_at_v * precision
    return total / n_pos


@pytest.mark.parametrize("seed", range(20))
def test_auroc_matches_brute_force_with_ties(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 14)
    labels = [rng.randint(0, 1) for _ in range(n)]
    # small discrete score alphabet forces ties
    scores = [rng.choice([0.0, 0.1, 0.2, 0.2, 0.3]) for _ in range(n)]
    got = m.auroc(labels, scores)
    want = brute_auroc(labels, scores)
    if want is None:
        assert got is None
    else:
        assert got == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_auprc_matches_brute_force_with_ties(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 14)
    labels = [rng.randint(0, 1) for _ in range(n)]
    scores = [rng.choice([0.0, 0.1, 0.2, 0.2, 0.3]) for _ in range(n)]
    got = m.auprc(labels, scores)
    want = brute_auprc(labels, scores)
    if want is None:
        assert got is None
    else:
        assert got == pytest.approx(want, abs=1e-9)


def test_auroc_no_ties_matches_hand_example():
    # pos=[0.9, 0.4], neg=[0.1, 0.2, 0.4] -> one tie (pos 0.4 == neg 0.4)
    labels = [1, 1, 0, 0, 0]
    scores = [0.9, 0.4, 0.1, 0.2, 0.4]
    # pairs: (0.9,0.1)>,(0.9,0.2)>,(0.9,0.4)> = 3
    #        (0.4,0.1)>,(0.4,0.2)>,(0.4,0.4)== 2 + 0.5 = 2.5
    # total 5.5 / (2*3) = 0.91666...
    assert m.auroc(labels, scores) == pytest.approx(5.5 / 6, abs=1e-9)


# ---------------------------------------------------------------------------
# missing-class None handling

def test_auroc_auprc_none_when_class_missing():
    assert m.auroc([1, 1, 1], [0.1, 0.5, 0.9]) is None
    assert m.auroc([0, 0], [0.1, 0.9]) is None
    assert m.auprc([1, 1, 1], [0.1, 0.5, 0.9]) is None
    assert m.auprc([0, 0], [0.1, 0.9]) is None


def test_auroc_auprc_none_when_empty():
    assert m.auroc([], []) is None
    assert m.auprc([], []) is None


def test_auroc_scores_with_none_raises():
    with pytest.raises(ValueError):
        m.auroc([1, 0], [0.5, None])


# ---------------------------------------------------------------------------
# threshold_at_fpr / frozen_threshold_metrics

def test_threshold_at_fpr_achieves_target_on_calibration_set():
    neg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # n=10, no ties
    thr = m.threshold_at_fpr(neg, target_fpr=0.2)
    # k = floor(0.2*10) = 2 -> threshold is the 3rd-largest value (index 2 desc) = 0.8
    assert thr == pytest.approx(0.8)
    frozen = m.frozen_threshold_metrics([0] * len(neg), neg, thr)
    assert frozen["fpr"] == pytest.approx(0.2)
    assert frozen["tpr"] is None  # no positives in this calibration-only check


def test_threshold_at_fpr_handles_ties_conservatively():
    neg = [0.5, 0.5, 0.5, 0.3, 0.2]  # n=5, target 0.4 -> k=2, but the top-2 are tied with a 3rd
    thr = m.threshold_at_fpr(neg, target_fpr=0.4)
    assert thr == pytest.approx(0.5)
    achieved_fpr = sum(1 for s in neg if s > thr) / len(neg)
    assert achieved_fpr <= 0.4
    assert achieved_fpr == pytest.approx(0.0)  # ties push it below the nominal target


def test_threshold_at_fpr_unachievable_small_n_undershoots():
    neg = [0.1, 0.2, 0.3]  # n=3, target 0.1 -> floor(0.1*3) = 0
    thr = m.threshold_at_fpr(neg, target_fpr=0.1)
    assert thr == pytest.approx(max(neg))
    assert sum(1 for s in neg if s > thr) == 0


def test_threshold_at_fpr_target_one_flags_everything():
    neg = [0.1, 0.5, 0.9]
    thr = m.threshold_at_fpr(neg, target_fpr=1.0)
    assert all(s > thr for s in neg)


def test_threshold_at_fpr_requires_negatives():
    with pytest.raises(ValueError):
        m.threshold_at_fpr([], target_fpr=0.1)


def test_threshold_at_fpr_rejects_out_of_range_target():
    with pytest.raises(ValueError):
        m.threshold_at_fpr([0.1, 0.2], target_fpr=1.5)


def test_frozen_threshold_metrics_reports_achieved_fpr_on_test_set_not_assumed_target():
    # threshold calibrated on one (calibration) negative population...
    cal_neg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thr = m.threshold_at_fpr(cal_neg, target_fpr=0.2)
    # ...applied to a *different* test population whose achieved FPR need not equal 0.2
    test_labels = [0, 0, 0, 0, 1, 1]
    test_scores = [0.1, 0.85, 0.95, 0.99, 0.95, 0.2]
    out = m.frozen_threshold_metrics(test_labels, test_scores, thr)
    # 4 negatives: 0.1 (<=thr), 0.85,0.95,0.99 (>thr) -> 3/4 = 0.75, not 0.2
    assert out["fpr"] == pytest.approx(0.75)
    assert out["fpr"] != pytest.approx(0.2)
    assert out["tp"] == 1 and out["fn"] == 1  # 0.95 > thr (TP), 0.2 <= thr (FN)
    assert out["tpr"] == pytest.approx(0.5)


def test_frozen_threshold_metrics_none_for_missing_classes():
    out_no_pos = m.frozen_threshold_metrics([0, 0, 0], [0.1, 0.9, 0.9], threshold=0.5)
    assert out_no_pos["tpr"] is None
    assert out_no_pos["fpr"] == pytest.approx(2 / 3)
    out_no_neg = m.frozen_threshold_metrics([1, 1], [0.1, 0.9], threshold=0.5)
    assert out_no_neg["fpr"] is None
    assert out_no_neg["tpr"] == pytest.approx(0.5)
    out_no_pred_pos = m.frozen_threshold_metrics([0, 1], [0.1, 0.2], threshold=0.9)
    assert out_no_pred_pos["precision"] is None


# ---------------------------------------------------------------------------
# cluster_bootstrap keeps whole tasks together

def _rows_fixture_two_rows_per_task(n_tasks=6):
    rows = []
    for i in range(n_tasks):
        task_id = f"t{i}"
        label = i % 2  # alternate so both classes exist
        for j in range(2):
            rows.append({"rollout_id": f"{task_id}-{j}", "task_id": task_id,
                        "label": label, "score": 0.1 + 0.05 * (i + j)})
    return rows


def test_cluster_bootstrap_keeps_clusters_intact():
    rows = _rows_fixture_two_rows_per_task()

    def intact_fn(sample_rows):
        counts: dict[str, int] = {}
        for r in sample_rows:
            counts[r["task_id"]] = counts.get(r["task_id"], 0) + 1
        # every task_id present in a resample must appear with a count that is a
        # multiple of 2 (its fixed per-task row count); a resampler that drew
        # individual rows instead of whole clusters would break this.
        return 1.0 if all(c % 2 == 0 for c in counts.values()) else 0.0

    out = m.cluster_bootstrap(rows, intact_fn, cluster_key="task_id", n_boot=300, seed=1)
    assert out["point"] == 1.0
    assert out["n_undefined"] == 0
    assert out["n_valid"] == 300
    assert out["ci_low"] == 1.0
    assert out["ci_high"] == 1.0
    assert out["n_clusters"] == 6


def test_cluster_bootstrap_point_is_unresampled_statistic():
    rows = _rows_fixture_two_rows_per_task()
    out = m.cluster_bootstrap(rows, m._rows_auroc, cluster_key="task_id", n_boot=50, seed=0)
    assert out["point"] == pytest.approx(m._rows_auroc(rows))


def test_cluster_bootstrap_counts_undefined_draws():
    # One task holds all the positives; when it's never drawn, AUROC is undefined.
    rows = [
        {"rollout_id": "a", "task_id": "pos_task", "label": 1, "score": 0.9},
        {"rollout_id": "b", "task_id": "neg_task_1", "label": 0, "score": 0.1},
        {"rollout_id": "c", "task_id": "neg_task_2", "label": 0, "score": 0.2},
    ]
    out = m.cluster_bootstrap(rows, m._rows_auroc, cluster_key="task_id", n_boot=500, seed=2)
    assert out["n_undefined"] > 0
    assert out["n_undefined"] + out["n_valid"] == 500


# ---------------------------------------------------------------------------
# paired_difference

def _paired_fixture():
    # Every task contributes one positive and one negative row, so any bootstrap
    # draw (which always resamples at least one cluster) always has both classes
    # represented -- AUROC is defined on every draw, with no undefined draws to
    # muddy the "self-diff is exactly 0" check.
    rows = []
    for i in range(5):
        task_id = f"t{i}"
        rows.append({"rollout_id": f"r{i}a", "task_id": task_id, "label": 1,
                    "score": 0.1 + 0.1 * i})
        rows.append({"rollout_id": f"r{i}b", "task_id": task_id, "label": 0,
                    "score": 0.05 + 0.1 * i})
    return rows


def test_paired_difference_self_is_zero_every_draw():
    rows = _paired_fixture()
    out = m.paired_difference(rows, rows, m._rows_auroc, cluster_key="task_id",
                              n_boot=300, seed=3)
    assert out["point"] == 0.0
    assert out["n_undefined"] == 0
    assert out["ci_low"] == 0.0
    assert out["ci_high"] == 0.0


def test_paired_difference_requires_matching_rollout_ids():
    rows_a = _paired_fixture()
    rows_b = [dict(r, rollout_id=r["rollout_id"] + "-x") for r in rows_a]
    with pytest.raises(ValueError):
        m.paired_difference(rows_a, rows_b, m._rows_auroc)


def test_paired_difference_nonzero_for_different_scores():
    rows_a = _paired_fixture()
    rows_b = [dict(r, score=1.0 - r["score"]) for r in rows_a]  # reversed ranking
    out = m.paired_difference(rows_a, rows_b, m._rows_auroc, cluster_key="task_id",
                              n_boot=300, seed=4)
    assert out["point"] == pytest.approx(m._rows_auroc(rows_a) - m._rows_auroc(rows_b))
    assert out["point"] != 0.0


# ---------------------------------------------------------------------------
# length_strata

def test_length_strata_partitions_all_rows_into_bins_by_rank():
    rows = [{"rollout_id": str(i), "task_id": str(i), "label": i % 2,
            "score": 0.5, "prompt_tokens": i} for i in range(9)]
    out = m.length_strata(rows, "prompt_tokens", n_bins=3)
    assert out["n_bins"] == 3
    assert sum(b["n"] for b in out["bins"]) == 9
    assert [b["n"] for b in out["bins"]] == [3, 3, 3]
    assert out["bins"][0]["length_max"] < out["bins"][1]["length_min"]
    assert out["bins"][1]["length_max"] < out["bins"][2]["length_min"]


def test_length_strata_empty_rows():
    out = m.length_strata([], "prompt_tokens", n_bins=3)
    assert [b["n"] for b in out["bins"]] == [0, 0, 0]
