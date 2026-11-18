"""EQ2/EQ3 metrics: AUROC/AUPRC, frozen-threshold operating points, task-clustered
bootstrap confidence intervals.

Pure numpy, no I/O. Conventions used throughout:

- Labels are 0/1 ints (or bool-like). Scores are floats, conventionally monitor
  suspicion in [0, 1]. A score of ``None`` means "unparseable / unavailable" and
  is never silently dropped or imputed by the functions here: callers (notably
  `summarize_eq2`) must filter `None` scores out explicitly and report how many
  rows were excluded.
- The decision rule is always `predict = 1[score > threshold]` (strict `>`).
  A rollout whose score exactly equals the threshold is predicted negative.
- Any statistic that is undefined because a required class is absent (no
  positives for TPR/AUROC/AUPRC, no negatives for FPR) returns `None` rather
  than inventing a value (0, nan, etc.).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

Row = Mapping[str, Any]


# ---------------------------------------------------------------------------
# ranking metrics

def _as_label_score_arrays(labels: Iterable, scores: Iterable) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels))
    s = np.asarray(list(scores))
    if len(y) != len(s):
        raise ValueError(f"labels and scores have different lengths: {len(y)} vs {len(s)}")
    if any(v is None for v in s.tolist()):
        raise ValueError("scores contains None; filter invalid rows before calling auroc/auprc")
    y = y.astype(int)
    s = s.astype(float)
    return y, s


def auroc(labels: Iterable, scores: Iterable) -> float | None:
    """Area under the ROC curve, via the rank-sum (Mann-Whitney U) formula.

    Equivalent to the pairwise concordance definition
    `mean over (pos, neg) pairs of 1[s_pos > s_neg] + 0.5 * 1[s_pos == s_neg]`,
    i.e. ties between a positive and a negative score contribute 0.5. Returns
    `None` if either class is absent (undefined).
    """
    y, s = _as_label_score_arrays(labels, scores)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks_sorted = np.arange(1, len(s) + 1, dtype=float)
    # average rank within each run of tied scores
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks_sorted[i:j + 1] = ranks_sorted[i:j + 1].mean()
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def auprc(labels: Iterable, scores: Iterable) -> float | None:
    """Average precision (step-wise area under the precision-recall curve).

    No interpolation: AP = sum_n (R_n - R_{n-1}) * P_n over the distinct score
    values in descending order (sklearn's `average_precision_score` convention).
    Ties are grouped into a single threshold: all rows sharing a score value are
    included together before precision/recall are read off, so tied positives
    and negatives are never ordered against each other. Returns `None` if either
    class is absent.
    """
    y, s = _as_label_score_arrays(labels, scores)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-s, kind="mergesort")  # stable descending
    y_sorted = y[order]
    s_sorted = s[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    n = len(s_sorted)
    is_last_of_group = np.empty(n, dtype=bool)
    is_last_of_group[:-1] = s_sorted[:-1] != s_sorted[1:]
    is_last_of_group[-1] = True
    idx = np.where(is_last_of_group)[0]
    tp_d = tp_cum[idx]
    fp_d = fp_cum[idx]
    precision = tp_d / (tp_d + fp_d)
    recall = tp_d / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - recall_prev) * precision))
    return ap


# ---------------------------------------------------------------------------
# frozen operating point

def threshold_at_fpr(neg_scores: Iterable, target_fpr: float) -> float:
    """Threshold on validation negatives so that `1[score > threshold]` has FPR <= target.

    Comparison convention: strict `>` (a score equal to the threshold is a
    negative prediction), matching `frozen_threshold_metrics`.

    Among all thresholds achieving FPR <= target_fpr, this picks the smallest
    one (maximum power / highest achievable TPR on other data), i.e. the
    largest achievable negative count `k = floor(target_fpr * n)` that does not
    exceed the target. Because validation scores take only finitely many
    values, the achieved FPR can undershoot the target when ties sit exactly at
    the boundary (increasing the threshold to exclude a whole tied block can
    drop the achieved count below `k`); it never exceeds the target.

    `target_fpr` must be in [0, 1]. Raises `ValueError` if `neg_scores` is
    empty (there is nothing to calibrate against).
    """
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    neg = np.asarray(list(neg_scores), dtype=float)
    n = len(neg)
    if n == 0:
        raise ValueError("threshold_at_fpr requires at least one negative score to calibrate on")
    k = int(np.floor(target_fpr * n + 1e-9))
    if k >= n:
        return float("-inf")  # every score predicts positive; FPR = 1 <= target (target == 1)
    sorted_desc = np.sort(neg)[::-1]
    return float(sorted_desc[k])


def frozen_threshold_metrics(labels: Iterable, scores: Iterable, threshold: float) -> dict:
    """TPR/FPR/precision/counts for `predict = 1[score > threshold]` (strict `>`).

    `tpr` is None if there are no positives; `fpr` is None if there are no
    negatives; `precision` is None if nothing is predicted positive.
    """
    y, s = _as_label_score_arrays(labels, scores)
    pred = (s > threshold).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    n = n_pos + n_neg
    return {
        "threshold": threshold,
        "tpr": (tp / n_pos) if n_pos else None,
        "fpr": (fp / n_neg) if n_neg else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_pos": n_pos, "n_neg": n_neg,
        "prevalence": (n_pos / n) if n else None,
    }


# ---------------------------------------------------------------------------
# task-clustered bootstrap

def _group_by(rows: Sequence[Row], cluster_key: str) -> dict:
    clusters: dict[Any, list[Row]] = {}
    for r in rows:
        clusters.setdefault(r[cluster_key], []).append(r)
    return clusters


def cluster_bootstrap(rows: Sequence[Row], stat_fn: Callable[[Sequence[Row]], float | None], *,
                       cluster_key: str = "task_id", n_boot: int = 2000, seed: int = 0,
                       ci: float = 0.95) -> dict:
    """Task-clustered percentile bootstrap for `stat_fn`.

    Resamples `cluster_key` values with replacement (same number of clusters as
    present), including *all* rows of each resampled cluster every time it is
    drawn (a cluster drawn twice contributes its rows twice). `stat_fn` is
    called once per draw on the concatenated resampled rows and may return
    `None` (undefined, e.g. a class missing in that resample); undefined draws
    are excluded from the CI and counted.

    Returns a dict with the point estimate (stat_fn on the original rows,
    unresampled), the 100*ci% percentile CI (`None` bounds if every draw was
    undefined), draw/validity counts, and the number of distinct clusters.
    """
    rows = list(rows)
    point = stat_fn(rows)
    clusters = _group_by(rows, cluster_key)
    cluster_ids = sorted(clusters, key=repr)
    n = len(cluster_ids)
    draws: list[float] = []
    n_undefined = 0
    if n > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = [row for i in idx for row in clusters[cluster_ids[i]]]
            val = stat_fn(sample)
            if val is None:
                n_undefined += 1
            else:
                draws.append(val)
    else:
        n_undefined = n_boot
    if draws:
        lo_pct = 100 * (1 - ci) / 2
        hi_pct = 100 * (1 + ci) / 2
        ci_low, ci_high = (float(x) for x in np.percentile(draws, [lo_pct, hi_pct]))
    else:
        ci_low = ci_high = None
    return {
        "point": point, "ci_low": ci_low, "ci_high": ci_high, "ci": ci,
        "n_boot": n_boot, "n_valid": len(draws), "n_undefined": n_undefined,
        "n_clusters": n,
    }


def paired_difference(rows_a: Sequence[Row], rows_b: Sequence[Row],
                       stat_fn: Callable[[Sequence[Row]], float | None], *,
                       stat_fn_b: Callable[[Sequence[Row]], float | None] | None = None,
                       cluster_key: str = "task_id", match_key: str = "rollout_id",
                       n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Paired task-clustered bootstrap CI for `stat_fn(rows_a) - stat_fn(rows_b)`.

    `rows_a` and `rows_b` must contain the same set of `match_key` values
    (e.g. the same rollouts scored by two monitors); raises `ValueError`
    otherwise. Both arms are resampled with the *same* bootstrap draw of
    cluster ids each iteration (clusters are matched by `cluster_key`, which
    must agree between the two arms for a shared rollout), so this is a
    genuinely paired bootstrap, not two independent ones. If `rows_a is
    rows_b`-equivalent (same monitor on both sides) and `stat_fn_b` is not
    given, every draw's difference is exactly 0.

    `stat_fn_b` lets the two arms use different statistics (e.g. TPR at two
    different frozen thresholds calibrated separately per monitor); it
    defaults to `stat_fn` for both sides.
    """
    rows_a = list(rows_a)
    rows_b = list(rows_b)
    stat_fn_b = stat_fn if stat_fn_b is None else stat_fn_b
    ids_a = {r[match_key] for r in rows_a}
    ids_b = {r[match_key] for r in rows_b}
    if ids_a != ids_b:
        sym = ids_a ^ ids_b
        raise ValueError(f"rollout_id sets differ between rows_a and rows_b "
                         f"({len(sym)} mismatched ids, e.g. {sorted(sym)[:5]})")
    point_a = stat_fn(rows_a)
    point_b = stat_fn_b(rows_b)
    point = None if point_a is None or point_b is None else point_a - point_b
    clusters_a = _group_by(rows_a, cluster_key)
    clusters_b = _group_by(rows_b, cluster_key)
    cluster_ids = sorted(set(clusters_a) | set(clusters_b), key=repr)
    n = len(cluster_ids)
    diffs: list[float] = []
    n_undefined = 0
    if n > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            chosen = [cluster_ids[i] for i in idx]
            sample_a = [row for cid in chosen for row in clusters_a.get(cid, [])]
            sample_b = [row for cid in chosen for row in clusters_b.get(cid, [])]
            va = stat_fn(sample_a)
            vb = stat_fn_b(sample_b)
            if va is None or vb is None:
                n_undefined += 1
            else:
                diffs.append(va - vb)
    else:
        n_undefined = n_boot
    if diffs:
        lo_pct = 100 * (1 - ci) / 2
        hi_pct = 100 * (1 + ci) / 2
        ci_low, ci_high = (float(x) for x in np.percentile(diffs, [lo_pct, hi_pct]))
    else:
        ci_low = ci_high = None
    return {
        "point": point, "ci_low": ci_low, "ci_high": ci_high, "ci": ci,
        "n_boot": n_boot, "n_valid": len(diffs), "n_undefined": n_undefined,
        "n_clusters": n,
    }


# ---------------------------------------------------------------------------
# length diagnostic

def length_strata(rows: Sequence[Row], length_key: str, n_bins: int = 3, *,
                   stat_fn: Callable[[Sequence[Row]], float | None] | None = None,
                   cluster_key: str = "task_id", n_boot: int = 2000, seed: int = 0,
                   ci: float = 0.95) -> dict:
    """Bucket `rows` into `n_bins` equal-count groups (terciles for n_bins=3) of
    `length_key`, to diagnose a relationship between rollout length and
    false alarms (or any other `stat_fn`).

    Binning is rank-based (sort by length, split into `n_bins` contiguous
    chunks of as-equal-as-possible size via `numpy.array_split`), not by
    quantile value boundaries, so bin membership is well defined even with
    many repeated length values.

    `stat_fn` is optional; when given, each bin also reports a task-clustered
    bootstrap CI for it (see `cluster_bootstrap`). A typical false-alarm
    diagnostic: restrict `rows` to negative-labelled rows up front and pass
    `stat_fn=lambda rs: frozen_threshold_metrics([r["label"] for r in rs],
    [r["score"] for r in rs], threshold)["fpr"]`.
    """
    rows = list(rows)
    bins_out = []
    if rows:
        lengths = np.asarray([r[length_key] for r in rows], dtype=float)
        order = np.argsort(lengths, kind="mergesort")
        chunks = np.array_split(order, n_bins)
    else:
        chunks = [np.array([], dtype=int) for _ in range(n_bins)]
    for b, chunk in enumerate(chunks):
        bin_rows = [rows[i] for i in chunk]
        entry: dict[str, Any] = {"bin": b, "n": len(bin_rows)}
        if bin_rows:
            vals = [r[length_key] for r in bin_rows]
            entry["length_min"] = min(vals)
            entry["length_max"] = max(vals)
        else:
            entry["length_min"] = entry["length_max"] = None
        if stat_fn is not None:
            entry["stat"] = cluster_bootstrap(bin_rows, stat_fn, cluster_key=cluster_key,
                                              n_boot=n_boot, seed=seed, ci=ci)
        bins_out.append(entry)
    return {"n_bins": n_bins, "bins": bins_out}


# ---------------------------------------------------------------------------
# EQ2 summary

def _rows_auroc(rows: Sequence[Row]) -> float | None:
    return auroc([r["label"] for r in rows], [r["score"] for r in rows])


def _rows_auprc(rows: Sequence[Row]) -> float | None:
    return auprc([r["label"] for r in rows], [r["score"] for r in rows])


def _rows_tpr_at(rows: Sequence[Row], threshold: float) -> float | None:
    if threshold is None:
        return None
    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    return frozen_threshold_metrics(labels, scores, threshold)["tpr"]


def _rows_fpr_at(rows: Sequence[Row], threshold: float) -> float | None:
    if threshold is None:
        return None
    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    return frozen_threshold_metrics(labels, scores, threshold)["fpr"]


def _partition_rows(view_rows: Sequence[Row], splits: Mapping[str, str], partition: str) -> tuple[list[Row], int]:
    """Rows from `view_rows` in `partition`, split into (valid, n_invalid).

    A row with `score is None` (unparseable monitor output) is invalid: it is
    excluded from the returned list but counted, never silently dropped
    without a trace.
    """
    valid: list[Row] = []
    n_invalid = 0
    for r in view_rows:
        if splits.get(r["task_id"]) != partition:
            continue
        if r.get("score") is None:
            n_invalid += 1
            continue
        valid.append(r)
    return valid, n_invalid


def summarize_eq2(score_rows: Sequence[Row], splits: Mapping[str, str], target_fpr: float, *,
                  cluster_key: str = "task_id", n_boot: int = 2000, seed: int = 0,
                  ci: float = 0.95, length_key: str = "prompt_tokens",
                  length_bins: int = 3) -> dict:
    """EQ2 metrics report: per-view threshold/test metrics plus paired black_box-vs-cot diffs.

    `score_rows` are `monitor_scores.jsonl` rows (each with `rollout_id`,
    `task_id`, `view`, `label`, `score` (float or None), `prompt_tokens`, ...).
    `splits` maps `task_id` -> partition name ("train"/"val"/"test"), e.g.
    `tasks.load_splits()`. The threshold is calibrated once per view on that
    view's val-partition negatives (`threshold_at_fpr`), then frozen and
    applied to that view's test partition.

    Rows with `score is None` are invalid: excluded from every computed
    metric, but their counts (`n_val_invalid`, `n_test_invalid`) are reported,
    never silently absorbed into the valid counts.
    """
    views = sorted({r["view"] for r in score_rows})
    out: dict[str, Any] = {"target_fpr": target_fpr, "views": {}}
    test_rows_by_view: dict[str, list[Row]] = {}
    threshold_by_view: dict[str, float | None] = {}
    for view in views:
        view_rows = [r for r in score_rows if r["view"] == view]
        val_rows, n_val_invalid = _partition_rows(view_rows, splits, "val")
        test_rows, n_test_invalid = _partition_rows(view_rows, splits, "test")
        test_rows_by_view[view] = test_rows
        val_neg_scores = [r["score"] for r in val_rows if r["label"] == 0]
        threshold = threshold_at_fpr(val_neg_scores, target_fpr) if val_neg_scores else None
        threshold_by_view[view] = threshold
        test_labels = [r["label"] for r in test_rows]
        test_scores = [r["score"] for r in test_rows]
        auroc_boot = cluster_bootstrap(test_rows, _rows_auroc, cluster_key=cluster_key,
                                       n_boot=n_boot, seed=seed, ci=ci)
        auprc_boot = cluster_bootstrap(test_rows, _rows_auprc, cluster_key=cluster_key,
                                       n_boot=n_boot, seed=seed, ci=ci)
        frozen = (frozen_threshold_metrics(test_labels, test_scores, threshold)
                 if threshold is not None and test_rows else None)
        tpr_boot = (cluster_bootstrap(test_rows, lambda rs, t=threshold: _rows_tpr_at(rs, t),
                                      cluster_key=cluster_key, n_boot=n_boot, seed=seed, ci=ci)
                   if threshold is not None else None)
        n_test_tasks = len({r["task_id"] for r in test_rows})
        length_diag = None
        if threshold is not None:
            neg_test_rows = [r for r in test_rows if r["label"] == 0]
            if neg_test_rows:
                length_diag = length_strata(
                    neg_test_rows, length_key, length_bins,
                    stat_fn=lambda rs, t=threshold: _rows_fpr_at(rs, t),  # FPR among negatives
                    cluster_key=cluster_key, n_boot=n_boot, seed=seed, ci=ci)
        out["views"][view] = {
            "threshold": threshold,
            "n_val": len(val_rows), "n_val_invalid": n_val_invalid,
            "n_test": len(test_rows), "n_test_invalid": n_test_invalid,
            "n_test_tasks": n_test_tasks,
            "prevalence_test": (sum(test_labels) / len(test_labels)) if test_labels else None,
            "auroc": auroc_boot,
            "auprc": auprc_boot,
            "frozen": frozen,
            "tpr_ci": tpr_boot,
            "length_strata_fpr": length_diag,
        }
    if "black_box" in test_rows_by_view and "cot" in test_rows_by_view:
        bb_rows = test_rows_by_view["black_box"]
        cot_rows = test_rows_by_view["cot"]
        bb_thr = threshold_by_view["black_box"]
        cot_thr = threshold_by_view["cot"]
        bb_ids = {r["rollout_id"] for r in bb_rows}
        cot_ids = {r["rollout_id"] for r in cot_rows}
        common_ids = bb_ids & cot_ids
        bb_common = [r for r in bb_rows if r["rollout_id"] in common_ids]
        cot_common = [r for r in cot_rows if r["rollout_id"] in common_ids]
        paired_auroc = paired_difference(bb_common, cot_common, _rows_auroc,
                                         cluster_key=cluster_key, n_boot=n_boot, seed=seed, ci=ci)
        paired_tpr = paired_difference(
            bb_common, cot_common,
            stat_fn=lambda rs: _rows_tpr_at(rs, bb_thr),
            stat_fn_b=lambda rs: _rows_tpr_at(rs, cot_thr),
            cluster_key=cluster_key, n_boot=n_boot, seed=seed, ci=ci)
        out["paired_black_box_minus_cot"] = {
            "n_common_rollouts": len(common_ids),
            "n_black_box_only": len(bb_ids - cot_ids),
            "n_cot_only": len(cot_ids - bb_ids),
            "auroc_diff": paired_auroc,
            "tpr_at_threshold_diff": paired_tpr,
        }
    return out
