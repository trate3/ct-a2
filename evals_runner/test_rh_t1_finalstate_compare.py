import json

from evals_runner.rh_t1_finalstate_compare import _load_view_scores, _paired_auc


def _row(rid, task, label, score, cls="hack", present=True):
    return {"rollout_id": rid, "task_id": task, "label": label,
            "score": score, "valid": score is not None, "score_present": present,
            "status_class": cls}


def test_paired_auc_is_primary_only_and_task_clustered():
    a = [_row("h", "task-a", 1, .9), _row("n", "task-b", 0, .1, "honest"),
         _row("x", "task-c", 0, 0.0, "budget_truncated")]
    b = [_row("h", "task-a", 1, .1), _row("n", "task-b", 0, .9, "honest"),
         _row("x", "task-c", 0, 1.0, "budget_truncated")]
    got = _paired_auc(a, b, n_boot=100, seed=0)
    assert got["n_common_for_auc"] == 2
    assert got["auroc_difference_finalstate_minus_history_only"]["point"] == 1.0


def test_score_loader_selects_black_box_from_history_family(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (
        {"rollout_id": "r1", "view": "black_box", "score": .2},
        {"rollout_id": "r1", "view": "cot", "score": .9},
    )) + "\n")
    scores = _load_view_scores({"m": path}, "history_only")
    assert list(scores) == ["m/history_only"]
    assert scores["m/history_only"]["r1"]["score"] == .2
