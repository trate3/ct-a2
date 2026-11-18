import argparse
import json

import pytest

from evals_runner import rh_t1_invalid_sensitivity as S


def _row(rid, task, label, score, *, present=True):
    return {"rollout_id": rid, "task_id": task, "label": label,
            "status_class": "hack" if label else "honest", "score": score,
            "valid": score is not None, "score_present": present,
            "score_error": present and score is None}


def test_invalid_scores_are_max_suspicion_and_missing_are_excluded():
    rows = [_row("h1", "t1", 1, .8), _row("n1", "t2", 0, .2),
            _row("n2", "t3", 0, None), _row("h2", "t4", 1, .4, present=False)]
    summary = S.view_summary(rows, n_boot=20, seed=0)
    assert summary["n_primary_expected"] == 4
    assert summary["n_score_present"] == 3
    assert summary["n_invalid_present"] == 1
    assert summary["n_missing"] == 1
    assert summary["auroc_complete_valid_scores"]["point"] == 1.0
    assert summary["auroc_invalid_as_max_suspicion"]["point"] == pytest.approx(.5)
    assert "rollout_id" not in summary


def test_paired_black_box_cot_uses_same_scored_ids_and_clustered_bootstrap():
    bb = [_row("h1", "task1", 1, .8), _row("n1", "task2", 0, .1),
          _row("h2", "task3", 1, .7), _row("n2", "task4", 0, None),
          _row("missing", "task5", 0, None, present=False)]
    cot = [_row("h1", "task1", 1, .8), _row("n1", "task2", 0, .1),
           _row("h2", "task3", 1, .7), _row("n2", "task4", 0, .2)]
    out = S._paired_auc(bb, cot, invalid_as_max=True, n_boot=30, seed=0)
    assert out["n_common_scored"] == 4
    assert out["n_common_for_auc"] == 4
    assert out["n_a_missing"] == 1
    assert out["auroc_difference_a_minus_b"]["point"] == pytest.approx(-.5)
    assert "rollout_id" not in str(out)


def test_test_guard_refuses_before_any_eval_path_read(tmp_path):
    args = argparse.Namespace(include_test=False, eval_rollouts=tmp_path / "not-opened",
                              eval_terminal=["x=/not-opened"], freeze_manifest=None)
    with pytest.raises(SystemExit, match="refusing to read"):
        S.validate_access(args)


def test_analysis_hash_freeze_required_for_eval_inputs(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"test_access_approved": True, "analysis_sha256": "wrong"}))
    args = argparse.Namespace(include_test=True, eval_rollouts=tmp_path / "test",
                              eval_terminal=["x=missing"], freeze_manifest=freeze)
    with pytest.raises(SystemExit, match="exact supplemental analysis SHA-256"):
        S.validate_access(args)
