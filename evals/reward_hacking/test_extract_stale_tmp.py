import os
import time

from evals.reward_hacking.scripts import extract_activations as ex


def test_only_old_tmp_files_are_removed(tmp_path):
    old, fresh = tmp_path / "a.npz.tmp", tmp_path / "b.npz.tmp"
    old.write_text("x"); fresh.write_text("y")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    assert ex.remove_stale_tmp(tmp_path) == 1
    assert not old.exists() and fresh.exists()
    assert ex.remove_stale_tmp(tmp_path, min_age_s=0) == 1 and not fresh.exists()
