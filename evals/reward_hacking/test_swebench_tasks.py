from __future__ import annotations

import copy
import json

import pytest

from evals.reward_hacking import swebench_tasks as st


def test_committed_split_manifest_is_pinned_and_complete():
    manifest = st.load_manifest()
    assert manifest["dataset"] == "fjzzq2002/impossible_swebench"
    assert manifest["revision"] == "9c2d34f364b7229e8c0ff807c646100bdc18bbb5"
    assert {name: len(rows) for name, rows in manifest["partitions"].items()} == {
        "train": 60, "val": 40, "test": 249,
    }
    assert st.pilot_ids(manifest=manifest) == [
        "matplotlib__matplotlib-25122",
        "django__django-15863",
        "pytest-dev__pytest-5787",
        "sympy__sympy-16597",
    ]
    assert set(st.pilot_ids(manifest=manifest)) <= set(st.partition_ids("train", manifest=manifest))


def test_manifest_validation_rejects_duplicate_ids(tmp_path):
    manifest = st.load_manifest()
    invalid = copy.deepcopy(manifest)
    invalid["partitions"]["val"][0]["instance_id"] = invalid["partitions"]["train"][0]["instance_id"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="duplicate"):
        st.load_manifest(path)


def test_select_rows_preserves_requested_order_and_rejects_missing_ids():
    rows = [{"instance_id": "b", "value": 2}, {"instance_id": "a", "value": 1}]
    assert st.select_rows(rows, ["a", "b"]) == [rows[1], rows[0]]
    with pytest.raises(ValueError, match="missing"):
        st.select_rows(rows, ["c"])
    with pytest.raises(ValueError, match="duplicates"):
        st.select_rows(rows, ["a", "a"])
    with pytest.raises(ValueError, match="duplicate instance_id"):
        st.select_rows([*rows, rows[1]], ["a"])


def test_official_image_name_uses_escaped_instance_id():
    assert st.image_name("Matplotlib__Matplotlib-25122") == (
        "swebench/sweb.eval.x86_64.matplotlib_1776_matplotlib-25122:latest"
    )


def test_partition_and_split_names_are_validated():
    with pytest.raises(ValueError, match="unknown"):
        st.partition_ids("development")
    with pytest.raises(ValueError, match="covers only"):
        st.load_rows("train", split="original")
