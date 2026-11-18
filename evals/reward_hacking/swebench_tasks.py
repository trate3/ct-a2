"""Pinned Impossible-SWE-bench task selection helpers.

This module intentionally stops at data selection. The Inspect task, solver and
scorer remain owned by the pinned ImpossibleBench package; sandbox construction
is delegated to the separately configured Inspect provider.
"""

from __future__ import annotations

import json
import hashlib
import pathlib
from collections.abc import Iterable, Mapping

DATASET = "fjzzq2002/impossible_swebench"
DATASET_REVISION = "9c2d34f364b7229e8c0ff807c646100bdc18bbb5"
IMPOSSIBLEBENCH_COMMIT = "061dc3dce6a96ab6cf02a855157263033dcfa3ba"
SPLIT = "conflicting"
SPLIT_SALT = "certified-taps/swebench/v1"
SPLITS_PATH = pathlib.Path(__file__).parent / "data" / "swebench" / "splits_v1.json"


def load_manifest(path: pathlib.Path = SPLITS_PATH) -> dict:
    """Read and validate the committed task split manifest."""
    manifest = json.loads(path.read_text())
    expected = {
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "split": SPLIT,
        "salt": SPLIT_SALT,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"SWE-bench split manifest {key} mismatch: {manifest.get(key)!r}")

    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "val", "test"}:
        raise ValueError("SWE-bench split manifest must define train, val and test")
    if {name: len(rows) for name, rows in partitions.items()} != {
        "train": 60, "val": 40, "test": 249,
    }:
        raise ValueError("SWE-bench split manifest has unexpected partition sizes")
    seen: set[str] = set()
    for name, rows in partitions.items():
        if not isinstance(rows, list):
            raise ValueError(f"SWE-bench {name} partition must be a list")
        ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or not row.get("instance_id") or not row.get("repo"):
                raise ValueError(f"SWE-bench {name} rows need instance_id and repo")
            iid = row["instance_id"]
            if iid in ids or iid in seen:
                raise ValueError(f"duplicate SWE-bench instance_id: {iid}")
            ids.add(iid)
        seen.update(ids)
        ordered = sorted(ids, key=lambda iid: hashlib.sha256(
            f"{SPLIT_SALT}\0{iid}".encode("utf-8")).hexdigest())
        if [row["instance_id"] for row in rows] != ordered:
            raise ValueError(f"SWE-bench {name} partition is not in frozen hash order")

    pilot = manifest.get("pilot")
    train_ids = {r["instance_id"] for r in partitions["train"]}
    if (not isinstance(pilot, list) or len(pilot) != len(set(pilot))
            or not set(pilot) <= train_ids):
        raise ValueError("SWE-bench pilot IDs must be unique members of train")
    first_per_repo = []
    seen_repos: set[str] = set()
    for row in partitions["train"]:
        if row["repo"] not in seen_repos:
            first_per_repo.append(row["instance_id"])
            seen_repos.add(row["repo"])
    if pilot != first_per_repo[:len(pilot)] or len(pilot) != 4:
        raise ValueError("SWE-bench pilot must be the first four train instances from distinct repos")
    return manifest


def partition_rows(name: str, *, manifest: Mapping | None = None) -> list[dict]:
    """Return manifest entries in their frozen, deterministic order."""
    manifest = manifest or load_manifest()
    if name not in ("train", "val", "test"):
        raise ValueError(f"unknown SWE-bench partition: {name!r}")
    return list(manifest["partitions"][name])


def partition_ids(name: str, *, manifest: Mapping | None = None) -> list[str]:
    return [row["instance_id"] for row in partition_rows(name, manifest=manifest)]


def pilot_ids(*, manifest: Mapping | None = None) -> list[str]:
    manifest = manifest or load_manifest()
    return list(manifest["pilot"])


def select_rows(rows: Iterable[Mapping], instance_ids: Iterable[str]) -> list[dict]:
    """Select dataset rows in the supplied ID order, rejecting missing/duplicate IDs."""
    wanted = list(instance_ids)
    if len(wanted) != len(set(wanted)):
        raise ValueError("requested SWE-bench IDs contain duplicates")
    by_id: dict[str, dict] = {}
    for row in rows:
        iid = row.get("instance_id")
        if iid in by_id:
            raise ValueError(f"duplicate instance_id in SWE-bench dataset: {iid}")
        if iid in wanted:
            by_id[iid] = dict(row)
    missing = [iid for iid in wanted if iid not in by_id]
    if missing:
        raise ValueError(f"SWE-bench dataset is missing requested IDs: {missing}")
    return [by_id[iid] for iid in wanted]


def load_rows(name: str, *, split: str = SPLIT, manifest: Mapping | None = None) -> list[dict]:
    """Load the pinned Hugging Face split and return its selected partition rows."""
    if split != SPLIT:
        raise ValueError(f"the frozen split manifest covers only {SPLIT!r}, got {split!r}")
    ids = partition_ids(name, manifest=manifest)
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split=split, revision=DATASET_REVISION)
    return select_rows(dataset, ids)


def image_name(instance_id: str) -> str:
    """Official prebuilt SWE-bench x86_64 image name for an instance."""
    encoded = instance_id.lower().replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{encoded}:latest"


def image_map(name: str = "train", *, manifest: Mapping | None = None) -> dict[str, str]:
    """Map selected instances to their official prebuilt image names."""
    return {iid: image_name(iid) for iid in partition_ids(name, manifest=manifest)}
