import copy
import json

import pytest

pytest.importorskip("inspect_ai")

from evals.reward_hacking import swebench_tasks
from evals.reward_hacking.scripts import swebench_pilot as pilot


def test_pilot_selection_is_fixed_training_only():
    ids = swebench_tasks.pilot_ids()
    assert pilot.pilot_ids(all_ids=True) == ids
    assert pilot.pilot_ids(selected_id=ids[2]) == [ids[2]]
    test_id = swebench_tasks.partition_ids("test")[0]
    with pytest.raises(ValueError, match="not in the frozen"):
        pilot.pilot_ids(selected_id=test_id)
    with pytest.raises(ValueError, match="exactly one"):
        pilot.pilot_ids(selected_id=ids[0], all_ids=True)


def test_image_manifest_requires_verified_pins_for_every_selected_id(tmp_path):
    known = "pytest-dev__pytest-5787"
    digest = "sha256:" + "a" * 64
    payload = {
        "version": 1,
        "dataset_revision": pilot.DATASET_REVISION,
        "images": {
            known: {
                "image": swebench_tasks.image_name(known).removesuffix(":latest"),
                "manifest_digest": digest,
                "sif_sha256": "b" * 64,
                "verified": True,
            }
        },
    }
    path = tmp_path / "images.json"
    path.write_text(json.dumps(payload))
    digests, entries, sha = pilot.load_image_pins(path, [known])
    assert digests == {known: digest}
    assert entries[known]["sif_sha256"] == "b" * 64
    assert len(sha) == 64

    with pytest.raises(ValueError, match="missing verified image pin"):
        pilot.load_image_pins(path, [known, "sympy__sympy-16597"])

    bad = copy.deepcopy(payload)
    bad["images"][known]["manifest_digest"] = "sha256:abc"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="full OCI manifest digest"):
        pilot.load_image_pins(path, [known])


def test_checked_in_pilot_image_manifest_pins_all_four_ids():
    digests, entries, _ = pilot.load_image_pins(
        pilot.DEFAULT_IMAGE_MANIFEST, swebench_tasks.pilot_ids()
    )
    assert set(digests) == set(swebench_tasks.pilot_ids())
    assert all(entry["verified"] is True for entry in entries.values())


def test_run_manifest_rejects_config_drift(tmp_path):
    path = tmp_path / "manifest.json"
    first = {"model": pilot.MODEL, "ids": swebench_tasks.pilot_ids()}
    pilot._load_or_write_run_manifest(path, first)
    pilot._load_or_write_run_manifest(path, first)
    with pytest.raises(ValueError, match="manifest mismatch"):
        pilot._load_or_write_run_manifest(path, {**first, "model": "other/model"})
