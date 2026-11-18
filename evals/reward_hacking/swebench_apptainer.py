"""Run the official ImpossibleBench SWE task/scorer with Apptainer (for hosts without Docker).

The upstream task factory currently hard-codes Docker image preparation and a
Docker sandbox type. This adapter keeps its dataset, solver, and scorer, but
disables its Docker client, selects the published per-instance image name, and
binds that image directly to the registered ``ct_swe_apptainer`` provider.
"""

from __future__ import annotations

from unittest.mock import patch

from inspect_ai.dataset import MemoryDataset
from inspect_ai.util import SandboxEnvironmentSpec

# Import for its @sandboxenv registration side effect.
from evals.reward_hacking import inspect_apptainer as _inspect_apptainer  # noqa: F401

DATASET_REVISION = "9c2d34f364b7229e8c0ff807c646100bdc18bbb5"


def dockerhub_image_for_id(instance_id: str, arch: str = "x86_64", digest: str | None = None) -> str:
    """Docker Hub's official SWE-bench image tag for an instance."""
    normalized = instance_id.replace("__", "_1776_").lower()
    image = f"swebench/sweb.eval.{arch}.{normalized}"
    if digest is not None:
        if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
            raise ValueError("image digests must be full sha256:<64 hex chars>")
        return f"{image}@{digest}"
    return f"{image}:latest"


def impossible_swebench_apptainer(*, split: str, **kwargs):
    """Build ImpossibleBench's official Inspect task using Apptainer.

    Example::

        task = impossible_swebench_apptainer(
            split="conflicting", instance_ids=["astropy__astropy-123"],
            agent_type="tools", limit=1,
        )
        eval(task, model="openai/my-open-weight-endpoint")
    """
    from impossiblebench.swebench_tasks import impossible_swebench

    instance_ids = kwargs.get("instance_ids")
    if not instance_ids:
        raise ValueError("pass an explicit instance_ids list; the benchmark split name is a task variant, not a held-out partition")
    image_digests = kwargs.pop("image_digests", {})

    def image_for(instance_id: str) -> str:
        return dockerhub_image_for_id(str(instance_id), digest=image_digests.get(str(instance_id)))

    module = __import__("impossiblebench.swebench_tasks", fromlist=["hf_dataset"])
    original_hf_dataset = module.hf_dataset

    def pinned_hf_dataset(*args, **dataset_kwargs):
        path = dataset_kwargs.get("path", args[0] if args else None)
        if path == "fjzzq2002/impossible_swebench":
            supplied = dataset_kwargs.get("revision")
            if supplied not in (None, DATASET_REVISION):
                raise ValueError(f"dataset revision is pinned to {DATASET_REVISION}, got {supplied}")
            dataset_kwargs["revision"] = DATASET_REVISION
        return original_hf_dataset(*args, **dataset_kwargs)

    # The upstream factory doesn't expose HF revision. Dataset loading is
    # synchronous, so the patch is scoped to this call and restored immediately.
    with patch.object(module, "hf_dataset", pinned_hf_dataset):
        task = impossible_swebench(
            split=split,
            sandbox_type="docker",  # upstream task factory only accepts docker/k8s
            build_docker_images=False,
            docker_image_from_id=image_for,
            **kwargs,
        )
    samples = list(task.dataset)
    for sample in samples:
        sample.metadata = sample.metadata or {}
        sample.metadata["instance_id"] = str(sample.id)
        sample.sandbox = SandboxEnvironmentSpec(
            "ct_swe_apptainer", image_for(str(sample.id))
        )
    task.dataset = MemoryDataset(samples, name=f"{task.name}_apptainer")
    # Task.name is read-only in recent Inspect releases. The dataset name makes
    # the backend visible in logs without mutating the official task identity.
    return task
