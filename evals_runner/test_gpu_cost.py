import time

import pytest

from evals_runner import gpu_cost
from evals_runner.gpu_cost import GpuBusySampler


def test_gpu_busy_seconds_integrates_device_utilization():
    sampler = GpuBusySampler(interval=0.002, read_util=lambda: 50.0).start()
    time.sleep(0.02)
    seconds = sampler.stop()
    assert 0.005 < seconds < 0.1


class FakeNvml:
    def __init__(self):
        self.values = {"GPU-a": 11.0, "GPU-b": 22.0, "GPU-c": 33.0}

    def device_uuids(self):
        return list(self.values)

    def utilization(self, uuid):
        return self.values[uuid]


def test_ctypes_nvml_reader_uses_slurm_physical_gpu_for_remapped_cuda_ordinal(monkeypatch):
    fake = FakeNvml()
    monkeypatch.setattr(gpu_cost, "_CtypesNvml", lambda: fake)
    monkeypatch.setenv("SLURM_JOB_GPUS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    read = GpuBusySampler._nvml_reader()
    assert read() == 33.0


def test_resolver_maps_local_visible_ordinal_by_allocation_order():
    assert gpu_cost.resolve_gpu_uuid(["GPU-a", "GPU-b", "GPU-c"],
                                     slurm_job_gpus="gpu:1,gpu:2", cuda_visible_devices="1") == "GPU-c"


def test_uuid_visible_device_is_checked_against_slurm_allocation():
    assert gpu_cost.resolve_gpu_uuid(["GPU-a", "GPU-b"],
                                     slurm_job_gpus="0,1", cuda_visible_devices="GPU-b") == "GPU-b"
    with pytest.raises(RuntimeError, match="not in SLURM_JOB_GPUS"):
        gpu_cost.resolve_gpu_uuid(["GPU-a", "GPU-b"],
                                  slurm_job_gpus="0", cuda_visible_devices="GPU-b")


def test_ambiguous_numeric_cuda_ordinal_fails_closed_without_slurm():
    with pytest.raises(RuntimeError, match="ambiguous"):
        gpu_cost.resolve_gpu_uuid(["GPU-a", "GPU-b"], cuda_visible_devices="0")


def test_slurm_gpu_ranges_resolve_to_physical_uuid():
    assert gpu_cost.resolve_gpu_uuid(["GPU-a", "GPU-b", "GPU-c"], slurm_job_gpus="1-2") == "GPU-b"
