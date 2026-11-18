"""NVML sampler for serialized monitor cost runs, implemented without pynvml.

The utilization integral is a secondary GPU-busy-seconds statistic. The request's
allocated GPU wall time remains the per-check GPU-seconds measure. Device selection
must be grounded in a physical NVML UUID or Slurm's physical GPU allocation; an
ambiguous CUDA ordinal fails closed instead of silently measuring another GPU.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import threading
import time


def _normal_uuid(value: str) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _expand_gpu_ids(value: str) -> list[str]:
    """Parse common SLURM_JOB_GPUS forms: `0,2`, `gpu:0`, and numeric ranges."""
    out = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        token = token.split(":")[-1]
        m = re.fullmatch(r"(\d+)-(\d+)", token)
        if m:
            lo, hi = map(int, m.groups())
            if hi < lo:
                raise RuntimeError(f"invalid SLURM_JOB_GPUS range {raw!r}")
            out.extend(str(i) for i in range(lo, hi + 1))
        else:
            out.append(token)
    return out


def resolve_gpu_uuid(device_uuids, *, slurm_job_gpus=None, cuda_visible_devices=None) -> str:
    """Resolve the first visible allocated GPU to its physical NVML UUID.

    Numeric CUDA_VISIBLE_DEVICES values are local/remapped ordinals under Slurm,
    so their position maps to the ordered physical SLURM_JOB_GPUS list. Outside
    Slurm only UUID-based CUDA visibility is accepted; numeric ordinals are too
    ambiguous to report as a physical device safely.
    """
    uuids = [_normal_uuid(x) for x in device_uuids]
    by_lower = {u.lower(): u for u in uuids}
    slurm_ids = _expand_gpu_ids(slurm_job_gpus) if slurm_job_gpus else []

    def physical_id(token):
        t = token.strip()
        if t.lower() in by_lower:
            return by_lower[t.lower()]
        if t.isdigit():
            idx = int(t)
            if idx >= len(uuids):
                raise RuntimeError(f"GPU index {idx} from SLURM_JOB_GPUS is outside NVML device count {len(uuids)}")
            return uuids[idx]
        raise RuntimeError(f"cannot resolve allocated GPU identifier {token!r} to an NVML UUID")

    allocated = [physical_id(x) for x in slurm_ids]
    cvd = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()] if cuda_visible_devices is not None else []
    if slurm_ids:
        if not allocated:
            raise RuntimeError("SLURM_JOB_GPUS did not identify any GPU")
        if not cvd:
            return allocated[0]
        first = cvd[0]
        if first.lower() in by_lower:
            chosen = by_lower[first.lower()]
            if chosen not in allocated:
                raise RuntimeError(f"CUDA_VISIBLE_DEVICES UUID {first!r} is not in SLURM_JOB_GPUS allocation")
            return chosen
        if not first.isdigit():
            raise RuntimeError(f"cannot resolve CUDA_VISIBLE_DEVICES entry {first!r}")
        # Slurm remaps physical devices to local CUDA ordinals (typically 0..n-1).
        local = int(first)
        if local >= len(allocated):
            raise RuntimeError(f"CUDA local ordinal {local} is outside the {len(allocated)} allocated GPU(s)")
        return allocated[local]
    if cvd and cvd[0].lower() in by_lower:
        return by_lower[cvd[0].lower()]
    if cvd:
        raise RuntimeError("numeric CUDA_VISIBLE_DEVICES without SLURM_JOB_GPUS is ambiguous; refusing GPU cost sample")
    raise RuntimeError("cannot identify allocated GPU: neither SLURM_JOB_GPUS nor a CUDA UUID is available")


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _CtypesNvml:
    """Small NVML binding for UUID and utilization calls; no Python package required."""

    def __init__(self):
        path = ctypes.util.find_library("nvidia-ml")
        if not path:
            raise RuntimeError("NVML library libnvidia-ml was not found")
        self.lib = ctypes.CDLL(path)
        self._bind("nvmlInit_v2", [], ctypes.c_int, fallback="nvmlInit")
        self._bind("nvmlDeviceGetCount_v2", [ctypes.POINTER(ctypes.c_uint)], ctypes.c_int,
                   fallback="nvmlDeviceGetCount")
        self._bind("nvmlDeviceGetHandleByIndex_v2", [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                   ctypes.c_int, fallback="nvmlDeviceGetHandleByIndex")
        self._bind("nvmlDeviceGetUUID", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint], ctypes.c_int)
        self._bind("nvmlDeviceGetHandleByUUID", [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int)
        self._bind("nvmlDeviceGetUtilizationRates", [ctypes.c_void_p, ctypes.POINTER(_Utilization)], ctypes.c_int)
        self._check(self.lib.nvmlInit_v2(), "nvmlInit")
        count = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "nvmlDeviceGetCount")
        self.handles_by_uuid = {}
        for i in range(count.value):
            handle = ctypes.c_void_p()
            self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)), "nvmlDeviceGetHandleByIndex")
            buf = ctypes.create_string_buffer(96)
            self._check(self.lib.nvmlDeviceGetUUID(handle, buf, len(buf)), "nvmlDeviceGetUUID")
            self.handles_by_uuid[buf.value.decode()] = handle

    def _bind(self, name, argtypes, restype, fallback=None):
        symbol = name if hasattr(self.lib, name) else fallback
        if symbol is None or not hasattr(self.lib, symbol):
            raise RuntimeError(f"NVML library lacks required function {name}")
        fn = getattr(self.lib, symbol)
        fn.argtypes = argtypes
        fn.restype = restype
        if symbol != name:
            setattr(self.lib, name, fn)

    def _check(self, rc, operation):
        if rc != 0:
            detail = ""
            if hasattr(self.lib, "nvmlErrorString"):
                self.lib.nvmlErrorString.restype = ctypes.c_char_p
                detail = f": {self.lib.nvmlErrorString(rc).decode(errors='replace')}"
            raise RuntimeError(f"{operation} failed with NVML error {rc}{detail}")

    def device_uuids(self):
        return list(self.handles_by_uuid)

    def utilization(self, uuid):
        handle = self.handles_by_uuid[uuid]
        rates = _Utilization()
        self._check(self.lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(rates)),
                    "nvmlDeviceGetUtilizationRates")
        return float(rates.gpu)


class GpuBusySampler:
    def __init__(self, *, interval=0.05, read_util=None):
        self.interval = interval
        self.read_util = read_util or self._nvml_reader()
        self.stop_event = threading.Event()
        self.thread = None
        self.samples = []

    @staticmethod
    def _nvml_reader():
        nvml = _CtypesNvml()
        uuid = resolve_gpu_uuid(nvml.device_uuids(),
                                slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
                                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        return lambda: nvml.utilization(uuid)

    def start(self):
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        self.samples.append((time.perf_counter(), self.read_util() / 100.0))

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def stop(self):
        self._sample()
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        return sum((t1 - t0) * (u0 + u1) / 2 for (t0, u0), (t1, u1) in zip(self.samples, self.samples[1:]))
