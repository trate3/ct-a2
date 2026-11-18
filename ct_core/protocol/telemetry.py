"""Generic stage-timing primitives used by the protocol engine.

`ct_core` depends on nothing outside itself, so the RQ1 experiment's higher-level
records (run metadata: machine, model, tap, workload) live in
`evals_runner/util/telemetry.py`, which imports these primitives rather than the
other way around. What belongs here is only what `ct_core/certification.py` itself
needs to call: a way to time a block of code and write the result as one JSONL line,
with a true no-op default so instrumentation costs nothing when it isn't requested.
"""

from __future__ import annotations

import json
import resource
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageRecord:
    run_id: str
    trial_index: int    # 0 for session-level stages not tied to one trial
    stage: str
    wall_seconds: float
    cpu_seconds: float
    bytes_processed: int = 0
    start_s: float | None = None    # seconds since the recorder was created (monotonic); with `thread`, shows overlap
    thread: str | None = None       # name of the thread that ran the block

    def to_json(self) -> dict:
        return asdict(self)


class JsonlWriter:
    """Append-only JSONL sink. One line per record, flushed immediately."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record) -> None:
        line = json.dumps(record.to_json(), sort_keys=True, separators=(",", ":"))
        with self._path.open("a") as handle:
            handle.write(line + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class StageRecorder:
    """Times blocks of code and writes one StageRecord per block.

    `run_id` ties every stage back to its run record. A `None` sink makes this a
    pure in-memory recorder (`.records`), useful for tests.
    """

    def __init__(self, run_id: str, sink: JsonlWriter | None = None):
        self.run_id = run_id
        self._sink = sink
        self._t0 = time.perf_counter()
        self.records: list[StageRecord] = []

    @contextmanager
    def stage(self, name: str, *, trial_index: int = 0, bytes_processed: int = 0):
        before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - start
            after = resource.getrusage(resource.RUSAGE_SELF)
            cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
            record = StageRecord(self.run_id, trial_index, name, wall, cpu, bytes_processed,
                                 start_s=start - self._t0, thread=threading.current_thread().name)
            self.records.append(record)
            if self._sink is not None:
                self._sink.write(record)


class NullStageRecorder:
    """No-op recorder: the default everywhere a `stage_recorder` isn't supplied.

    Lets `ct_core/certification.py` call `stage_recorder.stage(...)` unconditionally
    without branching on whether telemetry was requested.
    """

    @contextmanager
    def stage(self, name: str, *, trial_index: int = 0, bytes_processed: int = 0):
        yield
