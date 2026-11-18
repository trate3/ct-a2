"""The commit's artifact hashing: same digests, fewer reads, tamper check intact."""

from __future__ import annotations

import json

import pytest

import ct_core.certification as certification
from ct_core.certification import committed_artifact_bytes, prepare_components
from ct_core.interfaces import AgentDriver, ArtifactPackage, TapDriver


class _FileAgent(AgentDriver):
    DRIVER = "file-agent/v1"
    on_load = None

    @classmethod
    def code_paths(cls):
        return ()

    @classmethod
    def from_package(cls, config, artifacts):
        if cls.on_load:
            cls.on_load(artifacts)
        return cls()


class _FileTap(TapDriver):
    DRIVER = "file-tap/v1"

    @classmethod
    def code_paths(cls):
        return ()

    @classmethod
    def from_package(cls, config, artifacts):
        return cls()


def _shards(tmp_path, n=3):
    paths = []
    for index in range(n):
        path = tmp_path / f"model-{index:05d}.safetensors"
        path.write_bytes(bytes([index]) * (1024 * 1024 + index))
        paths.append(path)
    return paths


def _packages(shards, tap_shares_model=True):
    files = tuple((f"model/{p.name}", str(p)) for p in shards)
    agent = ArtifactPackage(_FileAgent.DRIVER, {}, files)
    tap = ArtifactPackage(_FileTap.DRIVER, {}, files if tap_shares_model else ())
    return agent, tap


def test_parallel_digests_match_the_sequential_definition(tmp_path):
    shards = _shards(tmp_path)
    digests = certification._file_digests(*_packages(shards))
    assert digests == {str(p): certification._file_digest(p) for p in shards}


def test_a_shared_checkpoint_is_hashed_once_per_pass(tmp_path, monkeypatch):
    shards = _shards(tmp_path)
    reads = []
    original = certification._file_digest
    monkeypatch.setattr(certification, "_file_digest",
                        lambda path: reads.append(str(path)) or original(path))
    agent, tap = _packages(shards, tap_shares_model=True)
    components = prepare_components(agent, tap, (_FileAgent,), (_FileTap,))
    # two passes (before and after loading), each file once per pass
    assert sorted(reads) == sorted([str(p) for p in shards] * 2)
    record = json.loads(components.encoded)
    assert record["agent"]["artifacts"] == record["tap"]["artifacts"]
    assert committed_artifact_bytes(agent, tap) == 2 * sum(p.stat().st_size for p in shards)


def test_commitment_is_unchanged_by_the_hashing_schedule(tmp_path, monkeypatch):
    shards = _shards(tmp_path)
    parallel = prepare_components(*_packages(shards), (_FileAgent,), (_FileTap,)).encoded
    monkeypatch.setattr(certification, "_file_digests", lambda *packages: {})
    sequential = prepare_components(*_packages(shards), (_FileAgent,), (_FileTap,)).encoded
    assert parallel == sequential


def test_artifact_changed_during_load_is_still_rejected(tmp_path, monkeypatch):
    shards = _shards(tmp_path)

    def tamper(artifacts):
        with open(shards[1], "r+b") as handle:
            handle.write(b"\xff")

    monkeypatch.setattr(_FileAgent, "on_load", staticmethod(tamper))
    with pytest.raises(ValueError, match="changed while it was loaded"):
        prepare_components(*_packages(shards), (_FileAgent,), (_FileTap,))
