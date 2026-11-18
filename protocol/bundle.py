"""Canonical provider identity and streaming, inert component provisioning."""
from __future__ import annotations

from contextlib import contextmanager
import base64
import inspect
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from ct_core import certification as core
from ct_core.interfaces import ArtifactPackage

from .codec import CHUNK, MAX_ARTIFACTS, MAX_INLINE, MAX_METADATA, canonical, decode, digest, read_exact

BUNDLE_VERSION = "ct-components/1"


def _name(value: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 512
            or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", value)
            or PurePosixPath(value).is_absolute()
            or any(x in ("", ".", "..") for x in value.split("/"))):
        raise ValueError("unsafe artifact name")
    return value


def _validate_metadata(meta):
    if (not isinstance(meta, dict) or set(meta) != {"version", "components", "storage"}
            or meta["version"] != BUNDLE_VERSION
            or not isinstance(meta["components"], dict)
            or set(meta["components"]) != {"version", "agent", "tap"}
            or meta["components"]["version"] != core.COMPONENT_VERSION
            or not isinstance(meta["storage"], dict) or set(meta["storage"]) != {"agent", "tap"}):
        raise ValueError("malformed component manifest")
    inline, count = 0, 0
    for role in ("agent", "tap"):
        record, storage = meta["components"][role], meta["storage"][role]
        if (not isinstance(record, dict)
                or set(record) != {"version", "kind", "driver", "code", "config", "artifacts"}
                or record["version"] != core.COMPONENT_VERSION or record["kind"] != role
                or not isinstance(record["driver"], str) or not record["driver"]
                or not isinstance(record["code"], dict) or not isinstance(record["config"], dict)
                or not isinstance(record["artifacts"], dict) or not isinstance(storage, dict)
                or storage.keys() != record["artifacts"].keys()):
            raise ValueError("malformed component record")
        names = sorted(record["artifacts"])
        for name in names:
            _name(name)
            if any(other.startswith(name + "/") for other in names):
                raise ValueError("artifact name is a file/directory collision")
            entry = record["artifacts"][name]
            if (not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}
                    or type(entry["bytes"]) is not int or entry["bytes"] <= 0
                    or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in entry["sha256"])
                    or storage[name] not in ("file", "bytes")):
                raise ValueError("malformed artifact entry")
            count += 1
            if storage[name] == "bytes":
                inline += entry["bytes"]
    if count > MAX_ARTIFACTS or inline > MAX_INLINE:
        raise ValueError("too many artifacts or excessive inline data; use file artifacts")


def _layout(meta):
    """One payload per content digest/length; identity metadata is unchanged."""
    unique, seen, references = [], set(), {}
    for role in ("agent", "tap"):
        for name, entry in sorted(meta["components"][role]["artifacts"].items()):
            key = entry["bytes"], entry["sha256"]
            references[role, name] = key
            if key not in seen:
                seen.add(key)
                unique.append((key, role, name))
    return unique, references


class BundlePlan:
    """Provider-side snapshot description. Paths never enter the encrypted metadata.

    Source files must remain immutable while this plan is hashed and transmitted.
    Any content change is independently rejected by the receiver before loading.
    """

    def __init__(self, agent: ArtifactPackage, tap: ArtifactPackage, *, agent_drivers, tap_drivers):
        self._sources = {}
        records, storage = {}, {}
        for role, package, drivers in (("agent", agent, agent_drivers), ("tap", tap, tap_drivers)):
            driver = core._registry(drivers, role).get(package.driver)
            if driver is None:
                raise ValueError("unregistered provider driver")
            for name, _ in package.artifacts:
                _name(name)
            records[role], contents = core._component_record(role, package, driver)
            storage[role] = {}
            for name, value in contents.items():
                kind = "bytes" if isinstance(value, bytes) else "file"
                storage[role][name] = kind
                self._sources[role, name] = value
        meta = {"version": BUNDLE_VERSION, "components": {"version": core.COMPONENT_VERSION, **records}, "storage": storage}
        _validate_metadata(meta)
        self._encoded = canonical(meta)
        if len(self._encoded) > MAX_METADATA:
            raise ValueError("component metadata too large")
        self.identity = digest(self._encoded)
        unique, _ = _layout(meta)
        self.unique_artifact_bytes = sum(key[0] for key, role, name in unique)
        self.minimum_bytes = 8 + len(self._encoded) + self.unique_artifact_bytes

    def plaintext(self, padded_bytes: int):
        if type(padded_bytes) is not int or padded_bytes < self.minimum_bytes:
            raise ValueError("components do not fit the public envelope")
        meta = decode(self._encoded)
        yield len(self._encoded).to_bytes(8, "big")
        yield self._encoded
        unique, _ = _layout(meta)
        for key, role, name in unique:
            value = self._sources[role, name]
            if isinstance(value, bytes):
                yield value
            else:
                with Path(value).open("rb") as stream:
                    while block := stream.read(CHUNK):
                        yield block
        left = padded_bytes - self.minimum_bytes
        while left:
            take = min(CHUNK, left)
            yield bytes(take)
            left -= take


def _read_manifest(source, padded_bytes, agent_drivers, tap_drivers):
    length = int.from_bytes(read_exact(source, 8), "big")
    if not 0 < length <= min(MAX_METADATA, padded_bytes - 8):
        raise ValueError("invalid component metadata length")
    raw = read_exact(source, length)
    meta = decode(raw)
    _validate_metadata(meta)
    if canonical(meta) != raw:
        raise ValueError("noncanonical component manifest")
    unique, references = _layout(meta)
    total = 8 + length + sum(key[0] for key, role, name in unique)
    if total > padded_bytes:
        raise ValueError("artifacts exceed the public envelope")
    # Check selected code before disk allocation or any provider-controlled config is loaded.
    for role, drivers in (("agent", agent_drivers), ("tap", tap_drivers)):
        record = meta["components"][role]
        driver = core._registry(drivers, role).get(record["driver"])
        if driver is None:
            raise ValueError("unregistered component driver")
        expected_code = {n: core.sha256_hex(base64.b64decode(v)) for n, v in core._driver_artifacts(driver).items()}
        if record["code"] != expected_code:
            raise ValueError("component code differs from the registered workload")
    return raw, meta, unique, references, total


@contextmanager
def materialize(source, padded_bytes: int, *, scratch_root: Path, agent_drivers, tap_drivers, stage_recorder=None):
    """Authenticate the entire stream and private snapshot before any loader runs.

    scratch_root must be workload-private storage. Real confidential deployments
    must supply confidential storage (e.g. adequate protected tmpfs without swap).
    Ordinary local disk is only appropriate for the explicitly local demo.
    """
    recorder = stage_recorder or core.NullStageRecorder()
    with recorder.stage("stream_manifest_receive_validate"):
        raw, meta, unique, references, total = _read_manifest(source, padded_bytes, agent_drivers, tap_drivers)
    with tempfile.TemporaryDirectory(prefix="ct-private-", dir=scratch_root) as temp:
        blobs = {}
        with recorder.stage("stream_decrypt_write"):
            blob_root = Path(temp) / 'blobs'
            blob_root.mkdir(mode=0o700)
            for key, role, name in unique:
                path = blob_root / f'{key[0]}-{key[1]}'
                with path.open('xb') as target:
                    left = key[0]
                    while left:
                        block = read_exact(source, min(CHUNK, left))
                        target.write(block)
                        left -= len(block)
                path.chmod(0o400)
                blobs[key] = path
            remaining = padded_bytes - total
            while remaining:
                block = read_exact(source, min(CHUNK, remaining))
                if any(block):
                    raise ValueError("nonzero component padding")
                remaining -= len(block)
            if source.read(1):
                raise ValueError("trailing component plaintext")
        # Final crypto authentication precedes content verification and loading.
        # Hash every unique blob once; keep the two core preparation passes too.
        with recorder.stage("artifact_verify"):
            for key, path in blobs.items():
                if core._file_digest(path) != key[1]:
                    raise ValueError("artifact content digest mismatch")
        packages = {}
        with recorder.stage("private_snapshot_layout"):
            for role in ("agent", "tap"):
                record, pairs = meta["components"][role], []
                files = {name: entry for name, entry in record["artifacts"].items()
                         if meta["storage"][role][name] == 'file'}
                # Identical full file trees use the same physical paths, enabling
                # existing path-keyed model caches to share the same checkpoint.
                tree_root = Path(temp) / 'trees' / digest(canonical(files))
                for name, entry in sorted(record["artifacts"].items()):
                    blob = blobs[references[role, name]]
                    if meta["storage"][role][name] == 'bytes':
                        value = blob.read_bytes()  # total inline size is bounded
                    else:
                        path = tree_root / name
                        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        if not path.exists():
                            os.link(blob, path)
                        elif not path.samefile(blob):
                            raise ValueError("private snapshot path collision")
                        value = str(path)
                    pairs.append((name, value))
                packages[role] = ArtifactPackage(record["driver"], record["config"], tuple(pairs))
        kwargs = {}
        if 'stage_recorder' in inspect.signature(core.prepare_components).parameters:
            kwargs['stage_recorder'] = recorder
        with recorder.stage("component_prepare_total"):
            prepared = core.prepare_components(packages["agent"], packages["tap"],
                                                tuple(agent_drivers), tuple(tap_drivers), **kwargs)
        if prepared.encoded != canonical(meta["components"]):
            raise ValueError("loaded component record differs from the provider manifest")
        yield prepared, digest(raw)

