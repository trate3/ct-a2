"""Minimal per-sample Apptainer provider for official Inspect SWE-bench tasks.

This is an Inspect adapter, separate from ``sandbox.ApptainerSandbox`` (the
project's own LiveCodeBench runner backend). Each SWE task supplies a Docker
image reference as its sandbox config. We cache an Apptainer SIF and writable
rootfs per image, then give every sample a private copy of that rootfs.

Import this module before constructing/running an Inspect task, or install the
``certified_taps`` Inspect entry point from this package.
"""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import time
from typing import Any

from inspect_ai.util import (
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentLimits,
    SandboxUnavailableError,
    sandboxenv,
)
try:
    from inspect_ai.util import SandboxUserUnsupportedError
except ImportError:  # older supported Inspect versions
    class SandboxUserUnsupportedError(RuntimeError):
        pass
from inspect_ai.util._subprocess import ExecResult


def _binary() -> str:
    return os.environ.get("CT_SWEBENCH_APPTAINER", "apptainer")


def _root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CT_SWEBENCH_APPTAINER_ROOT", "/scratch/$USER/swebench-apptainer").replace("$USER", os.environ.get("USER", "unknown"))).expanduser()


def _image_key(image: str) -> str:
    return hashlib.sha256(image.encode()).hexdigest()[:20]


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sif_sha256(instance_id: str) -> str | None:
    mapping = os.environ.get("CT_SWEBENCH_APPTAINER_SIF_SHA256S")
    if mapping:
        return json.loads(mapping).get(instance_id)
    return os.environ.get("CT_SWEBENCH_APPTAINER_SIF_SHA256")


def _run(args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, timeout=timeout, check=False)


def _ensure_image(image: str, cached_sif: pathlib.Path | None = None) -> pathlib.Path:
    """Pull a remote OCI image and unpack its SIF into a writable rootfs once."""
    root = _root()
    key = _image_key(image)
    image_dir = root / "images" / key
    sif = image_dir / "image.sif"
    base = image_dir / "rootfs"
    manifest = image_dir / "manifest.txt"
    image_dir.mkdir(parents=True, exist_ok=True)
    with (image_dir / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if base.is_dir() and manifest.exists() and sif.exists():
            record = json.loads(manifest.read_text())
            actual = _file_sha256(sif)
            if record.get("image") == image and record.get("sif_sha256") == actual:
                if cached_sif is not None:
                    expected = _expected_sif_sha256(cached_sif.stem)
                    if expected and actual != expected:
                        raise RuntimeError(f"cached SIF hash mismatch for {sif}: {actual} != {expected}")
                return base
            raise RuntimeError(f"cached SWE image does not match its manifest: {image_dir}")
        if not sif.exists():
            if cached_sif is not None and cached_sif.is_file():
                instance_id = cached_sif.stem
                expected = _expected_sif_sha256(instance_id)
                if not expected:
                    raise ValueError(f"set CT_SWEBENCH_APPTAINER_SIF_SHA256S for reused SIF {cached_sif}")
                actual = _file_sha256(cached_sif)
                if actual != expected:
                    raise RuntimeError(f"cached SIF hash mismatch for {cached_sif}: {actual} != {expected}")
                shutil.copyfile(cached_sif, sif)
            else:
                tmp_sif = image_dir / f"image.sif.tmp.{os.getpid()}.{time.time_ns()}"
                result = _run([_binary(), "pull", str(tmp_sif), f"docker://{image}"], timeout=7200)
                if result.returncode:
                    raise RuntimeError(f"Apptainer pull failed for {image}: {result.stderr.decode(errors='replace')[-4000:]}")
                tmp_sif.replace(sif)
        if not base.is_dir():
            tmp_base = image_dir / f"rootfs.tmp.{os.getpid()}.{time.time_ns()}"
            result = _run([_binary(), "build", "--sandbox", str(tmp_base), str(sif)], timeout=7200)
            if result.returncode:
                raise RuntimeError(f"Apptainer sandbox build failed for {image}: {result.stderr.decode(errors='replace')[-4000:]}")
            tmp_base.replace(base)
        manifest.write_text(json.dumps({"image": image, "sif_sha256": _file_sha256(sif)}, sort_keys=True))
    return base


LOCALHOST_LINES = "127.0.0.1 localhost\n::1 localhost\n"


def ensure_localhost(rootfs: pathlib.Path) -> None:
    """Make `localhost` resolve inside the offline container.

    Docker writes /etc/hosts at run time, so images ship it empty; with --containall and no network, Apptainer
    does not fill it either, and tools that bind "localhost" (e.g. pytest-rerunfailures in the matplotlib
    images) fail before any test runs. Existing entries are kept.
    """
    hosts = rootfs / "etc" / "hosts"
    current = "" if hosts.is_symlink() or not hosts.is_file() else hosts.read_text(errors="replace")
    if any("localhost" in line.split()[1:] for line in current.splitlines() if not line.lstrip().startswith("#")):
        return
    if hosts.is_symlink():
        hosts.unlink()
    hosts.parent.mkdir(parents=True, exist_ok=True)
    hosts.write_text(current + ("" if not current or current.endswith("\n") else "\n") + LOCALHOST_LINES)


@sandboxenv("ct_swe_apptainer")
class SweApptainerEnvironment(SandboxEnvironment):
    """Inspect sandbox for SWE images; one writable rootfs copy per sample."""

    def __init__(self, rootfs: pathlib.Path, workdir: pathlib.Path, image: str) -> None:
        super().__init__()
        self.rootfs = rootfs
        self.workdir = workdir
        self.image = image
        self.workdir.mkdir(parents=True, exist_ok=True)

    @classmethod
    async def task_init(cls, task_name: str, config: Any) -> None:
        if not isinstance(config, str) or not config.startswith("swebench/"):
            raise ValueError("ct_swe_apptainer sandbox config must be a SWE Docker image reference")

    @classmethod
    async def sample_init(cls, task_name: str, config: Any, metadata: dict[str, Any]) -> dict[str, "SweApptainerEnvironment"]:
        if not isinstance(config, str):
            raise ValueError("ct_swe_apptainer requires a per-sample image reference")
        sif_dir = os.environ.get("CT_SWEBENCH_APPTAINER_SIF_DIR")
        instance_id = str(metadata.get("instance_id", ""))
        cached_sif = pathlib.Path(sif_dir, f"{instance_id}.sif") if sif_dir and instance_id else None
        base = await asyncio.to_thread(_ensure_image, config, cached_sif)
        sample_root = _root() / "samples" / f"{os.getpid()}-{time.time_ns()}-{_image_key(config)}"
        rootfs = sample_root / "rootfs"
        workdir = sample_root / "work"
        sample_root.mkdir(parents=True)
        workdir.mkdir()
        try:
            cp = await asyncio.to_thread(
                subprocess.run,
                ["cp", "-a", "--reflink=auto", str(base), str(rootfs)],
                capture_output=True,
                check=False,
            )
            if cp.returncode:
                raise RuntimeError(f"copying SWE rootfs failed: {cp.stderr.decode(errors='replace')}")
        except BaseException:
            shutil.rmtree(sample_root, ignore_errors=True)
            raise
        await asyncio.to_thread(ensure_localhost, rootfs)
        env = cls(rootfs, workdir, config)
        env._sample_root = sample_root
        base_commit = metadata.get("base_commit")
        if not base_commit:
            await asyncio.to_thread(shutil.rmtree, sample_root, True)
            raise ValueError("SWE sample metadata must include base_commit")
        # Fail before the solver applies its impossible test patch if the
        # prebuilt image does not contain the dataset's exact base revision.
        present = await env.exec(["git", "cat-file", "-e", f"{base_commit}^{{commit}}"], cwd="/testbed")
        if not present.success:
            await asyncio.to_thread(shutil.rmtree, sample_root, True)
            raise RuntimeError(f"SWE image {config} lacks required base commit {base_commit}: {present.stderr[-2000:]}")
        reset = await env.exec(["git", "reset", "--hard", base_commit], cwd="/testbed")
        if not reset.success:
            await asyncio.to_thread(shutil.rmtree, sample_root, True)
            raise RuntimeError(f"could not check out SWE base commit {base_commit}: {reset.stderr[-2000:]}")
        test_patch = metadata.get("test_patch")
        if test_patch:
            await env.write_file("/tmp/ct-swe-test-patch.diff", str(test_patch))
            patch_check = await env.exec(["git", "apply", "--check", "/tmp/ct-swe-test-patch.diff"], cwd="/testbed")
            await env.exec(["rm", "-f", "/tmp/ct-swe-test-patch.diff"], cwd="/testbed")
            if not patch_check.success:
                await asyncio.to_thread(shutil.rmtree, sample_root, True)
                raise RuntimeError(
                    f"test_patch does not apply at pinned base {base_commit}; refusing a potentially invalid scorer label: {patch_check.stderr[-2000:]}"
                )
        return {"default": env}

    @classmethod
    async def sample_cleanup(cls, task_name: str, config: Any, environments: dict[str, SandboxEnvironment], interrupted: bool) -> None:
        for env in environments.values():
            sample_root = getattr(env, "_sample_root", None)
            if sample_root is not None:
                await asyncio.to_thread(shutil.rmtree, sample_root, True)

    @classmethod
    def default_concurrency(cls) -> int | None:
        # Rootfs copies are large; keep the initial default conservative.
        return int(os.environ.get("CT_SWEBENCH_APPTAINER_CONCURRENCY", "1"))

    async def exec(self, cmd: list[str], input: str | bytes | None = None, cwd: str | None = None,
                   env: dict[str, str] | None = None, user: str | None = None,
                   timeout: int | None = None, timeout_retry: bool = True,
                   concurrency: bool = True) -> ExecResult[str]:
        if user is not None and not (user in ("root", "0") and os.environ.get("CT_SWEBENCH_FAKEROOT") == "1"):
            raise SandboxUserUnsupportedError("ct_swe_apptainer runs as the invoking user; enable supported fakeroot to request root")
        workdir = cwd or "/testbed"
        args = [_binary(), "exec", "--cleanenv", "--containall", "--net", "--network", "none",
                "--no-home", "--no-mount", "bind-paths,cwd", "--writable", "--pwd", workdir,
                "--workdir", str(self.workdir)]
        if os.environ.get("CT_SWEBENCH_FAKEROOT") == "1":
            args.append("--fakeroot")
        command_env = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory",
                       "GIT_CONFIG_VALUE_0": "/testbed", **(env or {})}
        for key, value in command_env.items():
            args.extend(["--env", f"{key}={value}"])
        args.extend([str(self.rootfs), *cmd])
        timeout_s = timeout
        try:
            result = await asyncio.to_thread(subprocess.run, args, input=input.encode() if isinstance(input, str) else input,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s,
                                             check=False, cwd="/")
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"Command timed out after {timeout}s") from e
        except OSError as e:
            raise SandboxUnavailableError(f"Could not start Apptainer: {e}") from e
        stdout = result.stdout.decode("utf-8", errors="strict")
        stderr = result.stderr.decode("utf-8", errors="strict")
        limit = SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE
        if len(stdout.encode()) > limit:
            stdout = stdout.encode()[-limit:].decode("utf-8", errors="replace")
        if len(stderr.encode()) > limit:
            stderr = stderr.encode()[-limit:].decode("utf-8", errors="replace")
        return ExecResult(result.returncode == 0, result.returncode, stdout, stderr)

    async def write_file(self, file: str, contents: str | bytes) -> None:
        data = contents.encode() if isinstance(contents, str) else contents
        encoded = base64.b64encode(data)
        path = shlex.quote(file)
        parent = shlex.quote(str(pathlib.PurePosixPath(file).parent))
        result = await self.exec(["bash", "-c", f"mkdir -p {parent} && base64 -d > {path}"], input=encoded)
        if not result.success:
            raise PermissionError(result.stderr or f"could not write {file}")

    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        max_size = int(os.environ.get("INSPECT_SANDBOX_MAX_READ_FILE_SIZE", 100 * 1024**2))
        path = shlex.quote(file)
        stat_result = await self.exec(["bash", "-c", f"wc -c < {path}"])
        if not stat_result.success:
            if "No such file" in stat_result.stderr:
                raise FileNotFoundError(file)
            raise PermissionError(stat_result.stderr or f"could not read {file}")
        size = int(stat_result.stdout.strip())
        if size > max_size:
            raise OutputLimitExceededError(f"file exceeds {max_size} byte read limit")
        chunk_size = 3 * 1024**2
        chunks = []
        for offset in range(0, size, chunk_size):
            result = await self.exec([
                "bash", "-c",
                f"tail -c +{offset + 1} {path} | head -c {chunk_size} | base64 | tr -d '\\n'",
            ])
            if not result.success:
                raise PermissionError(result.stderr or f"could not read {file}")
            chunks.append(base64.b64decode(result.stdout.strip()))
        data = b"".join(chunks)
        return data.decode("utf-8") if text else data
