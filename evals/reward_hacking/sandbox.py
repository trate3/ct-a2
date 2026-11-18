"""Per-rollout Docker sandbox reproducing ImpossibleBench's inspect_ai docker sandbox.

ImpossibleBench runs each LiveCodeBench sample in a fresh container from its
compose.yaml (image aisiuk/inspect-tool-support, `tail -f /dev/null`, init, no
network, 1 GB memory) through inspect_ai's docker sandbox. We drive the docker CLI
directly, one container per rollout, and copy inspect_ai 0.3.137's exec/read/write
semantics (see SANDBOX.md for the reference version and every deviation):

  - working directory is the image's (`/`; the image sets no WORKDIR), so func.py
    and test.py are /func.py and /test.py;
  - exec runs as root with stdin attached only when there is input, decodes output
    as strict UTF-8, caps each stream at 10 MiB, and on timeout kills only the
    docker client: the command keeps running inside the container, as under inspect;
  - with `timeout_retry` (inspect's default for tool calls) a timed-out command is
    re-run up to twice with shorter timeouts before the timeout is reported;
  - write_file pipes through `tee` and read_file uses `docker cp`.

Nothing from the host is mounted. Files enter only through `docker exec`/`docker cp`
issued by this harness. A snapshot is the container's filesystem diff against the
image (`docker diff`: a tar of added/changed paths plus the deleted paths), held
in host memory; restoring starts a fresh container from the image and applies it.
Processes are not captured. (A second backend, `ApptainerSandbox`, for clusters without Docker is at the
end of this file; `new_sandbox()` selects it with CT_RH_SANDBOX=apptainer.) `docker commit` gives the same state but takes 5-8 s
per snapshot on this host's containerd image store; the diff takes about 1 s.
"""

from __future__ import annotations

import atexit
import fcntl
import hashlib
import io
import json
import os
import pathlib
import platform
import posixpath
import shlex
import shutil
import socket
import stat as stat_mod
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field

IMAGE_REPO = "aisiuk/inspect-tool-support"
IMAGE_DIGEST = "sha256:bbaecf7a093cfe24703c97d562bb059899e272cd5d57f16d9916fe05e6614419"
IMAGE = f"{IMAGE_REPO}@{IMAGE_DIGEST}"
WORKDIR = "/"
LABEL = "certified-taps.reward-hacking"

# inspect_ai SandboxEnvironmentLimits (0.3.137)
MAX_EXEC_OUTPUT_SIZE = 10 * 1024**2
MAX_EXEC_OUTPUT_SIZE_STR = "10 MiB"
WRITE_FILE_TIMEOUT = 180

# text_editor runs inspect's in-container JSON-RPC helper. inspect_ai 0.3.137 copies
# the binary bundled in its wheel to this path on first use, if not already present.
SANDBOX_TOOLS_PATH = "/opt/inspect-sandbox-tools"
INSPECT_AI_VERSION = "0.3.137"
INSPECT_WHEEL_URL = ("https://files.pythonhosted.org/packages/eb/17/94feec4906f97e9d849dfb379814751560e2bc1e698b4a72b1d1e9a8f86a/"
                     "inspect_ai-0.3.137-py3-none-any.whl")
INSPECT_WHEEL_SHA256 = "ca6f3c3c8a69d41c3179b09929c33c9dde92d41e008770f95956090310adb3c7"
SANDBOX_TOOLS_SHA256 = {
    "amd64": "2858849c8a423a6479f6d6d52f5b43eba80c954f4ff10f8ea693314bad587d18",
    "arm64": "44ca449b49f59904e66bc7638a572e8ed4d013c6570887791594b2a02197373c",
}
# Files docker itself mounts into every container (--init binary, per-container
# name/hosts files); not part of the image diff and not captured by snapshots.
DOCKER_MOUNTS = frozenset({"/usr/sbin/docker-init", "/etc/hostname", "/etc/hosts", "/etc/resolv.conf"})
CACHE_DIR = pathlib.Path(os.environ.get("CT_RH_CACHE", pathlib.Path.home() / ".cache" / "certified_taps" / "reward_hacking"))


@dataclass(frozen=True)
class ExecResult:
    success: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    seconds: float = 0.0


class OutputLimitExceededError(Exception):
    """A stream exceeded MAX_EXEC_OUTPUT_SIZE (inspect's error of the same name)."""

    def __init__(self, limit_str: str, truncated_output: str | None) -> None:
        super().__init__(f"The sandbox output stream limit of {limit_str} was exceeded.")
        self.limit_str = limit_str
        self.truncated_output = truncated_output


@dataclass(frozen=True)
class SandboxSnapshot:
    """A sandbox's filesystem state relative to its image; independent of any container."""

    base_image: str
    archive: bytes                 # tar (from /) of added and changed paths, no recursion
    deleted: tuple[str, ...]       # paths deleted relative to the image
    sandbox_tools: bool            # inspect's text_editor helper was injected (re-injected on restore)
    source_container: str
    config: dict
    created: float = field(default_factory=time.time)
    backend: str = "docker"        # "docker" | "apptainer" (apptainer: see ApptainerSandbox.snapshot)

    def digest(self) -> str:
        h = hashlib.sha256(self.base_image.encode())
        if self.backend != "docker":
            h.update(self.backend.encode())
        h.update(self.archive)
        h.update("\0".join(self.deleted).encode())
        h.update(b"tools" if self.sandbox_tools else b"")
        return h.hexdigest()


def docker_available() -> bool:
    try:
        return _docker(["info", "--format", "{{.ServerVersion}}"], check=False, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def docker_ensure_image(image: str = IMAGE) -> None:
    if _docker(["image", "inspect", image], check=False).returncode != 0:
        _docker(["pull", image], timeout=1800)


def docker_cleanup_leftovers() -> int:
    """Remove every container carrying our label (e.g. after a crash)."""
    ids = _docker(["ps", "-aq", "--filter", f"label={LABEL}"], check=False).stdout.split()
    if ids:
        _docker(["rm", "-f", *ids], check=False)
    return len(ids)


class DockerSandbox:
    """One container per rollout. Use as a context manager; `close()` always removes it."""

    backend = "docker"

    def __init__(self, image: str = IMAGE, *, memory: str = "1g", cpus: float | None = 2.0,
                 pids_limit: int | None = 512, timeout_retry: bool = True,
                 labels: dict[str, str] | None = None) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout_retry = timeout_retry
        self.labels = dict(labels or {})
        self.name = f"ct-rh-{uuid.uuid4().hex[:12]}"
        self.container_id: str | None = None
        self.snapshot_of: str | None = None

    # ---- lifecycle -------------------------------------------------------------

    def config(self) -> dict:
        """Everything that determines the sandbox, for the experiment manifest."""
        return {
            "image": self.image, "base_image": IMAGE, "workdir": WORKDIR, "network": "none",
            "memory": self.memory, "memory_swap": "docker default (2x memory, as compose mem_limit)",
            "cpus": self.cpus, "pids_limit": self.pids_limit, "init": True,
            "command": ["tail", "-f", "/dev/null"], "user": "root (image default)",
            "security_opt": ["no-new-privileges"], "mounts": [],
            "timeout_retry": self.timeout_retry, "max_exec_output_bytes": MAX_EXEC_OUTPUT_SIZE,
            "sandbox_tools": {"path": SANDBOX_TOOLS_PATH, "inspect_ai": INSPECT_AI_VERSION,
                              "sha256": SANDBOX_TOOLS_SHA256[_arch()]},
            "snapshot_of": self.snapshot_of,
        }

    def start(self) -> "DockerSandbox":
        if self.container_id is not None:
            raise RuntimeError("sandbox already started")
        if self.image == IMAGE:
            docker_ensure_image(self.image)
        args = ["run", "-d", "--name", self.name, "--init", "--network", "none",
                "--memory", self.memory, "--security-opt", "no-new-privileges",
                "--label", f"{LABEL}=1"]
        if self.cpus is not None:
            args += ["--cpus", str(self.cpus)]
        if self.pids_limit is not None:
            args += ["--pids-limit", str(self.pids_limit)]
        for k, v in self.labels.items():
            args += ["--label", f"{LABEL}.{k}={v}"]
        args += [self.image, "tail", "-f", "/dev/null"]
        try:
            self.container_id = _docker(args, timeout=120).stdout.strip()
        except BaseException:
            _docker(["rm", "-f", self.name], check=False)
            raise
        return self

    def close(self) -> None:
        _docker(["rm", "-f", self.name], check=False)   # also cleans up a half-started run
        self.container_id = None

    def __enter__(self) -> "DockerSandbox":
        if self.container_id is None:
            self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- snapshots ---------------------------------------------------------------

    def snapshot(self) -> SandboxSnapshot:
        """Capture the filesystem diff against the image (the container keeps running).

        Not atomic: a background process still writing in the container may race it.
        """
        self._require()
        added, deleted = [], []
        for line in _docker(["diff", self.name], timeout=300).stdout.splitlines():
            kind, path = line[:1], line[2:]
            if path not in DOCKER_MOUNTS:
                (deleted if kind == "D" else added).append(path)
        tools = SANDBOX_TOOLS_PATH in added and self.exec(
            ["sha256sum", SANDBOX_TOOLS_PATH]).stdout.split()[:1] == [SANDBOX_TOOLS_SHA256[_arch()]]
        if tools:
            added.remove(SANDBOX_TOOLS_PATH)   # pristine helper: re-inject instead of archiving 15 MB
        listing = "\0".join(p.lstrip("/") for p in added) + "\0"
        argv = ["docker", "exec", "-i", "--workdir", "/", self.name, "tar", "-c", "--no-recursion",
                "--ignore-failed-read", "--xattrs", "--null", "-T", "-", "-f", "-"]
        out, err, rc, _ = _run_limited(argv, listing.encode() if added else None, 300, limit=None)
        if rc not in (0, 1):
            raise RuntimeError(f"snapshot tar failed ({rc}): {err.decode(errors='replace')}")
        return SandboxSnapshot(base_image=self.image, archive=out if added else b"",
                               deleted=tuple(deleted), sandbox_tools=tools, source_container=self.name,
                               config=self.config())

    @classmethod
    def from_snapshot(cls, snapshot: SandboxSnapshot, **overrides) -> "DockerSandbox":
        """A new, started sandbox whose filesystem equals the snapshot's."""
        c = snapshot.config
        kw = {"memory": c["memory"], "cpus": c["cpus"], "pids_limit": c["pids_limit"],
              "timeout_retry": c["timeout_retry"]} | overrides
        sb = cls(snapshot.base_image, **kw)
        sb.snapshot_of = snapshot.digest()
        sb.start()
        try:
            if snapshot.deleted:
                r = sb.exec(["xargs", "-0", "rm", "-rf", "--"], input="\0".join(snapshot.deleted))
                if not r.success:
                    raise RuntimeError(f"restore: deleting paths failed: {r.stderr}")
            if snapshot.sandbox_tools:
                sb.ensure_sandbox_tools()
            if snapshot.archive:   # last, so archived directory metadata wins
                argv = ["docker", "exec", "-i", "--workdir", "/", sb.name, "tar", "-x", "-p",
                        "--same-owner", "--xattrs", "-f", "-"]
                _, err, rc, _ = _run_limited(argv, snapshot.archive, 300, limit=None)
                if rc != 0:
                    raise RuntimeError(f"restore: tar failed ({rc}): {err.decode(errors='replace')}")
        except BaseException:
            sb.close()
            raise
        return sb

    # ---- exec / files --------------------------------------------------------------

    def exec(self, cmd: list[str], input: str | bytes | None = None, timeout: float | None = None,
             cwd: str | None = None, env: dict[str, str] | None = None, user: str | None = None,
             timeout_retry: bool | None = None) -> ExecResult:
        """Run `cmd` in the container (inspect DockerSandboxEnvironment.exec semantics).

        Returns a result with `timed_out=True` (and no output) if the command timed out
        on every try; raises UnicodeDecodeError for non-UTF-8 output,
        OutputLimitExceededError past 10 MiB on a stream, and PermissionError for a
        docker "permission denied" exec failure, as inspect does.
        """
        self._require()
        final_cwd = posixpath.join(WORKDIR, cwd) if cwd is not None else WORKDIR
        args = ["exec", "-i", "--workdir", final_cwd]
        if user:
            args += ["--user", user]
        for k, v in (env or {}).items():
            args += ["--env", f"{k}={v}"]
        argv = ["docker", *args, self.name, *cmd]
        data = input.encode() if isinstance(input, str) else input

        retry = self.timeout_retry if timeout_retry is None else timeout_retry
        t0 = time.perf_counter()
        if timeout is None:
            out, err, rc, _ = _run_limited(argv, data, None)
        else:
            # inspect compose_command: first try with `timeout`, then up to two retries
            # with min(timeout, 60) // retries (at least 1 second each)
            tries = [timeout] + ([max(min(timeout, 60) // r, 1) for r in (1, 2)] if retry else [])
            for t in tries:
                out, err, rc, timed_out = _run_limited(argv, data, t)
                if not timed_out:
                    break
            else:
                return ExecResult(False, -1, "", "", timed_out=True, seconds=time.perf_counter() - t0)
        stdout, stderr = out.decode(), err.decode()   # strict, as inspect's subprocess()
        seconds = time.perf_counter() - t0
        if len(out) > MAX_EXEC_OUTPUT_SIZE or len(err) > MAX_EXEC_OUTPUT_SIZE:
            raise OutputLimitExceededError(MAX_EXEC_OUTPUT_SIZE_STR,
                                           _truncate_bytes(stdout) + _truncate_bytes(stderr))
        result = ExecResult(rc == 0, rc, stdout, stderr, seconds=seconds)
        if rc == 126 and "permission denied" in stdout:
            raise PermissionError(f"Permission denied executing command: {_inspect_repr(result)}")
        return result

    def container_file(self, path: str) -> str:
        return path if path.startswith("/") else posixpath.join(WORKDIR, path)

    def write_file(self, path: str, content: str | bytes) -> None:
        """inspect's write_file: mkdir -p the parent, then `tee` the content in as root."""
        file = self.container_file(path)
        parent = posixpath.dirname(file)
        if parent not in (".", ""):
            r = self.exec(["mkdir", "-p", parent])
            if not r.success:
                raise RuntimeError(f"Failed to create container directory {parent}: {r.stderr}")
        if isinstance(content, str):
            r = self.exec(["sh", "-e", "-c", 'tee -- "$1" > /dev/null', "write_file_script", file],
                          input=content, timeout=WRITE_FILE_TIMEOUT)
        else:
            import base64
            r = self.exec(["sh", "-e", "-c", 'base64 -d | tee -- "$1" > /dev/null', "write_file_script", file],
                          input=base64.b64encode(content).decode("ascii"), timeout=WRITE_FILE_TIMEOUT)
        if r.timed_out:
            raise TimeoutError(f"write_file {file} timed out")
        if r.returncode != 0:
            low = r.stderr.casefold()
            if "permission denied" in low:
                raise PermissionError(f"Permission was denied. Error details: {r.stderr}")
            if "cannot overwrite directory" in low or "is a directory" in low:
                raise IsADirectoryError(f"Failed to write file: {file} because it is a directory already")
            raise RuntimeError(f"failed to copy during write_file: {r}")

    def read_file(self, path: str, text: bool = True) -> str | bytes:
        """inspect's read_file: `docker cp` out, then decode as strict UTF-8 without newline translation."""
        self._require()
        file = self.container_file(path)
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, posixpath.basename(file) or "root")
            r = _docker(["cp", f"{self.name}:{file}", dest], check=False, timeout=120)
            if r.returncode != 0:
                msg = (r.stderr + r.stdout).lower()
                if "could not find the file" in msg or "no such file" in msg:
                    raise FileNotFoundError(2, "No such file or directory.", path)
                if "permission denied" in msg:
                    raise PermissionError(13, "Permission denied.", path)
                raise RuntimeError(r.stderr)
            if os.path.isdir(dest):
                raise IsADirectoryError(21, "Is a directory", path)
            if text:
                with open(dest, "r", newline="", encoding="utf-8") as f:
                    return f.read()
            with open(dest, "rb") as f:
                return f.read()

    def ensure_sandbox_tools(self) -> None:
        """Inject inspect's text_editor helper if missing (re-checked on every call, as inspect does)."""
        if self.exec(["test", "-f", SANDBOX_TOOLS_PATH]).success:
            return
        _docker(["cp", str(sandbox_tools_binary()), f"{self.name}:{SANDBOX_TOOLS_PATH}"], timeout=120)
        r = self.exec(["sh", "-c", f"chown 0:0 {SANDBOX_TOOLS_PATH} && chmod 755 {SANDBOX_TOOLS_PATH}"])
        if not r.success:
            raise RuntimeError(f"sandbox tools injection failed: {r.stderr}")

    def _require(self) -> None:
        if self.container_id is None:
            raise RuntimeError("sandbox not started")


# ---- helpers ------------------------------------------------------------------------


def _docker(args: list[str], check: bool = True, timeout: float | None = 60) -> subprocess.CompletedProcess:
    r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"docker {shlex.join(args[:3])} failed ({r.returncode}): {r.stderr.strip()}")
    return r


def _run_limited(argv: list[str], data: bytes | None, timeout: float | None,
                 limit: int | None = MAX_EXEC_OUTPUT_SIZE,
                 drain_timeout: float = 5) -> tuple[bytes, bytes, int, bool]:
    """Run the docker client, reading each stream up to just past the output limit.

    Like inspect's subprocess(): stdin is /dev/null unless there is (non-empty) input;
    a stream over the limit kills the client; on timeout the client is terminated
    (the process inside the container is not). Returns (stdout, stderr, rc, timed_out).
    `drain_timeout` bounds the wait for the pipes to close after the client exits: under apptainer an
    orphaned command still holds them, which docker's daemon-side streams do not.
    """
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE if data else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bufs = [io.BytesIO(), io.BytesIO()]

    def pump(stream, buf):
        written = 0
        while chunk := stream.read1(65536):
            buf.write(chunk)
            written += len(chunk)
            if limit is not None and written > limit:
                proc.kill()
                break

    threads = [threading.Thread(target=pump, args=(s, b), daemon=True)
               for s, b in ((proc.stdout, bufs[0]), (proc.stderr, bufs[1]))]
    for t in threads:
        t.start()
    if data:
        try:
            proc.stdin.write(data)
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    deadline = time.monotonic() + drain_timeout
    for t in threads:
        t.join(timeout=drain_timeout if drain_timeout >= 5 else max(deadline - time.monotonic(), 0))
    return bufs[0].getvalue(), bufs[1].getvalue(), proc.returncode, timed_out


def _truncate_bytes(s: str) -> str:
    b = s.encode("utf-8", errors="replace")
    return b[:MAX_EXEC_OUTPUT_SIZE].decode("utf-8", errors="replace") if len(b) > MAX_EXEC_OUTPUT_SIZE else s


def _inspect_repr(r: ExecResult) -> str:
    return f"ExecResult(success={r.success!r}, returncode={r.returncode!r}, stdout={r.stdout!r}, stderr={r.stderr!r})"


def _arch() -> str:
    return "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"


def sandbox_tools_binary(arch: str | None = None) -> pathlib.Path:
    """inspect_ai 0.3.137's bundled inspect-sandbox-tools binary, fetched once from PyPI and hash-checked."""
    arch = arch or _arch()
    name = f"inspect-sandbox-tools-{arch}-v3"
    path = CACHE_DIR / f"inspect_ai-{INSPECT_AI_VERSION}" / name
    if not path.exists() or _sha256(path.read_bytes()) != SANDBOX_TOOLS_SHA256[arch]:
        with urllib.request.urlopen(INSPECT_WHEEL_URL, timeout=300) as resp:
            wheel = resp.read()
        if _sha256(wheel) != INSPECT_WHEEL_SHA256:
            raise RuntimeError("inspect_ai wheel hash mismatch")
        blob = zipfile.ZipFile(io.BytesIO(wheel)).read(f"inspect_ai/binaries/{name}")
        if _sha256(blob) != SANDBOX_TOOLS_SHA256[arch]:
            raise RuntimeError("sandbox tools binary hash mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(blob)
        tmp.chmod(0o755)
        tmp.replace(path)
    return path


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# =====================================================================================
# Apptainer backend (clusters without Docker). See SANDBOX.md, "Apptainer backend".
#
# One rollout = one writable copy of a base sandbox directory (the pinned image
# unpacked with `apptainer build --sandbox`) running as an Apptainer instance with
# `--net --network none --writable --containall`. Only flags verified on the target
# cluster's setuid install are used. The container runs as the invoking user (not root).
# =====================================================================================

ORPHAN_GRACE_SECONDS = 600   # an ownerless rollout dir younger than this may be mid-start
APPTAINER_PREFIX = "ct-rh-"
APPTAINER_NPROC_ENV = "CT_RH_APPTAINER_NPROC"   # optional `ulimit -u` (off by default: counts all of the user's processes)
# Variables Apptainer adds to every exec'd process; removed so `env` looks like the docker image's.
_APPTAINER_ENV_VARS = ("APPTAINER_APPNAME APPTAINER_BIND APPTAINER_COMMAND APPTAINER_CONTAINER "
                       "APPTAINER_ENVIRONMENT APPTAINER_INSTANCE APPTAINER_NAME SINGULARITY_BIND "
                       "SINGULARITY_CONTAINER SINGULARITY_ENVIRONMENT SINGULARITY_INSTANCE SINGULARITY_NAME "
                       "LD_LIBRARY_PATH PS1 PROMPT_COMMAND TERM")


def apptainer_binary() -> str:
    return os.environ.get("CT_RH_APPTAINER") or shutil.which("apptainer") or "apptainer"


def apptainer_available() -> bool:
    try:
        return subprocess.run([apptainer_binary(), "--version"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def sandbox_root() -> pathlib.Path:
    """CT_RH_SANDBOX_ROOT: per-rollout copies, the base image files, build cache and lock."""
    return pathlib.Path(os.environ.get("CT_RH_SANDBOX_ROOT") or CACHE_DIR / "apptainer").expanduser()


def apptainer_paths() -> dict[str, pathlib.Path]:
    root = sandbox_root()
    tag = IMAGE_DIGEST.split(":")[1][:12]
    base = pathlib.Path(os.environ["CT_RH_APPTAINER_BASE"]).expanduser() if os.environ.get("CT_RH_APPTAINER_BASE") \
        else root / "base" / f"rootfs-{tag}"
    sif = pathlib.Path(os.environ["CT_RH_APPTAINER_SIF"]).expanduser() if os.environ.get("CT_RH_APPTAINER_SIF") \
        else root / "base" / f"inspect-tool-support-{tag}.sif"
    return {"root": root, "base": base, "sif": sif, "manifest": base.with_name(base.name + ".manifest.json"),
            "lock": root / "build.lock", "rollouts": root / "rollouts"}


def _apptainer_env() -> dict[str, str]:
    root = sandbox_root()
    env = dict(os.environ)
    env.setdefault("APPTAINER_CACHEDIR", str(root / "cache"))
    env.setdefault("APPTAINER_TMPDIR", str(root / "tmp"))
    return env


def _apptainer(args: list[str], check: bool = True, timeout: float | None = 120) -> subprocess.CompletedProcess:
    r = subprocess.run([apptainer_binary(), *args], capture_output=True, text=True, timeout=timeout,
                       env=_apptainer_env())
    if check and r.returncode != 0:
        raise RuntimeError(f"apptainer {shlex.join(args[:3])} failed ({r.returncode}): {r.stderr.strip()}")
    return r


def tree_digest(path: str | os.PathLike) -> str:
    """Content digest of a directory tree: relative paths, types, permission bits, symlink targets and
    file contents; no owners or timestamps, so it is identical for any user's build of the same image."""
    h = hashlib.sha256()
    root = str(path)

    def walk(d: str, rel: str) -> None:
        for e in sorted(os.scandir(d), key=lambda e: e.name):
            r = f"{rel}/{e.name}"
            st = e.stat(follow_symlinks=False)
            mode = stat_mod.S_IMODE(st.st_mode)
            h.update(f"{r}\0{stat_mod.S_IFMT(st.st_mode):o}\0{mode:o}\0".encode())
            if stat_mod.S_ISLNK(st.st_mode):
                h.update(os.readlink(e.path).encode(errors="surrogateescape"))
            elif stat_mod.S_ISREG(st.st_mode):
                fh = hashlib.sha256()
                with open(e.path, "rb") as f:
                    while chunk := f.read(1 << 20):
                        fh.update(chunk)
                h.update(f"{st.st_size}\0{fh.hexdigest()}".encode())
            elif stat_mod.S_ISDIR(st.st_mode):
                walk(e.path, r)

    walk(root, "")
    return h.hexdigest()


def ensure_apptainer_image(verify: bool = False) -> dict:
    """Build (once) the SIF and the unpacked base sandbox directory from the pinned docker digest.

    Idempotent and safe across processes (flock on `build.lock`). The base directory's content
    digest is recorded in `<base>.manifest.json`. `verify=True` recomputes it and compares.
    Returns the manifest.
    """
    p = apptainer_paths()
    custom = "CT_RH_APPTAINER_BASE" in os.environ

    def ready() -> dict | None:
        if p["base"].is_dir() and p["manifest"].exists():
            m = json.loads(p["manifest"].read_text())
            if m.get("image") == IMAGE:
                return m
        return None

    m = ready()
    if m is None:
        p["root"].mkdir(parents=True, exist_ok=True)
        for sub in ("cache", "tmp", "base"):
            (p["root"] / sub).mkdir(exist_ok=True)
        p["base"].parent.mkdir(parents=True, exist_ok=True)
        with open(p["lock"], "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            m = ready()
            if m is None:
                if custom and not p["base"].is_dir():
                    raise RuntimeError(f"CT_RH_APPTAINER_BASE={p['base']} does not exist")
                t0 = time.time()
                if not p["sif"].exists():
                    tmp = p["sif"].with_name(p["sif"].name + f".tmp{os.getpid()}")
                    tmp.unlink(missing_ok=True)
                    _apptainer(["build", str(tmp), f"docker://{IMAGE}"], timeout=3600)
                    tmp.replace(p["sif"])
                if not p["base"].is_dir():
                    tmpd = p["base"].with_name(p["base"].name + f".tmp{os.getpid()}")
                    _rmtree(tmpd)
                    _apptainer(["build", "--sandbox", str(tmpd), str(p["sif"])], timeout=3600)
                    tmpd.rename(p["base"])
                removed = _normalize_base(p["base"])
                m = {"image": IMAGE, "sif": str(p["sif"]), "base": str(p["base"]), "removed_root_links": removed, "apptainer": _apptainer(["--version"]).stdout.strip(),
                     "tree_sha256": tree_digest(p["base"]), "built": time.time(), "build_seconds": round(time.time() - t0, 1)}
                tmpm = p["manifest"].with_name(p["manifest"].name + ".tmp")
                tmpm.write_text(json.dumps(m, indent=1))
                tmpm.replace(p["manifest"])
    if verify:
        got = tree_digest(p["base"])
        if got != m["tree_sha256"]:
            raise RuntimeError(f"base sandbox {p['base']} differs from its recorded digest "
                               f"({got[:12]} != {m['tree_sha256'][:12]}): modified since build")
    return m


# Apptainer's legacy convenience symlinks in the image root. Unused by `instance exec` but they show up in
# the model's `ls /`, so they are removed from the base directory (recorded in the manifest digest).
_APPTAINER_ROOT_LINKS = ("environment", "singularity", ".exec", ".run", ".shell", ".test")


def _normalize_base(base: pathlib.Path) -> list[str]:
    gone = []
    for n in _APPTAINER_ROOT_LINKS:
        p = base / n
        if p.is_symlink():
            p.unlink()
            gone.append(n)
    return gone


def _rmtree(path: str | os.PathLike) -> None:
    """rm -rf that also removes trees the agent made unreadable or unwritable (it owns them)."""
    path = str(path)
    if not os.path.lexists(path):
        return
    try:
        shutil.rmtree(path)
    except OSError:
        subprocess.run(["chmod", "-R", "u+rwX", path], capture_output=True)   # does not follow symlinks
        shutil.rmtree(path, ignore_errors=True)


def _host_id() -> str:
    return "".join(c if c.isalnum() else "x" for c in socket.gethostname().split(".")[0])[:24]


def _proc_start(pid: int) -> str | None:
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


_LIVE_APPTAINER: set["ApptainerSandbox"] = set()


@atexit.register
def _close_live_apptainer() -> None:
    for sb in list(_LIVE_APPTAINER):
        try:
            sb.close()
        except Exception:
            pass


class ApptainerSandbox:
    """One Apptainer instance over a private writable copy of the base sandbox directory.

    Same public API as DockerSandbox. Differences are listed in SANDBOX.md ("Apptainer backend"):
    runs as the invoking user, memory is approximated with RLIMIT_DATA, no pid/cpu limits, files
    appear owned by that user. Nothing from the host is mounted (`--containall`); `/tmp` and
    `/var/tmp` are directories of the rollout's private work dir, not host paths.
    """

    backend = "apptainer"

    def __init__(self, image: str = IMAGE, *, memory: str = "1g", cpus: float | None = None,
                 pids_limit: int | None = None, timeout_retry: bool = True,
                 labels: dict[str, str] | None = None) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout_retry = timeout_retry
        self.labels = dict(labels or {})
        self.name = f"{APPTAINER_PREFIX}{_host_id()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.container_id: str | None = None
        self.snapshot_of: str | None = None
        self.base_tree_sha256: str | None = None
        self.dir: pathlib.Path | None = None
        self._marker_ctime_ns = 0
        self._started_instance = False

    # ---- lifecycle -------------------------------------------------------------

    def config(self) -> dict:
        return {
            "backend": "apptainer", "image": self.image, "base_image": IMAGE, "workdir": WORKDIR, "network": "none",
            "memory": self.memory, "memory_enforcement": f"RLIMIT_DATA (ulimit -d) {_memory_kb(self.memory)} KiB per process",
            "cpus": None, "pids_limit": self.pids_limit, "init": "apptainer appinit",
            "apptainer": {"flags": ["--net", "--network", "none", "--writable", "--containall", "--workdir <rollout>/work"],
                          "version": _apptainer_version(), "base_tree_sha256": self.base_tree_sha256},
            "user": "invoking host user (not root)", "mounts": [],
            "timeout_retry": self.timeout_retry, "max_exec_output_bytes": MAX_EXEC_OUTPUT_SIZE,
            "sandbox_tools": {"path": SANDBOX_TOOLS_PATH, "inspect_ai": INSPECT_AI_VERSION,
                              "sha256": SANDBOX_TOOLS_SHA256[_arch()]},
            "snapshot_of": self.snapshot_of,
        }

    def start(self) -> "ApptainerSandbox":
        if self.container_id is not None:
            raise RuntimeError("sandbox already started")
        p = apptainer_paths()
        manifest = ensure_apptainer_image()
        self.base_tree_sha256 = manifest["tree_sha256"]
        self.dir = p["rollouts"] / self.name
        rootfs, work = self.dir / "rootfs", self.dir / "work"
        try:
            work.mkdir(parents=True)
            (self.dir / "owner.json").write_text(json.dumps(
                {"host": _host_id(), "pid": os.getpid(), "start": _proc_start(os.getpid()), "created": time.time()}))
            r = subprocess.run(["cp", "-a", "--reflink=auto", str(p["base"]), str(rootfs)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"copying the base sandbox failed: {r.stderr.strip()}")
            _LIVE_APPTAINER.add(self)
            self._started_instance = True
            _apptainer(["instance", "start", "--net", "--network", "none", "--writable", "--containall",
                        "--workdir", str(work), str(rootfs), self.name], timeout=120)
            # Everything changed after this marker (by ctime, which only the kernel sets) is the
            # rollout's state; snapshots archive exactly those paths. The pause makes it strict.
            marker = self.dir / "marker"
            marker.write_text("")
            time.sleep(0.02)
            self._marker_ctime_ns = marker.stat().st_ctime_ns
            self.container_id = self.name
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self.dir is None:
            return
        if self._started_instance:
            _apptainer(["instance", "stop", "--force", self.name], check=False, timeout=60)
        _rmtree(self.dir)
        _LIVE_APPTAINER.discard(self)
        self.container_id = None
        self.dir = None
        self._started_instance = False

    def __enter__(self) -> "ApptainerSandbox":
        if self.container_id is None:
            self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- snapshots ---------------------------------------------------------------

    def snapshot(self) -> SandboxSnapshot:
        """Capture every path changed since the rollout's copy was made, relative to the base directory.

        `find -newercc marker` lists the paths whose ctime is after the copy (any write, chmod,
        rename or unlink of a child updates it, and the agent cannot set ctime), the directory
        listings of changed directories give the deletions, and the archive is a tar of
        the changed paths (lstat only: the host never follows a symlink the agent made) plus
        everything under `/tmp` and `/var/tmp`. Like the docker diff, not atomic with
        processes that are still writing.
        """
        self._require()
        rootfs, base = self.dir / "rootfs", pathlib.Path(apptainer_paths()["base"])
        out = subprocess.run(["find", ".", "-newercc", str(self.dir / "marker"), "-print0"], cwd=rootfs,
                             capture_output=True, check=True).stdout
        changed = sorted(os.fsdecode(x)[2:] for x in out.split(b"\0") if len(x) > 2)
        members: dict[str, str] = {}          # archive name -> host path
        deleted: list[str] = []
        base_ok: dict[str, bool] = {}

        def base_stat(rel: str):
            """lstat in the base dir, only if no ancestor there is a symlink (never leave the tree)."""
            parts = rel.split("/")
            for i in range(1, len(parts)):
                anc = "/".join(parts[:i])
                if anc not in base_ok:
                    try:
                        base_ok[anc] = stat_mod.S_ISDIR(os.lstat(base / anc).st_mode)
                    except OSError:
                        base_ok[anc] = False
                if not base_ok[anc]:
                    return None
            try:
                return os.lstat(base / rel)
            except OSError:
                return None

        def add_tree(rel: str) -> None:
            members[rel] = str(rootfs / rel)
            if stat_mod.S_ISDIR(os.lstat(rootfs / rel).st_mode):
                for dirpath, dirnames, filenames in os.walk(rootfs / rel):
                    for n in dirnames + filenames:
                        full = os.path.join(dirpath, n)
                        members[os.path.relpath(full, rootfs)] = full

        if os.lstat(rootfs).st_ctime_ns >= self._marker_ctime_ns:       # "/" itself: only its listing matters
            deleted += sorted(set(os.listdir(base)) - set(os.listdir(rootfs)))
        for rel in changed:
            try:
                st = os.lstat(rootfs / rel)
            except OSError:
                continue                       # vanished while listing
            bs = base_stat(rel)
            if bs is None:
                add_tree(rel)
            elif stat_mod.S_IFMT(bs.st_mode) != stat_mod.S_IFMT(st.st_mode):
                deleted.append(rel)            # replaced by a different type: remove, then extract
                add_tree(rel)
            else:
                members[rel] = str(rootfs / rel)
                if stat_mod.S_ISDIR(st.st_mode):
                    gone = set(os.listdir(base / rel)) - set(os.listdir(rootfs / rel))
                    deleted += [f"{rel}/{n}" for n in gone]
        tools = False
        if SANDBOX_TOOLS_PATH.lstrip("/") in members:
            tools = self.exec(["sha256sum", SANDBOX_TOOLS_PATH]).stdout.split()[:1] == [SANDBOX_TOOLS_SHA256[_arch()]]
            if tools:
                del members[SANDBOX_TOOLS_PATH.lstrip("/")]   # pristine helper: re-inject instead of archiving 15 MB
        for sub, arc in (("tmp", "tmp"), ("var_tmp", "var/tmp")):    # private /tmp, /var/tmp (empty in the base)
            wdir = self.dir / "work" / sub
            for dirpath, dirnames, filenames in os.walk(wdir):
                for n in dirnames + filenames:
                    full = os.path.join(dirpath, n)
                    members[f"{arc}/{os.path.relpath(full, wdir)}"] = full
        # parents first; deleted paths that also appear as members are re-created by the extract
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for arc in sorted(members):
                try:
                    ti = tar.gettarinfo(members[arc], arcname=arc)
                except OSError:
                    continue
                if ti is None:
                    continue                   # socket
                ti.uname = ti.gname = ""
                if ti.isreg():
                    try:
                        with open(members[arc], "rb") as f:
                            tar.addfile(ti, f)
                    except OSError:
                        continue
                else:
                    tar.addfile(ti)
        return SandboxSnapshot(base_image=self.image, archive=buf.getvalue() if members else b"",
                               deleted=tuple(sorted(set(deleted))), sandbox_tools=tools, source_container=self.name,
                               config=self.config(), backend="apptainer")

    @classmethod
    def from_snapshot(cls, snapshot: SandboxSnapshot, **overrides) -> "ApptainerSandbox":
        """A new, started sandbox (its own copy of the base directory) whose filesystem equals the snapshot's."""
        c = snapshot.config
        kw = {"memory": c["memory"], "timeout_retry": c["timeout_retry"], "pids_limit": c.get("pids_limit")} | overrides
        sb = cls(snapshot.base_image, **kw)
        sb.snapshot_of = snapshot.digest()
        sb.start()
        try:
            if snapshot.deleted:
                r = sb.exec(["xargs", "-0", "rm", "-rf", "--"], input="\0".join(snapshot.deleted))
                if not r.success:
                    raise RuntimeError(f"restore: deleting paths failed: {r.stderr}")
            if snapshot.sandbox_tools:
                sb.ensure_sandbox_tools()
            if snapshot.archive:   # in-container tar, so symlinks resolve inside the container; last: directory metadata wins
                argv = sb._argv(["tar", "-x", "-p", "-f", "-"], "/", None)
                _, err, rc, _ = _run_limited(argv, snapshot.archive, 300, limit=None, drain_timeout=0.5)
                if rc != 0:
                    raise RuntimeError(f"restore: tar failed ({rc}): {err.decode(errors='replace')}")
        except BaseException:
            sb.close()
            raise
        return sb

    # ---- exec / files --------------------------------------------------------------

    def _wrapper(self) -> str:
        """Per-exec shell prologue: umask and HOME as in the image's root session, memory limit, no apptainer vars."""
        pre = f"umask 022; ulimit -d {_memory_kb(self.memory)} 2>/dev/null; "
        nproc = self.pids_limit if self.pids_limit is not None else os.environ.get(APPTAINER_NPROC_ENV)
        if nproc:
            pre += f"ulimit -u {int(nproc)} 2>/dev/null; "
        return pre + f"unset {_APPTAINER_ENV_VARS}; HOME=/root; export HOME; exec \"$@\""

    def _argv(self, cmd: list[str], cwd: str, env: dict[str, str] | None) -> list[str]:
        if env:
            cmd = ["env", *[f"{k}={v}" for k, v in env.items()], *cmd]
        return [apptainer_binary(), "--silent", "exec", "--writable", "--cleanenv", "--pwd", cwd,
                f"instance://{self.name}", "sh", "-c", self._wrapper(), "ct-exec", *cmd]

    def exec(self, cmd: list[str], input: str | bytes | None = None, timeout: float | None = None,
             cwd: str | None = None, env: dict[str, str] | None = None, user: str | None = None,
             timeout_retry: bool | None = None) -> ExecResult:
        """DockerSandbox.exec semantics (inspect 0.3.137): retries, output limit, orphaned timed-out commands.

        On timeout only the apptainer client is killed and the command keeps running in the
        instance, as under docker. `user` is not supported (everything runs as the invoking user).
        """
        self._require()
        if user not in (None, "", "root"):
            raise NotImplementedError("apptainer backend cannot switch user")
        final_cwd = posixpath.join(WORKDIR, cwd) if cwd is not None else WORKDIR
        argv = self._argv(cmd, final_cwd, env)
        data = input.encode() if isinstance(input, str) else input
        retry = self.timeout_retry if timeout_retry is None else timeout_retry
        t0 = time.perf_counter()
        if timeout is None:
            out, err, rc, _ = _run_limited(argv, data, None, drain_timeout=0.5)
        else:
            tries = [timeout] + ([max(min(timeout, 60) // r, 1) for r in (1, 2)] if retry else [])
            for t in tries:
                out, err, rc, timed_out = _run_limited(argv, data, t, drain_timeout=0.5)
                if not timed_out:
                    break
            else:
                return ExecResult(False, -1, "", "", timed_out=True, seconds=time.perf_counter() - t0)
        stdout, stderr = out.decode(), err.decode()
        seconds = time.perf_counter() - t0
        if len(out) > MAX_EXEC_OUTPUT_SIZE or len(err) > MAX_EXEC_OUTPUT_SIZE:
            raise OutputLimitExceededError(MAX_EXEC_OUTPUT_SIZE_STR,
                                           _truncate_bytes(stdout) + _truncate_bytes(stderr))
        result = ExecResult(rc == 0, rc, stdout, stderr, seconds=seconds)
        if rc == 126 and "permission denied" in stdout:
            raise PermissionError(f"Permission denied executing command: {_inspect_repr(result)}")
        return result

    def container_file(self, path: str) -> str:
        return path if path.startswith("/") else posixpath.join(WORKDIR, path)

    def write_file(self, path: str, content: str | bytes) -> None:
        """As DockerSandbox.write_file: mkdir -p the parent, then `tee` the content in (inside the instance)."""
        file = self.container_file(path)
        parent = posixpath.dirname(file)
        if parent not in (".", ""):
            r = self.exec(["mkdir", "-p", parent])
            if not r.success:
                raise RuntimeError(f"Failed to create container directory {parent}: {r.stderr}")
        if isinstance(content, str):
            r = self.exec(["sh", "-e", "-c", 'tee -- "$1" > /dev/null', "write_file_script", file],
                          input=content, timeout=WRITE_FILE_TIMEOUT)
        else:
            import base64
            r = self.exec(["sh", "-e", "-c", 'base64 -d | tee -- "$1" > /dev/null', "write_file_script", file],
                          input=base64.b64encode(content).decode("ascii"), timeout=WRITE_FILE_TIMEOUT)
        if r.timed_out:
            raise TimeoutError(f"write_file {file} timed out")
        if r.returncode != 0:
            low = r.stderr.casefold()
            if "permission denied" in low:
                raise PermissionError(f"Permission was denied. Error details: {r.stderr}")
            if "cannot overwrite directory" in low or "is a directory" in low:
                raise IsADirectoryError(f"Failed to write file: {file} because it is a directory already")
            raise RuntimeError(f"failed to copy during write_file: {r}")

    def read_file(self, path: str, text: bool = True) -> str | bytes:
        """Read through the instance (`cat`), never through the host path: a symlink the agent
        planted resolves inside the container. Strict UTF-8, no newline translation, no size limit."""
        self._require()
        file = self.container_file(path)
        out, err, rc, timed_out = _run_limited(self._argv(["cat", "--", file], WORKDIR, None), None, 120, limit=None, drain_timeout=0.5)
        if timed_out:
            raise TimeoutError(f"read_file {file} timed out")
        if rc != 0:
            msg = err.decode(errors="replace").lower()
            if "no such file" in msg:
                raise FileNotFoundError(2, "No such file or directory.", path)
            if "is a directory" in msg:
                raise IsADirectoryError(21, "Is a directory", path)
            if "permission denied" in msg:
                raise PermissionError(13, "Permission denied.", path)
            raise RuntimeError(err.decode(errors="replace"))
        return out.decode("utf-8") if text else out

    def ensure_sandbox_tools(self) -> None:
        """Inject inspect's text_editor helper if missing (re-checked on every call), via exec stdin."""
        if self.exec(["test", "-f", SANDBOX_TOOLS_PATH]).success:
            return
        blob = sandbox_tools_binary().read_bytes()
        r = self.exec(["sh", "-e", "-c", 'cat > "$1" && chmod 755 "$1"', "inject", SANDBOX_TOOLS_PATH],
                      input=blob, timeout=120)
        if not r.success:
            raise RuntimeError(f"sandbox tools injection failed: {r.stderr}")

    def _require(self) -> None:
        if self.container_id is None:
            raise RuntimeError("sandbox not started")


def _memory_kb(memory: str) -> int:
    m = memory.strip().lower()
    mult = {"k": 1, "m": 1024, "g": 1024**2}.get(m[-1:], None)
    return int(float(m[:-1] if mult else m) * (mult or 1 / 1024))


_APPT_VERSION: list[str] = []


def _apptainer_version() -> str:
    if not _APPT_VERSION:
        _APPT_VERSION.append(_apptainer(["--version"], check=False).stdout.strip())
    return _APPT_VERSION[0]


def apptainer_cleanup_leftovers(include_live: bool = False) -> int:
    """Stop our instances and delete our rollout copies whose runner process is gone.

    Only objects of this user (apptainer lists only the caller's instances), with our name
    prefix, created on this host, whose recorded owner pid is dead (pid + start time, so a reused
    pid does not count as alive). `include_live=True` also removes those of running processes,
    including other runners: do not use it while rollouts are running.
    """
    def owner_dead(d: pathlib.Path) -> bool:
        try:
            o = json.loads((d / "owner.json").read_text())
        except (OSError, ValueError):
            # No owner record: either a start() that died before writing it, or one that is
            # writing it right now in another runner (two runners starting together raced here
            # in S1). Only the first is garbage, so require the directory to be old.
            try:
                return time.time() - d.stat().st_mtime > ORPHAN_GRACE_SECONDS
            except OSError:
                return False
        if o.get("host") != _host_id():
            return False
        return _proc_start(o["pid"]) != o.get("start")

    rollouts = apptainer_paths()["rollouts"]
    n = 0
    listed = _apptainer(["instance", "list", "--json"], check=False)
    try:
        instances = json.loads(listed.stdout).get("instances", []) if listed.stdout.strip() else []
    except ValueError:
        instances = []
    for inst in instances:
        name = inst.get("instance", "")
        if name.startswith(APPTAINER_PREFIX) and (include_live or owner_dead(rollouts / name)):
            _apptainer(["instance", "stop", "--force", name], check=False, timeout=60)
            n += 1
    if rollouts.is_dir():
        for d in rollouts.iterdir():
            if d.name.startswith(APPTAINER_PREFIX) and (include_live or owner_dead(d)):
                _rmtree(d)
                n += 1
    return n


# ---- backend selection ---------------------------------------------------------------

BACKEND_ENV = "CT_RH_SANDBOX"


def sandbox_backend() -> str:
    """`CT_RH_SANDBOX=docker|apptainer` (default docker)."""
    b = os.environ.get(BACKEND_ENV, "docker").strip().lower() or "docker"
    if b not in ("docker", "apptainer"):
        raise ValueError(f"{BACKEND_ENV}={b!r}: expected 'docker' or 'apptainer'")
    return b


def sandbox_class(backend: str | None = None):
    return ApptainerSandbox if (backend or sandbox_backend()) == "apptainer" else DockerSandbox


def new_sandbox(**kw):
    """A new, not yet started sandbox of the selected backend (`with new_sandbox() as sb:`)."""
    return sandbox_class()(**kw)


def sandbox_from_snapshot(snapshot: SandboxSnapshot, **overrides):
    """Restore a snapshot with the backend that made it (a snapshot is not portable across backends)."""
    return sandbox_class(snapshot.backend).from_snapshot(snapshot, **overrides)


def backend_available(backend: str | None = None) -> bool:
    return apptainer_available() if (backend or sandbox_backend()) == "apptainer" else docker_available()


def ensure_image(image: str = IMAGE) -> None:
    """One-time image setup for the selected backend (docker pull / apptainer build of the pinned digest)."""
    if sandbox_backend() == "apptainer":
        ensure_apptainer_image()
    else:
        docker_ensure_image(image)


def cleanup_leftovers(**kw) -> int:
    """Remove leftovers of crashed runs. docker: every container with our label. apptainer: only this
    user's instances/copies of dead runner processes (`include_live=True` to also remove live ones)."""
    return apptainer_cleanup_leftovers(**kw) if sandbox_backend() == "apptainer" else docker_cleanup_leftovers()


Sandbox = DockerSandbox | ApptainerSandbox
