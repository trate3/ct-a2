"""Sandbox backends (docker, apptainer): isolation settings, exec/file semantics, snapshots, cleanup."""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import time

import pytest

from evals.reward_hacking import sandbox as sbmod
from evals.reward_hacking import tools
from evals.reward_hacking.sandbox import (IMAGE, LABEL, OutputLimitExceededError, new_sandbox,
                                          sandbox_from_snapshot)

from evals.reward_hacking.conftest import requires_sandbox

pytestmark = requires_sandbox


def _inspect(name: str) -> dict:
    return json.loads(subprocess.run(["docker", "inspect", name], capture_output=True, text=True, check=True).stdout)[0]


def _exists(sb) -> bool:
    if sbmod.sandbox_backend() == "docker":
        return subprocess.run(["docker", "inspect", sb.name], capture_output=True).returncode == 0
    listed = json.loads(sbmod._apptainer(["instance", "list", "--json"]).stdout or "{}").get("instances", [])
    return any(i["instance"] == sb.name for i in listed) or (sbmod.apptainer_paths()["rollouts"] / sb.name).exists()


def _docker_only(backend):
    if backend != "docker":
        pytest.skip("docker-specific")


def _apptainer_only(backend):
    if backend != "apptainer":
        pytest.skip("apptainer-specific")


def test_container_settings_and_isolation(sandbox, backend):
    # no network beyond loopback (both backends)
    r = sandbox.exec(["python", "-c", "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"])
    assert not r.success and "Network is unreachable" in r.stderr
    assert sandbox.exec(["sh", "-c", "ls /sys/class/net"]).stdout.split() == ["lo"]
    if backend == "docker":
        info = _inspect(sandbox.name)
        hc = info["HostConfig"]
        assert info["Config"]["Image"] == IMAGE
        assert hc["NetworkMode"] == "none" and hc["Init"] is True
        assert hc["Memory"] == 1024**3 and hc["PidsLimit"] == 512 and hc["NanoCpus"] == 2 * 10**9
        assert hc["Binds"] in (None, []) and info["Mounts"] == []
        assert hc["Privileged"] is False and "no-new-privileges" in hc["SecurityOpt"]
        assert info["Config"]["Labels"][LABEL] == "1"
    else:
        cfg = sandbox.config()
        assert cfg["network"] == "none" and cfg["mounts"] == []
        assert "--containall" in cfg["apptainer"]["flags"] and cfg["apptainer"]["base_tree_sha256"]
        # host paths are invisible: no /home/<user> content, no /scratch-like host mounts, no host env
        user_home = os.path.expanduser("~")
        assert sandbox.exec(["sh", "-c", f"ls -A {user_home} | wc -l"]).stdout.strip() == "0"
        env = sandbox.exec(["env"]).stdout
        assert not any(k in env for k in ("SSH_", "SLURM", "APPTAINER", "SINGULARITY", "CT_RH"))
        # the agent's writes land in its own copy only, never in the shared base directory
        sandbox.write_file("/func.py", "x")
        assert not (sbmod.apptainer_paths()["base"] / "func.py").exists()
        assert (sandbox.dir / "rootfs" / "func.py").read_text() == "x"
        # a symlink to a host path resolves inside the container, so read_file cannot be used to escape
        sandbox.exec(["ln", "-s", "/etc/hostname", "/hn"])
        assert sandbox.read_file("/hn") == sandbox.exec(["cat", "/etc/hostname"]).stdout


def test_exec_semantics(sandbox, backend):
    r = sandbox.exec(["sh", "-c", "pwd; echo err >&2; exit 3"])
    assert (r.success, r.returncode, r.stdout, r.stderr) == (False, 3, "/\n", "err\n")
    assert sandbox.exec(["cat"], input="piped").stdout == "piped"
    assert sandbox.exec(["pwd"], cwd="tmp").stdout == "/tmp\n"
    assert sandbox.exec(["sh", "-c", "echo $FOO"], env={"FOO": "bar"}).stdout == "bar\n"
    # the one model-visible difference: apptainer cannot run as root
    assert sandbox.exec(["id", "-un"]).stdout == ("root\n" if backend == "docker" else getpass.getuser() + "\n")
    assert sandbox.exec(["sh", "-c", "umask"]).stdout == "0022\n"
    assert sandbox.exec(["sh", "-c", "echo $HOME"]).stdout == "/root\n"
    with pytest.raises(UnicodeDecodeError):
        sandbox.exec(["printf", "\\377"])
    with pytest.raises(OutputLimitExceededError):
        sandbox.exec(["sh", "-c", "head -c 11000000 /dev/zero"])


def test_exec_timeout_retries_and_orphans(sandbox):
    """inspect retries a timed-out command twice; the client is killed, the command is not."""
    t0 = time.perf_counter()
    r = sandbox.exec(["sh", "-c", "echo run >> /tmp/count; sleep 30"], timeout=2)
    elapsed = time.perf_counter() - t0
    assert r.timed_out and not r.success and r.stdout == ""
    assert 4.5 < elapsed < 15          # 2 + 2 + 1 seconds of timeouts, plus client shutdown
    assert sandbox.read_file("/tmp/count") == "run\nrun\nrun\n"
    assert sandbox.exec(["sh", "-c", "ps -eo args | grep -c '^sleep 30'"]).stdout.strip() == "3"
    r = sandbox.exec(["sh", "-c", "echo once >> /tmp/once; sleep 30"], timeout=1, timeout_retry=False)
    assert r.timed_out and sandbox.read_file("/tmp/once") == "once\n"


def test_files(sandbox):
    sandbox.write_file("dir/sub/a.txt", "x\r\ny\n")
    assert sandbox.read_file("/dir/sub/a.txt") == "x\r\ny\n"      # no newline translation
    sandbox.write_file("/b.bin", b"\x00\xff")
    assert sandbox.read_file("b.bin", text=False) == b"\x00\xff"
    sandbox.write_file("empty.txt", "")
    assert sandbox.read_file("empty.txt") == ""
    with pytest.raises(FileNotFoundError):
        sandbox.read_file("missing.py")
    with pytest.raises(IsADirectoryError):
        sandbox.write_file("/tmp", "x")
    with pytest.raises(IsADirectoryError):
        sandbox.read_file("/tmp")
    with pytest.raises(UnicodeDecodeError):
        sandbox.read_file("b.bin")


def test_snapshot_restore_is_independent(sandbox, backend):
    sandbox.write_file("func.py", "v1\n")
    sandbox.write_file("/tmp/state", "s\n")
    sandbox.exec(["sh", "-c", "rm /etc/issue; mkdir -p /w/d; chmod 700 /w/d; ln -s /func.py /w/link; "
                              "mkdir /gone && echo a > /gone/f && mv /gone /moved; echo v > /var/tmp/vt"])
    if backend == "docker":
        sandbox.exec(["chown", "1000:1000", "/tmp/state"])
    edit = tools.execute_tool(sandbox, "text_editor", {"command": "str_replace", "path": "/func.py",
                                                       "old_str": "v1", "new_str": "v2"})
    assert edit.error is None
    snap = sandbox.snapshot()
    assert snap.sandbox_tools and len(snap.archive) < 1_000_000
    assert "/etc/issue" in {d if d.startswith("/") else "/" + d for d in snap.deleted}
    assert snap.backend == backend
    with sandbox_from_snapshot(snap) as clone:
        assert clone.name != sandbox.name and clone.snapshot_of == snap.digest()
        assert clone.read_file("func.py") == "v2\n" and clone.read_file("/tmp/state") == "s\n"
        assert clone.read_file("/var/tmp/vt") == "v\n" and clone.read_file("/moved/f") == "a\n"
        same = "stat -c '%a %u %N' /w/d /w/link /tmp/state /moved; test -e /etc/issue || echo gone; test -e /gone || echo gone2"
        assert clone.exec(["sh", "-c", same]).stdout == sandbox.exec(["sh", "-c", same]).stdout
        # inspect's editor history survives: undo in the clone restores v1
        undo = tools.execute_tool(clone, "text_editor", {"command": "undo_edit", "path": "/func.py"})
        assert undo.error is None and clone.read_file("func.py") == "v1\n"
        assert sandbox.read_file("func.py") == "v2\n"
        sandbox.write_file("func.py", "original-changed\n")
        clone.write_file("func.py", "clone-changed\n")
        clone.exec(["rm", "/tmp/state"])
        assert sandbox.read_file("func.py") == "original-changed\n"
        assert sandbox.read_file("/tmp/state") == "s\n"
        assert clone.read_file("func.py") == "clone-changed\n"
        if backend == "docker":
            assert _inspect(clone.name)["HostConfig"]["NetworkMode"] == "none"
        else:
            assert clone.config()["network"] == "none" and clone.dir != sandbox.dir
        # a second restore from the same snapshot sees the snapshot state, not the clone's
        with sandbox_from_snapshot(snap) as clone2:
            assert clone2.read_file("func.py") == "v2\n"
        assert not _exists(clone2)
        # a snapshot of a restored sandbox is again exact (changes are relative to the base, not the parent)
        snap2 = clone.snapshot()
        with sandbox_from_snapshot(snap2) as clone3:
            assert clone3.read_file("func.py") == "clone-changed\n" and clone3.exec(["test", "-e", "/tmp/state"]).returncode == 1
            assert clone3.read_file("/moved/f") == "a\n"
    assert not _exists(clone)


def test_close_always_removes():
    sb = new_sandbox()
    sb.start()
    assert _exists(sb)
    sb.close()
    assert not _exists(sb)
    with pytest.raises(RuntimeError):
        sb.exec(["true"])
    with pytest.raises(ZeroDivisionError):
        with new_sandbox() as sb2:
            1 / 0
    assert not _exists(sb2)


def test_apptainer_cleanup_only_removes_dead_runners(backend):
    _apptainer_only(backend)
    import json as _json
    import pathlib
    rollouts = sbmod.apptainer_paths()["rollouts"]
    with new_sandbox() as live:
        # a leftover directory of a dead runner (pid 0x7fffffff cannot exist) and one of another host
        dead = rollouts / f"{sbmod.APPTAINER_PREFIX}{sbmod._host_id()}-2147483646-deadbeef"
        other = rollouts / f"{sbmod.APPTAINER_PREFIX}otherhost-1-cafebabe"
        foreign = rollouts / "not-ours"
        for d, owner in ((dead, {"host": sbmod._host_id(), "pid": 2147483646, "start": "1"}),
                         (other, {"host": "elsewhere", "pid": 1, "start": "1"}), (foreign, None)):
            (d / "rootfs").mkdir(parents=True)
            if owner:
                (d / "owner.json").write_text(_json.dumps(owner))
        (dead / "rootfs" / "ro").mkdir()
        (dead / "rootfs" / "ro" / "f").write_text("x")
        (dead / "rootfs" / "ro").chmod(0o500)           # the agent made an unwritable tree
        assert sbmod.cleanup_leftovers() == 1
        assert not dead.exists() and other.exists() and foreign.exists() and _exists(live)
        assert live.exec(["true"]).success
        for d in (other, foreign):
            sbmod._rmtree(d)
