"""Local adapter checks; the fake CLI exercises Inspect's provider API, not Apptainer."""

import asyncio
import pathlib
import subprocess
import sys
import types

import pytest

pytest.importorskip("inspect_ai")

from evals.reward_hacking import inspect_apptainer as provider
from evals.reward_hacking.swebench_apptainer import (
    dockerhub_image_for_id,
    impossible_swebench_apptainer,
)
from inspect_ai import Task
from inspect_ai.dataset import Sample


def test_dockerhub_image_names_match_official_instance_tag_shape():
    assert dockerhub_image_for_id("Astropy__Astropy-123") == (
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-123:latest"
    )
    digest = "sha256:" + "a" * 64
    assert dockerhub_image_for_id("Repo__Project-42", digest=digest).endswith("@" + digest)


def test_official_task_adapter_disables_docker_and_sets_instance_sandbox(monkeypatch):
    captured = {}

    def fake_task(**kwargs):
        captured.update(kwargs)
        captured["dataset_revision"] = fake_module.hf_dataset(path="fjzzq2002/impossible_swebench")["revision"]
        return Task(dataset=[Sample(input="issue", id="Repo__Project-42")], name="swebench_conflicting")

    fake_module = types.ModuleType("impossiblebench.swebench_tasks")
    fake_module.impossible_swebench = fake_task
    fake_module.hf_dataset = lambda **kwargs: kwargs
    package = types.ModuleType("impossiblebench")
    monkeypatch.setitem(sys.modules, "impossiblebench", package)
    monkeypatch.setitem(sys.modules, "impossiblebench.swebench_tasks", fake_module)

    digest = "sha256:" + "b" * 64
    task = impossible_swebench_apptainer(
        split="conflicting", agent_type="tools", instance_ids=["Repo__Project-42"],
        image_digests={"Repo__Project-42": digest},
    )
    sample = task.dataset[0]
    assert captured["build_docker_images"] is False
    assert sample.sandbox.type == "ct_swe_apptainer"
    assert sample.sandbox.config == f"swebench/sweb.eval.x86_64.repo_1776_project-42@{digest}"
    assert sample.metadata["instance_id"] == "Repo__Project-42"
    assert captured["dataset_revision"] == "9c2d34f364b7229e8c0ff807c646100bdc18bbb5"


def test_sample_lifecycle_and_binary_file_io(tmp_path, monkeypatch):
    base = tmp_path / "base"
    (base / "testbed").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(base / "testbed")], check=True)
    subprocess.run(["git", "-C", str(base / "testbed"), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(base / "testbed"), "config", "user.email", "test@example.com"], check=True)
    (base / "testbed" / "README").write_text("base\n")
    subprocess.run(["git", "-C", str(base / "testbed"), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(base / "testbed"), "commit", "-qm", "base"], check=True)
    base_commit = subprocess.check_output(["git", "-C", str(base / "testbed"), "rev-parse", "HEAD"], text=True).strip()
    (base / "testbed" / "README").write_text("changed by test patch\n")
    test_patch = subprocess.check_output(["git", "-C", str(base / "testbed"), "diff"], text=True)
    subprocess.run(["git", "-C", str(base / "testbed"), "reset", "--hard", "-q", base_commit], check=True)
    monkeypatch.setattr(provider, "_ensure_image", lambda image, cached_sif=None: base)
    monkeypatch.setenv("CT_SWEBENCH_APPTAINER_ROOT", str(tmp_path / "cache"))

    fake = tmp_path / "apptainer"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, sys\n"
        "a=sys.argv[1:]; cwd='/'; root=None; i=1\n"
        "open(os.environ['FAKE_ARGS_PATH'],'w').write(' '.join(a))\n"
        "while i < len(a):\n"
        " if a[i] == '--pwd': cwd=a[i+1]; i+=2; continue\n"
        " if a[i] in ('--workdir','--network','--env','--no-mount'): i+=2; continue\n"
        " if a[i].startswith('-'): i+=1; continue\n"
        " root=a[i]; cmd=a[i+1:]; break\n"
        "if cmd and cmd[0] == 'python': cmd[0]=sys.executable\n"
        "p=subprocess.run(cmd, cwd=os.path.join(root, cwd.lstrip('/')), input=sys.stdin.buffer.read(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n"
        "sys.stdout.buffer.write(p.stdout); sys.stderr.buffer.write(p.stderr); sys.exit(p.returncode)\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("CT_SWEBENCH_APPTAINER", str(fake))
    args_file = tmp_path / "fake-args.txt"
    monkeypatch.setenv("FAKE_ARGS_PATH", str(args_file))

    async def run():
        envs = await provider.SweApptainerEnvironment.sample_init(
            "swe-test", "swebench/test:latest",
            {"instance_id": "test", "base_commit": base_commit, "test_patch": test_patch}
        )
        env = envs["default"]
        blob = b"hello\x00world\xff"
        await env.write_file("nested/blob.bin", blob)
        assert await env.read_file("nested/blob.bin", text=False) == blob
        result = await env.exec(["python3", "-c", "print('ok')"])
        assert result.success and result.stdout == "ok\n"
        head = await env.exec(["git", "rev-parse", "HEAD"], cwd="/testbed")
        assert head.success and head.stdout.strip() == base_commit
        args = args_file.read_text()
        assert "--no-home" in args and "--no-mount bind-paths,cwd" in args
        assert "--bind" not in args and "--network none" in args
        root = env._sample_root
        await provider.SweApptainerEnvironment.sample_cleanup("swe-test", None, envs, False)
        assert not pathlib.Path(root).exists()

    asyncio.run(run())


def test_ensure_localhost(tmp_path):
    from evals.reward_hacking.inspect_apptainer import LOCALHOST_LINES, ensure_localhost
    (tmp_path / "etc").mkdir()
    hosts = tmp_path / "etc" / "hosts"
    ensure_localhost(tmp_path)                      # missing file
    assert hosts.read_text() == LOCALHOST_LINES
    hosts.write_text("10.0.0.1 db")                 # existing entries kept, newline added
    ensure_localhost(tmp_path)
    assert hosts.read_text() == "10.0.0.1 db\n" + LOCALHOST_LINES
    hosts.write_text("127.0.0.1 localhost box\n")   # already resolvable: unchanged
    ensure_localhost(tmp_path)
    assert hosts.read_text() == "127.0.0.1 localhost box\n"
