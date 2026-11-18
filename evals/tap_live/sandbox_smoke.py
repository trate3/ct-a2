"""Standalone smoke test of the repo's sandbox (the one `rollout.run_rollout` uses) on the machine that will run the
live coding measurement. Prints what it did and the seconds each step took; exit status 0 only if every core step passed.

  export PATH="$PWD/evals/tap_live/podman_shim:$PATH"     # only on a machine with podman and no docker
  CT_RH_SANDBOX=docker python -m evals.tap_live.sandbox_smoke

It uses the pinned image aisiuk/inspect-tool-support@sha256:bbaecf7a093cfe24703c97d562bb059899e272cd5d57f16d9916fe05e6614419
(`sandbox.IMAGE`), starts one sandbox and runs: `python3 -c 'print(1+1)'` through `sandbox.exec`; a bash tool call and a
python tool call through `tools.execute_tool` (as the agent loop does); a write_file / read_file round trip (`docker cp`);
a snapshot (`docker diff`, `tar`); the text_editor tool (needs the injected helper binary, so network or the cache); a
timed-out command (`--with-timeout`, 3 s). Then it removes the container. Steps that need more than the core are reported
separately, so a podman difference shows up as one failing line.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

from evals.reward_hacking import sandbox, tools

results: list[tuple[str, bool, float, str]] = []


def step(name: str, core: bool = True):
    def deco(fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            ok = True
        except Exception as e:      # noqa: BLE001 - a smoke test reports every failure
            ok, detail = False, f"{type(e).__name__}: {e}"
        dt = time.perf_counter() - t0
        results.append((name + ("" if core else " (not core)"), ok, dt, str(detail)[:300].replace("\n", " | ")))
        print(f"{'ok  ' if ok else 'FAIL'} {dt:7.2f}s  {name}: {str(detail)[:300]!r}", flush=True)
        return fn
    return deco


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-timeout", action="store_true", help="also run a command that times out (takes ~5 s)")
    a = ap.parse_args(argv)
    print(f"backend: {sandbox.sandbox_backend()}; image: {sandbox.IMAGE}")
    exe = shutil.which("docker")
    print(f"docker on PATH: {exe}")
    if exe:
        print(subprocess.run(["docker", "--version"], capture_output=True, text=True).stdout.strip())
    box = {}

    @step("ensure_image (pull if missing)")
    def _():
        sandbox.ensure_image()
        return sandbox.IMAGE

    @step("sandbox_tools_binary (inspect helper, may fetch the wheel)", core=False)
    def _():
        return str(sandbox.sandbox_tools_binary())

    @step("start sandbox")
    def _():
        box["sb"] = sandbox.new_sandbox()
        box["sb"].start()
        return box["sb"].name

    sb = box.get("sb")
    if sb is not None:
        try:
            @step("exec python3 -c 'print(1+1)'")
            def _():
                r = sb.exec(["python3", "-c", "print(1+1)"], timeout=60)
                assert r.success and r.stdout.strip() == "2", (r.returncode, r.stdout, r.stderr)
                return r.stdout.strip()

            @step("bash tool call")
            def _():
                out = tools.execute_tool(sb, "bash", {"cmd": "echo hello; id -u; pwd"})
                assert out.infra_error is None and "hello" in out.output_text, out.output_text
                return out.output_text

            @step("python tool call")
            def _():
                out = tools.execute_tool(sb, "python", {"code": "print(sum(range(10)))"})
                assert out.infra_error is None and out.output_text.strip() == "45", out.output_text
                return out.output_text.strip()

            @step("write_file / read_file round trip")
            def _():
                sb.write_file("func.py", "def f():\n    return 1\n")
                got = sb.read_file("func.py")
                assert got == "def f():\n    return 1\n", got
                return "func.py"

            @step("read_file of a missing file raises FileNotFoundError")
            def _():
                try:
                    sb.read_file("does_not_exist.py")
                except FileNotFoundError:
                    return "FileNotFoundError"
                raise AssertionError("no error")

            @step("snapshot (diff + tar)")
            def _():
                snap = sb.snapshot()
                return f"{len(snap.archive)} bytes, {len(snap.deleted)} deleted"

            @step("text_editor tool (view)", core=False)
            def _():
                out = tools.execute_tool(sb, "text_editor", {"command": "view", "path": "/func.py"})
                assert out.infra_error is None and "return 1" in out.output_text, (out.infra_error, out.output_text)
                return out.output_text[:80]

            if a.with_timeout:
                @step("command timeout is reported", core=False)
                def _():
                    r = sb.exec(["sleep", "30"], timeout=3, timeout_retry=False)
                    assert r.timed_out
                    return f"timed out after {r.seconds:.1f}s"
        finally:
            @step("remove sandbox")
            def _():
                sb.close()
                return "removed"
    print("\nsummary:")
    bad = [r for r in results if not r[1] and "(not core)" not in r[0]]
    for name, ok, dt, _ in results:
        print(f"  {'ok  ' if ok else 'FAIL'} {dt:7.2f}s  {name}")
    print("CORE: " + ("PASS" if not bad else f"FAIL ({len(bad)} step(s))"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
