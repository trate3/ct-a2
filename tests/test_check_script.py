"""check.sh must fail when the protocol demo does not behave as intended.

A fake interpreter wraps the real one and breaks only the selected demo run, so the
test exercises the shell script's own status handling.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fake_python(tmp_path: Path, broken: str) -> Path:
    cond = {"honest": '[[ "$*" == *"protocol-integrity-exp e2e"* && "$*" != *"--tamper-environment"* ]]',
            "tamper": '[[ "$*" == *"protocol-integrity-exp e2e"* && "$*" == *"--tamper-environment"* ]]'}[broken]
    out = {"honest": 'echo "[certifier] ABORT: forced failure for the test"; exit 1',
           "tamper": 'echo "[certifier] CERTIFIED  m(FN)=0.000  m_fp=0.000"; exit 0'}[broken]
    fake = tmp_path / "python"
    fake.write_text(f"#!/usr/bin/env bash\nif {cond}; then {out}; fi\nexec {sys.executable} \"$@\"\n")
    fake.chmod(0o755)
    return fake


def _check(python: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "STEPS": "protocol", "PYTHON": python}
    return subprocess.run(["bash", str(ROOT / "check.sh")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_protocol_step_passes_with_the_real_demo():
    r = _check(sys.executable)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "certified, certificate VALID" in r.stdout and "aborted for the intended reason" in r.stdout


def test_failed_honest_demo_makes_check_exit_nonzero(tmp_path):
    r = _check(str(_fake_python(tmp_path, "honest")))
    assert r.returncode != 0
    assert "PROTOCOL DEMO FAILED" in r.stdout and "FAILED steps: protocol" in r.stdout


def test_tampered_demo_that_certifies_makes_check_exit_nonzero(tmp_path):
    r = _check(str(_fake_python(tmp_path, "tamper")))
    assert r.returncode != 0
    assert "FAILED steps: protocol" in r.stdout
