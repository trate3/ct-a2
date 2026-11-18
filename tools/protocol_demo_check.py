"""Run the two protocol demos and assert their intended outcomes (used by check.sh).

Honest run: must exit 0 and print both the certifier's CERTIFIED line and a VALID certificate.
Tampered run (a relay substitutes the environment ciphertext): must exit nonzero and abort
for the intended reason, "certificate is not bound to this environment message". Any other
outcome, including a different abort reason or a crash, is a failure.

Exit status 0 only if both demos behave as intended. Uses the local signing key.
"""

from __future__ import annotations

import os
import subprocess
import sys

TAMPER_REASON = "[certifier] ABORT: certificate is not bound to this environment message"


def run(*args: str) -> subprocess.CompletedProcess:
    cmd = [os.environ.get("PYTHON", sys.executable), "-m", "evals_runner", "protocol-integrity-exp", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    ok = True
    setup = run("attestation-setup")
    if setup.returncode != 0:
        print(f"PROTOCOL DEMO FAILED: attestation setup exited {setup.returncode}\n{setup.stderr[-800:]}")
        return 1
    honest = run("e2e", "--n", "8")
    lines = honest.stdout.splitlines()
    honest_ok = (honest.returncode == 0
                 and any(l.startswith("[certifier] CERTIFIED") for l in lines)
                 and any(l.split(":")[0].strip() == "certificate" and l.rstrip().endswith("VALID")
                         and not l.rstrip().endswith("INVALID") for l in lines))
    print("honest run:   " + ("certified, certificate VALID" if honest_ok else "FAILED"))
    if not honest_ok:
        print(f"  exit {honest.returncode}\n" + (honest.stdout + honest.stderr)[-1200:])
        ok = False
    tamper = run("e2e", "--n", "8", "--tamper-environment")
    tamper_ok = tamper.returncode != 0 and TAMPER_REASON in tamper.stdout.splitlines() \
        and "[certifier] CERTIFIED" not in tamper.stdout
    print("tampered run: " + ("aborted for the intended reason (certificate not bound to the environment message)"
                              if tamper_ok else "FAILED"))
    if not tamper_ok:
        print(f"  exit {tamper.returncode}\n" + (tamper.stdout + tamper.stderr)[-1200:])
        ok = False
    if not ok:
        print("PROTOCOL DEMO FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
