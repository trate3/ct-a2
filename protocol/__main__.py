"""Command line: run the CPU demo, or verify a certificate against a pinned local key."""
import argparse
import json
from pathlib import Path

from ct_core.protocol.attestor import StubAttestationVerifier

from .codec import decode
from .demo import run_demo
from .protocol import verify_certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify-local")
    verify.add_argument("--certificate", type=Path, required=True)
    verify.add_argument("--anchor", type=Path, required=True)
    verify.add_argument("--spec", type=Path, required=True, help="independently expected public spec JSON")
    verify.add_argument("--expected-id", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        print(json.dumps(run_demo(args.out), indent=2))
        return 0
    anchor = decode(args.anchor.read_bytes())
    verifier = StubAttestationVerifier(bytes.fromhex(anchor["verify_key"]), anchor["expected_measurement"])
    valid = verify_certificate(decode(args.certificate.read_bytes()), verifier,
                               expected_spec=decode(args.spec.read_bytes()), expected_id=args.expected_id)
    print(json.dumps({"valid": valid, "mode": "local pinned-key check"}))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

