"""Separate-process rehearsal of the three-message certified-taps protocol."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ct_core.certification import (
    COMMIT, TEST, CertifierAbort, prepare_environment, run_certifier,
    serve_session, verify_certificate,
)
from ct_core.config import ENCLAVE_PORT, RELAY_PORT
from ct_core.protocol import sealing
from ct_core.protocol.attestor import (
    GcpAttestationVerifier,
    GcpEnclaveSigner,
    initialize_local_attestation,
    publish_gcp_attestation,
)
from evals.basic_mocked import FIXTURE
from evals.basic_mocked.mocked_protocol_components import (
    MockAgentDriver, MockTapDriver, agent_package as mock_agent_package,
    build_agent as mock_agent, build_comparator,
    build_environment as mock_environment, build_tap as mock_tap,
    tap_package as mock_tap_package,
)
from evals_runner.util import run_tap_quality, write_certificate

REPO = Path(__file__).resolve().parents[1]


def _case_args(parser):
    parser.add_argument("--case", choices=("mock",), default="mock")
    parser.add_argument("--data", default=str(FIXTURE))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)


def configure_parser(parser):
    commands = parser.add_subparsers(dest="service", required=True)
    for name, run in (("attestation-setup", run_attestation_setup),
                      ("e2e", run_e2e), ("spec", run_spec), ("workload", run_workload),
                      ("certifier", run_certifier_service), ("relay", run_relay),
                      ("verify", run_verify), ("gcp-pin", run_gcp_pin)):
        command = commands.add_parser(name)
        if name in ("e2e", "spec", "workload", "certifier"):
            _case_args(command)
        if name == "e2e":
            command.add_argument("--tamper-environment", action="store_true")
            command.add_argument("--out", default=None)
        elif name in ("spec", "workload"):
            command.add_argument("--spec", required=True)
        elif name == "certifier":
            command.add_argument("--spec", required=True)
            command.add_argument("--peer-host", default="127.0.0.1")
            command.add_argument("--peer-port", type=int, default=RELAY_PORT)
            command.add_argument("--out", default=None)
        elif name == "relay":
            command.add_argument("--listen-host", default="127.0.0.1")
            command.add_argument("--listen-port", type=int, default=RELAY_PORT)
            command.add_argument("--workload-host", default="127.0.0.1")
            command.add_argument("--workload-port", type=int, default=ENCLAVE_PORT)
            command.add_argument("--replace-environment")
        elif name == "verify":
            command.add_argument("cert")
            command.add_argument("--spec", required=True)
        elif name == "gcp-pin":
            command.add_argument("--verify-key", required=True,
                                 help="hex-encoded verify_key printed by `workload --attestation gcp`")
            command.add_argument("--token", help="the raw attestation token (or use --token-file)")
            command.add_argument("--token-file")
            command.add_argument("--out", default=str(REPO / ".keys" / "gcp_attestation.json"))
        if name == "workload":
            command.add_argument("--listen-host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=ENCLAVE_PORT)
        if name in ("workload", "certifier"):
            command.add_argument("--attestation", choices=("local", "gcp"), default="local")
        if name == "workload":
            command.add_argument("--gcp-audience", default="certified-taps/v1")
            command.add_argument("--gcp-token-type", default="OIDC")
        if name == "certifier":
            command.add_argument("--gcp-publication", default=str(REPO / ".keys" / "gcp_attestation.json"))
        command.set_defaults(run=run)


def _components(args):
    return (mock_agent_package(args.data), mock_tap_package(), (MockAgentDriver,), (MockTapDriver,))


def _environment(args):
    return mock_environment(args.data, args.n, args.seed)


def _environment_driver(args):
    from util.dynamic_test_environment import StaticEnvironment
    return StaticEnvironment


def _dial(host, port):
    for _ in range(200):
        try:
            return socket.create_connection((host, port))
        except OSError:
            time.sleep(0.05)
    raise ConnectionError(f"could not connect to port {port}")


def run_spec(args):
    spec, _ = prepare_environment(_environment(args))
    Path(args.spec).write_text(json.dumps(spec, indent=2) + "\n")
    return 0


def run_attestation_setup(args):
    verifier = initialize_local_attestation()
    print(f"[setup] pinned m_ref={verifier.measurement()}")
    return 0


def run_workload(args):
    signer = None
    if args.attestation == "gcp":
        signer = GcpEnclaveSigner(audience=args.gcp_audience, token_type=args.gcp_token_type)
        print(json.dumps({"verify_key": signer.verify_key().hex(), "token": signer.token()}),
              flush=True)
        print("[workload] pin the line above with `gcp-pin` before the certifier connects",
              file=sys.stderr)
    server = socket.create_server((args.listen_host, args.port))
    connection, _ = server.accept()
    with connection:
        agent, tap, agent_drivers, tap_drivers = _components(args)
        serve_session(
            connection, agent, tap, json.loads(Path(args.spec).read_text()),
            agent_drivers=agent_drivers, tap_drivers=tap_drivers,
            environment_drivers=(_environment_driver(args),), signer=signer)
    server.close()
    return 0


def run_certifier_service(args):
    environment = _environment(args)
    spec = json.loads(Path(args.spec).read_text())
    verifier = (GcpAttestationVerifier.from_publication(Path(args.gcp_publication))
               if args.attestation == "gcp" else None)
    connection = _dial(args.peer_host, args.peer_port)
    try:
        cert = run_certifier(connection, environment, spec, verifier)
    except CertifierAbort as error:
        print(f"[certifier] ABORT: {error}")
        return 1
    finally:
        connection.close()
    write_certificate(cert, spec, args.out or f"evals_results/{args.case}/session")
    print(f"[certifier] CERTIFIED  m(FN)={cert['m']:.3f}  m_fp={cert['m_fp']:.3f}")
    return 0


def run_gcp_pin(args):
    token = args.token or Path(args.token_file).read_text().strip()
    verifier = publish_gcp_attestation(token, bytes.fromhex(args.verify_key), Path(args.out))
    print(f"[gcp-pin] pinned measurement={verifier.measurement()} -> {args.out}")
    return 0


def _pipe(source, destination, state, replacement=None):
    for raw in source.makefile("rb"):
        message = json.loads(raw)
        if message.get("type") == COMMIT:
            state["pk"] = bytes.fromhex(message["pk_ch"])
        elif replacement and message.get("type") == TEST:
            ciphertext = sealing.seal_bytes(state["pk"], replacement)
            raw = json.dumps({"type": TEST, "c_E": ciphertext}).encode() + b"\n"
        destination.sendall(raw)
    try:
        destination.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def run_relay(args):
    replacement = (Path(args.replace_environment).read_bytes()
                   if args.replace_environment else None)
    server = socket.create_server((args.listen_host, args.listen_port))
    certifier, _ = server.accept()
    enclave, state = _dial(args.workload_host, args.workload_port), {}
    threads = [threading.Thread(target=_pipe, args=(enclave, certifier, state)),
               threading.Thread(target=_pipe, args=(certifier, enclave, state, replacement))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    certifier.close(); enclave.close(); server.close()
    return 0


def run_verify(args):
    cert = json.loads(Path(args.cert).read_text())
    spec = json.loads(Path(args.spec).read_text())
    return 0 if verify_certificate(cert, spec=spec) else 1


def run_e2e(args):
    initialize_local_attestation()
    environment = _environment(args)
    spec, packed = prepare_environment(environment)
    process_env = {**os.environ, "PYTHONPATH": str(REPO)}
    with tempfile.TemporaryDirectory(prefix="ct-") as temporary:
        spec_path, environment_path = Path(temporary) / "spec.json", Path(temporary) / "E"
        spec_path.write_text(json.dumps(spec)); environment_path.write_bytes(packed)
        component_args = _component_args(args)
        workload = subprocess.Popen([
            sys.executable, "-m", "evals_runner", "protocol-integrity-exp", "workload",
            "--case", args.case, "--spec", str(spec_path), *component_args], env=process_env)
        relay_command = [sys.executable, "-m", "evals_runner", "protocol-integrity-exp", "relay"]
        if args.tamper_environment:
            relay_command += ["--replace-environment", str(environment_path)]
        relay = subprocess.Popen(relay_command, env=process_env)
        connection, result = _dial("127.0.0.1", RELAY_PORT), 0
        try:
            cert = run_certifier(connection, environment, spec)
            write_certificate(cert, spec, args.out or f"evals_results/{args.case}/session")
            print(f"[certifier] CERTIFIED  m(FN)={cert['m']:.3f}  m_fp={cert['m_fp']:.3f}")
            verify_certificate(cert, spec=spec)
            if args.case == "mock":
                quality = run_tap_quality(lambda: mock_agent(args.data), mock_tap,
                                          mock_environment(args.data, args.n, args.seed),
                                          seed=args.seed,
                                          build_comparator=build_comparator)
                print(f"separation (FN)      : {quality['comparator_minus_tap']['m']:+.3f}")
        except CertifierAbort as error:
            print(f"[certifier] ABORT: {error}")
            result = 1
        finally:
            connection.close()
            for process in (relay, workload):
                process.terminate(); process.wait(timeout=5)
        return result


def _component_args(args):
    return ["--data", args.data]
